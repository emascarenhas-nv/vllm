# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 sparse-decode merge writing o_proj's MXFP8 input.

The MXFP8_OUT merge must produce what o_proj's own input quant produces on the
bf16 merged output: mxfp8_e4m3_quantize with F8_128x4 swizzled scales. Rows
past total_q (cudagraph padding) are not written.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    _merge_topk_attn_out_kernel,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="the swizzled MXFP8 reference quant needs SM100+",
)


def _scale_numel(num_rows: int, k: int) -> int:
    return round_up(num_rows, 128) * round_up(k // 32, 4)


def _sf_offsets(num_rows: int, k: int, device) -> torch.Tensor:
    """F8_128x4 byte offsets of the scales of rows < num_rows."""
    k_tiles = (k // 32 + 3) // 4
    rows = torch.arange(num_rows, device=device)[:, None]
    cols = torch.arange(k_tiles * 4, device=device)[None, :]
    off = ((rows // 128) * k_tiles + cols // 4) * 512 + (rows % 32) * 16
    return (off + (rows % 128) // 32 * 4 + cols % 4).reshape(-1)


def _merge(o_partial, lse_partial, output, mxfp8_out=None):
    num_chunks, total_q, num_heads, head_dim = o_partial.shape
    q_out, q_scale = mxfp8_out if mxfp8_out is not None else (output, output)
    _merge_topk_attn_out_kernel[(total_q, num_heads)](
        o_partial,
        lse_partial,
        output,
        q_out,
        q_scale,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_chunks,
        USE_PDL=False,
        MXFP8_OUT=mxfp8_out is not None,
    )


@pytest.mark.parametrize("total_q", [1, 4, 37, 56, 128, 200])
@pytest.mark.parametrize("pad_rows", [0, 3, 100])
@pytest.mark.parametrize("num_chunks", [1, 4, 16])
@pytest.mark.parametrize("num_heads", [16, 8])
@torch.inference_mode()
def test_merge_mxfp8_matches_unfused(total_q, pad_rows, num_chunks, num_heads):
    torch.manual_seed(total_q * 7 + pad_rows + num_chunks)
    device = "cuda"
    head_dim = 128
    o_partial = torch.randn(
        num_chunks, total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    # Wide dynamic range per head so scale groups get different exponents.
    o_partial *= torch.exp2(
        torch.randint(-20, 20, (1, total_q, num_heads, 1), device=device).float()
    ).bfloat16()
    lse_partial = torch.randn(
        num_chunks, total_q, num_heads, dtype=torch.float32, device=device
    )
    if num_chunks > 1:
        lse_partial[-1, ::3] = float("-inf")  # empty chunks

    k = num_heads * head_dim
    output = torch.empty(
        total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    _merge(o_partial, lse_partial, output)
    ref_q, ref_sf = mxfp8_e4m3_quantize(
        output.view(total_q, k), is_sf_swizzled_layout=True
    )

    num_rows = total_q + pad_rows
    q_out = torch.full((num_rows, k), 0x55, dtype=torch.uint8, device=device)
    q_scale = torch.full(
        (_scale_numel(num_rows, k),), 0xAB, dtype=torch.uint8, device=device
    )
    untouched = torch.full_like(output, 7.0)
    _merge(
        o_partial,
        lse_partial,
        untouched,
        mxfp8_out=(q_out.view(torch.float8_e4m3fn), q_scale),
    )

    assert torch.equal(untouched, torch.full_like(output, 7.0)), "bf16 output written"
    assert torch.equal(q_out[:total_q], ref_q.view(torch.uint8))
    assert torch.all(q_out[total_q:] == 0x55), "padding rows written"
    off = _sf_offsets(total_q, k, device)
    assert torch.equal(q_scale[off], ref_sf.view(-1)[off])


@torch.inference_mode()
def test_merge_mxfp8_zero_and_saturation():
    device = "cuda"
    num_heads, head_dim, total_q = 16, 128, 5
    o_partial = torch.zeros(
        1, total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    o_partial[0, 1] = 3.0e38  # near bf16 max
    o_partial[0, 2, :, ::7] = -1.0e-30  # tiny values
    o_partial[0, 3, 0, 0] = 448.0
    lse_partial = torch.zeros(1, total_q, num_heads, dtype=torch.float32, device=device)
    k = num_heads * head_dim
    output = torch.empty(
        total_q, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    _merge(o_partial, lse_partial, output)
    ref_q, ref_sf = mxfp8_e4m3_quantize(
        output.view(total_q, k), is_sf_swizzled_layout=True
    )
    q_out = torch.empty(total_q, k, dtype=torch.float8_e4m3fn, device=device)
    q_scale = torch.empty(_scale_numel(total_q, k), dtype=torch.uint8, device=device)
    _merge(o_partial, lse_partial, output, mxfp8_out=(q_out, q_scale))
    assert torch.equal(q_out.view(torch.uint8), ref_q.view(torch.uint8))
    off = _sf_offsets(total_q, k, device)
    assert torch.equal(q_scale[off], ref_sf.view(-1)[off])
