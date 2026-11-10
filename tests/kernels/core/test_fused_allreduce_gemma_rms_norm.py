# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the manual AllReduce + GemmaRMSNorm fusion used by MiniMax M3.

`fused_allreduce_gemma_rms_norm` must match the unfused model path, i.e.
`GemmaRMSNorm(all_reduce(partial), residual)`, both on the flashinfer fast
path (TP>1 with flashinfer + NVSwitch) and on the eager fallback (TP==1, or when
flashinfer is unavailable / the GPU has no NVSwitch).
`flashinfer_allreduce_gemma_rms_norm_mxfp8` must match it followed by the MXFP8
quantization the next linear would otherwise run.
"""

import pytest
import torch
from torch.multiprocessing import spawn

from tests.utils import ensure_current_vllm_config, init_test_distributed_environment
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _AR_RESIDUAL_RMS_NORM_MXFP8,
    _can_use_flashinfer,
    flashinfer_allreduce_gemma_rms_norm_mxfp8,
    fused_allreduce_gemma_rms_norm,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_random_seed


@ensure_current_vllm_config()
def _worker_fused_ar_norm(
    local_rank,
    world_size,
    port,
    num_tokens,
    hidden_size,
    dtype,
    seed,
    eps,
):
    """Per-rank worker: compare the fused helper vs all_reduce + GemmaRMSNorm."""
    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )

    # Norm weights are identical across ranks (replicated GemmaRMSNorm).
    set_random_seed(seed)
    norm = GemmaRMSNorm(hidden_size, eps=eps).cuda().to(dtype)
    with torch.no_grad():
        norm.weight.normal_(mean=0.0, std=0.1)

    # Residual is shared across ranks; the partial o_proj output differs per rank
    # (each rank holds a partial sum that all_reduce combines).
    torch.manual_seed(seed + 7)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    torch.manual_seed(seed + 1000 + local_rank)
    partial = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Reference: the unfused model path.
    reduced = tensor_model_parallel_all_reduce(partial.clone())
    ref_out, ref_res = norm(reduced, residual.clone())

    # Fused helper (flashinfer fast path when available, else fallback).
    out, res = fused_allreduce_gemma_rms_norm(partial.clone(), residual.clone(), norm)
    torch.accelerator.synchronize()

    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(res, ref_res, atol=2e-2, rtol=2e-2)

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="CUDA or ROCm required",
)
# world_size=1 exercises the TP==1 identity branch on a single GPU; >1 exercises
# the all_reduce + GemmaRMSNorm equivalence (flashinfer kernel or fallback).
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("num_tokens", [1, 128, 333])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_fused_allreduce_gemma_rms_norm(
    world_size,
    num_tokens,
    hidden_size,
    dtype,
    eps,
    seed,
):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    port = str(get_open_port())
    spawn(
        _worker_fused_ar_norm,
        args=(
            world_size,
            port,
            num_tokens,
            hidden_size,
            dtype,
            seed,
            eps,
        ),
        nprocs=world_size,
        join=True,
    )


def _swizzled_scale_rows(scale: torch.Tensor, m: int, h: int) -> torch.Tensor:
    """(m, h // 32) view of 128x4-swizzled MXFP8 scales (padding rows dropped)."""
    r = torch.arange(m, device=scale.device)[:, None]
    k = torch.arange(h // 32, device=scale.device)[None, :]
    num_k_tiles = (h // 32 + 3) // 4
    offset = (
        ((r >> 7) * num_k_tiles + (k >> 2)) * 512
        + (r & 31) * 16
        + ((r >> 5) & 3) * 4
        + (k & 3)
    )
    return scale.flatten()[offset]


def _e4m3_codes(q: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude E4M3 bytes as ordered ints (+0 == -0)."""
    b = q.view(torch.uint8).int()
    return torch.where(b >= 128, -(b & 0x7F), b)


def _assert_mxfp8_close(q, scale, ref_q, ref_scale, m, h):
    """Codes and scales equal, except isolated one-step differences where a
    normed value sits on a rounding boundary."""
    for got, ref in (
        (_e4m3_codes(q), _e4m3_codes(ref_q)),
        (
            _swizzled_scale_rows(scale, m, h).int(),
            _swizzled_scale_rows(ref_scale.view(torch.uint8), m, h).int(),
        ),
    ):
        diff = (got - ref).abs()
        assert diff.max().item() <= 1
        assert (diff != 0).float().mean().item() <= 1e-3


@ensure_current_vllm_config()
def _worker_fused_ar_norm_mxfp8(local_rank, world_size, port, seed, eps):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        world_size, 1, local_rank, port, local_rank=local_rank
    )
    dtype = torch.bfloat16
    # 6144: MiniMax-M3's hidden size. 1 and 8 tokens take the one-shot
    # all-reduce, 128 and 384 the two-shot one.
    for hidden_size in (2048, 6144):
        set_random_seed(seed)
        norm = GemmaRMSNorm(hidden_size, eps=eps).cuda().to(dtype)
        with torch.no_grad():
            norm.weight.normal_(mean=0.0, std=0.1)
        for num_tokens in (1, 8, 128, 384):
            torch.manual_seed(seed + 7)
            residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
            torch.manual_seed(seed + 1000 + local_rank)
            partial = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

            # Reference: the bf16 fused path, then the linear's own quantization.
            ref_out, ref_res = fused_allreduce_gemma_rms_norm(
                partial.clone(), residual.clone(), norm
            )
            ref_q, ref_scale = mxfp8_e4m3_quantize(ref_out, is_sf_swizzled_layout=True)

            applies = _can_use_flashinfer(partial, world_size)[0]
            for return_normed in (False, True):
                fused = flashinfer_allreduce_gemma_rms_norm_mxfp8(
                    partial.clone(), residual.clone(), norm, return_normed
                )
                assert (fused is not None) == applies
                if fused is None:
                    continue
                quantized, normed, res = fused
                torch.accelerator.synchronize()
                torch.testing.assert_close(res, ref_res, atol=0, rtol=0)
                if return_normed:
                    assert normed is not None
                    torch.testing.assert_close(normed, ref_out, atol=0, rtol=0)
                else:
                    assert normed is None
                assert quantized.orig_shape == partial.shape
                _assert_mxfp8_close(
                    quantized.data,
                    quantized.scale,
                    ref_q,
                    ref_scale,
                    num_tokens,
                    hidden_size,
                )

    cleanup_dist_env_and_memory()


@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="the flashinfer MXFP8 all-reduce fusion targets SM100+",
)
@pytest.mark.skipif(
    _AR_RESIDUAL_RMS_NORM_MXFP8 is None,
    reason="flashinfer without the MXFP8 all-reduce fusion patterns",
)
@pytest.mark.parametrize("world_size", [2, 4])
def test_flashinfer_allreduce_gemma_rms_norm_mxfp8(world_size):
    num_gpus = current_platform.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    spawn(
        _worker_fused_ar_norm_mxfp8,
        args=(world_size, str(get_open_port()), 42, 1e-6),
        nprocs=world_size,
        join=True,
    )
