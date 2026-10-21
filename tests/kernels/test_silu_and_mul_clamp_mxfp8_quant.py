# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SiluAndMulWithClamp + dynamic MXFP8 producer: bit-exact with the unfused
path (the CUDA silu_and_mul_with_clamp kernel, then the consumer's
mxfp8_e4m3_quantize with F8_128x4 swizzled scales)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
)
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(100)),
    reason="the swizzled MXFP8 reference quant needs SM100+",
)


class _Mxfp8Linear(torch.nn.Module):
    """Stands in for a linear on the FlashInfer cute-dsl MXFP8 kernel, which
    advertises kMxfp8Dynamic with F8_128x4 swizzled scales."""

    def __init__(self) -> None:
        super().__init__()
        self._input_quant_key = kMxfp8Dynamic
        kernel = object.__new__(FlashInferCutedslMxfp8LinearKernel)
        self.quant_method = SimpleNamespace(kernel=kernel)


def _act(limit: float, alpha: float, beta: float) -> SiluAndMulWithClamp:
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        return SiluAndMulWithClamp(limit, alpha, beta)


def _written_sf(scale: torch.Tensor, num_rows: int, k: int) -> torch.Tensor:
    """Scale bytes of rows < num_rows in the F8_128x4 layout. Padding rows of
    the last 128-row tile are not read by any GEMM row and are not compared."""
    k_tiles = (k // 32 + 3) // 4
    rows = torch.arange(num_rows, device=scale.device)[:, None]
    cols = torch.arange(k_tiles * 4, device=scale.device)[None, :]
    off = ((rows // 128) * k_tiles + cols // 4) * 512 + (rows % 32) * 16
    off = off + (rows % 128) // 32 * 4 + cols % 4
    return scale.view(-1)[off.reshape(-1)]


@pytest.mark.parametrize("num_rows", [1, 4, 17, 128, 129, 512, 1023, 1024, 4097, 16384])
@pytest.mark.parametrize("d", [768, 3072])
@pytest.mark.parametrize("params", [(7.0, 1.702, 1.0), (1.0e4, 1.0, 0.0)])
@pytest.mark.parametrize("magnitude", [1.0, 30.0])
@torch.inference_mode()
def test_silu_and_mul_clamp_mxfp8_quant_matches_unfused(num_rows, d, params, magnitude):
    torch.manual_seed(num_rows + d)
    act_fn = _act(*params)
    x = torch.randn(num_rows, 2 * d, dtype=torch.bfloat16, device="cuda") * magnitude
    x[0, :32] = 0.0  # an all-zero scale group
    ref_q, ref_sf = mxfp8_e4m3_quantize(act_fn(x), is_sf_swizzled_layout=True)

    qa = maybe_fused_act_quant(act_fn, x, _Mxfp8Linear())

    assert isinstance(qa, QuantizedActivation)
    assert qa.quant_key == kMxfp8Dynamic
    assert qa.orig_dtype == torch.bfloat16 and qa.orig_shape == (num_rows, d)
    assert qa.data.shape == (num_rows, d) and qa.scale.numel() == ref_sf.numel()
    assert torch.equal(qa.data.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(
        _written_sf(qa.scale, num_rows, d), _written_sf(ref_sf, num_rows, d)
    )


@pytest.mark.parametrize("unsupported", ["key", "kernel", "dtype", "layout", "tile"])
@torch.inference_mode()
def test_silu_and_mul_clamp_mxfp8_quant_falls_back(unsupported: str) -> None:
    act_fn = _act(7.0, 1.702, 1.0)
    linear = _Mxfp8Linear()
    x = torch.randn(17, 2 * 768, dtype=torch.bfloat16, device="cuda")
    if unsupported == "key":
        linear._input_quant_key = None
    elif unsupported == "kernel":
        linear.quant_method = SimpleNamespace(kernel=None)  # unknown scale layout
    elif unsupported == "dtype":
        x = x.to(torch.float16)
    elif unsupported == "layout":
        x = torch.randn(2 * 768, 17, dtype=torch.bfloat16, device="cuda").T
    else:
        x = x[:, : 2 * 704]  # 704 / 32 = 22 scale columns: not whole 128x4 tiles

    result = maybe_fused_act_quant(act_fn, x, linear)

    assert not isinstance(result, QuantizedActivation)
    torch.testing.assert_close(result, act_fn(x), rtol=0, atol=0)
