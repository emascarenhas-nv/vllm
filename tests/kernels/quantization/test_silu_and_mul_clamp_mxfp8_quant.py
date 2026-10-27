# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""silu_and_mul_with_clamp_mxfp8_quant must be bit-exact with
silu_and_mul_with_clamp followed by mxfp8_e4m3_quantize (FlashInfer)."""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.kernels.linear import Mxfp8LinearKernel, Mxfp8LinearLayerConfig
from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
    FlashInferCutedslMxfp8LinearKernel,
    FlashInferCutlassMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.marlin import MarlinMxfp8LinearKernel
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import round_up

if not (
    current_platform.is_cuda()
    and current_platform.has_device_capability(100)
    and has_flashinfer()
):
    pytest.skip(
        reason="the reference MXFP8 quantization is FlashInfer's (SM100+)",
        allow_module_level=True,
    )

# (limit, alpha, beta): MiniMax-M3's SwiGLU-OAI, clamped SiLU, and a limit that
# bf16 cannot represent.
ACT_PARAMS = [(7.0, 1.702, 1.0), (7.0, 1.0, 0.0), (6.9, 1.702, 1.0)]
# MiniMax-M3 per-rank intermediate sizes: dense MLP 12288 and shared expert 3072
# at TP 1-8, plus widths whose last 128-column scale tile is partial.
INTERMEDIATE_SIZES = [32, 96, 160, 384, 768, 1536, 3072, 6144, 12288]
NUM_TOKENS = [1, 3, 32, 127, 128, 129, 1000, 4100]
DTYPES = [torch.bfloat16, torch.float16]


def unfused(
    x: torch.Tensor, limit: float, alpha: float, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    d = x.shape[-1] // 2
    act = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul_with_clamp(act, x, limit, alpha, beta)
    return mxfp8_e4m3_quantize(act.reshape(-1, d), is_sf_swizzled_layout=True)


def fused(
    x: torch.Tensor, limit: float, alpha: float, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    d = x.shape[-1] // 2
    num_tokens = x.numel() // x.shape[-1]
    q = torch.empty((num_tokens, d), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(
        round_up(num_tokens, 128) * round_up(d // 32, 4),
        dtype=torch.uint8,
        device=x.device,
    )
    torch.ops._C.silu_and_mul_with_clamp_mxfp8_quant(q, s, x, limit, alpha, beta)
    return q, s


def assert_bitwise_equal(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    (q, s), (q_ref, s_ref) = actual, expected
    assert q.shape == q_ref.shape and s.shape == s_ref.shape
    assert q.dtype == q_ref.dtype == torch.float8_e4m3fn
    mismatched = (q.view(torch.uint8) != q_ref.view(torch.uint8)).sum().item()
    assert mismatched == 0, f"{mismatched} of {q.numel()} e4m3 values differ"
    mismatched = (s != s_ref).sum().item()
    assert mismatched == 0, f"{mismatched} of {s.numel()} scales differ"


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", INTERMEDIATE_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act_params", ACT_PARAMS)
@torch.inference_mode()
def test_matches_unfused(
    num_tokens: int, d: int, dtype: torch.dtype, act_params: tuple
) -> None:
    torch.manual_seed(num_tokens * 100003 + d)
    # Scaled so that both clamps and the e4m3 scale range are exercised.
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device="cuda") * 4
    assert_bitwise_equal(fused(x, *act_params), unfused(x, *act_params))


def all_gate_values(dtype: torch.dtype, repeats: int) -> torch.Tensor:
    """Every 16-bit gate pattern, `repeats` times, each time against a different
    permutation of all up patterns, as [repeats * 256, 2 * 256]."""
    d = 256
    gate = torch.arange(1 << 16, dtype=torch.int32, device="cuda").to(torch.int16)
    gen = torch.Generator(device="cuda").manual_seed(0)
    rows = []
    for _ in range(repeats):
        up = gate[torch.randperm(1 << 16, device="cuda", generator=gen)]
        rows.append(torch.cat([gate.view(-1, d), up.view(-1, d)], dim=-1))
    return torch.cat(rows).view(dtype)


# Every gate value through each kernel variant: the tile kernel (1 repeat) and
# 4, 8 and 16 elements per lane (8, 16 and 40 repeats).
@pytest.mark.parametrize("repeats", [1, 8, 16, 40])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act_params", ACT_PARAMS)
@torch.inference_mode()
def test_all_gate_values(dtype: torch.dtype, act_params: tuple, repeats: int) -> None:
    x = all_gate_values(dtype, repeats)
    assert_bitwise_equal(fused(x, *act_params), unfused(x, *act_params))


# Small inputs (64 rows x 2 column tiles here) run the four-warp tile kernel.
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("act_params", ACT_PARAMS)
@torch.inference_mode()
def test_all_gate_values_small_calls(dtype: torch.dtype, act_params: tuple) -> None:
    for x in all_gate_values(dtype, 1).split(64):
        assert_bitwise_equal(fused(x, *act_params), unfused(x, *act_params))


# The tile kernel (1) and 16 elements per lane (40), captured and replayed.
@pytest.mark.parametrize("repeats", [1, 40])
@torch.inference_mode()
def test_cuda_graph_capture(repeats: int) -> None:
    params = ACT_PARAMS[0]
    x = all_gate_values(torch.bfloat16, repeats)
    expected = unfused(x, *params)
    q = torch.empty((x.shape[0], 256), dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(round_up(x.shape[0], 128) * 8, dtype=torch.uint8, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.silu_and_mul_with_clamp_mxfp8_quant(q, s, x, *params)
    graph.replay()
    torch.accelerator.synchronize()
    assert_bitwise_equal((q, s), expected)


# 17 rows run the four-warp tile kernel, 257 rows the one-warp kernel.
@pytest.mark.parametrize("num_tokens", [17, 257])
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_special_values(dtype: torch.dtype, num_tokens: int) -> None:
    info = torch.finfo(dtype)
    specials = torch.tensor(
        [
            float("inf"),
            -float("inf"),
            float("nan"),
            0.0,
            -0.0,
            info.max,
            -info.max,
            info.tiny,
            -info.tiny,
            info.tiny / 4,
            -info.tiny / 4,
            6.9,
            7.0,
            7.1,
            -7.1,
            -60.0,
            1e-30,
        ],
        dtype=dtype,
        device="cuda",
    )
    torch.manual_seed(0)
    d = 384
    x = torch.randn(num_tokens, 2 * d, dtype=dtype, device="cuda")
    # Sprinkle the specials, and make whole 32-element blocks of them.
    idx = torch.randint(0, x.numel(), (4096,), device="cuda")
    x.view(-1)[idx] = specials[idx % specials.numel()]
    for row, value in enumerate(specials.tolist()):
        x[row, :32] = value  # gate block
        x[row, d : d + 32] = value  # its up block
    for act_params in ACT_PARAMS:
        assert_bitwise_equal(fused(x, *act_params), unfused(x, *act_params))


@torch.inference_mode()
def test_strided_and_3d_input() -> None:
    torch.manual_seed(0)
    d = 768
    wide = torch.randn(300, 2 * d + 64, dtype=torch.bfloat16, device="cuda") * 4
    x = wide[:, : 2 * d]
    assert not x.is_contiguous()
    params = ACT_PARAMS[0]
    assert_bitwise_equal(fused(x, *params), unfused(x.contiguous(), *params))

    x3 = torch.randn(3, 45, 2 * d, dtype=torch.bfloat16, device="cuda") * 4
    assert_bitwise_equal(fused(x3, *params), unfused(x3, *params))


@torch.inference_mode()
def test_empty_input() -> None:
    x = torch.empty(0, 2 * 768, dtype=torch.bfloat16, device="cuda")
    q, s = fused(x, *ACT_PARAMS[0])
    assert q.shape == (0, 768) and s.numel() == 0
    # A zero-width input is rejected, not divided by.
    x = torch.empty(4, 0, dtype=torch.bfloat16, device="cuda")
    q = torch.empty((4, 0), dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(0, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="positive multiple of 32"):
        torch.ops._C.silu_and_mul_with_clamp_mxfp8_quant(q, s, x, *ACT_PARAMS[0])


@torch.inference_mode()
def test_opcheck() -> None:
    num_tokens, d = 37, 384
    x = torch.randn(num_tokens, 2 * d, dtype=torch.bfloat16, device="cuda")
    q = torch.empty((num_tokens, d), dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(128 * (d // 32), dtype=torch.uint8, device="cuda")
    opcheck(
        torch.ops._C.silu_and_mul_with_clamp_mxfp8_quant,
        (q, s, x, *ACT_PARAMS[0]),
    )


def _linear_with_kernel(kernel_cls: type) -> torch.nn.Module:
    linear = torch.nn.Module()
    linear._input_quant_key = kMxfp8Dynamic
    linear.quant_method = SimpleNamespace(kernel=object.__new__(kernel_cls))
    return linear


@pytest.mark.parametrize(
    "kernel_cls",
    [FlashInferCutlassMxfp8LinearKernel, FlashInferCutedslMxfp8LinearKernel],
)
@torch.inference_mode()
def test_maybe_fused_act_quant(default_vllm_config, kernel_cls: type) -> None:
    act_fn = SiluAndMulWithClamp(swiglu_limit=7.0, alpha=1.702, beta=1.0)
    x = torch.randn(129, 2 * 768, dtype=torch.bfloat16, device="cuda") * 4
    qa = maybe_fused_act_quant(act_fn, x, _linear_with_kernel(kernel_cls))
    assert isinstance(qa, QuantizedActivation)
    assert qa.quant_key == kMxfp8Dynamic
    assert qa.orig_dtype == x.dtype and qa.orig_shape == (129, 768)
    assert_bitwise_equal((qa.data, qa.scale), unfused(x, 7.0, 1.702, 1.0))


@torch.inference_mode()
def test_maybe_fused_act_quant_falls_back(default_vllm_config) -> None:
    act_fn = SiluAndMulWithClamp(swiglu_limit=7.0, alpha=1.702, beta=1.0)
    x = torch.randn(16, 2 * 768, dtype=torch.bfloat16, device="cuda")
    expected = act_fn(x)

    # A consumer that does not take F8_128x4 swizzled scales.
    out = maybe_fused_act_quant(act_fn, x, _linear_with_kernel(MarlinMxfp8LinearKernel))
    assert not isinstance(out, QuantizedActivation)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)

    # A consumer whose output is also read unquantized by another branch.
    linear = _linear_with_kernel(FlashInferCutlassMxfp8LinearKernel)
    linear.requires_unquantized_input = True
    assert not isinstance(maybe_fused_act_quant(act_fn, x, linear), QuantizedActivation)

    # Inputs outside the kernel's support: fp32, a width that is not a multiple
    # of 32, and a non-contiguous input.
    linear = _linear_with_kernel(FlashInferCutlassMxfp8LinearKernel)
    narrow = torch.randn(16, 2 * 752, dtype=torch.bfloat16, device="cuda")
    for bad in (x.float(), narrow, x.t().contiguous().t()):
        assert not isinstance(
            maybe_fused_act_quant(act_fn, bad, linear), QuantizedActivation
        )


@pytest.mark.parametrize(
    "kernel_cls",
    [FlashInferCutlassMxfp8LinearKernel, FlashInferCutedslMxfp8LinearKernel],
)
@pytest.mark.parametrize("num_tokens", [1, 200])
@torch.inference_mode()
def test_feeds_flashinfer_mxfp8_linear(
    default_vllm_config, kernel_cls: type[Mxfp8LinearKernel], num_tokens: int
) -> None:
    supported, reason = kernel_cls.is_supported()
    if not supported:
        pytest.skip(reason)
    from flashinfer import SfLayout, mxfp8_quantize

    torch.manual_seed(0)
    n, k = 512, 768
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / 8
    weight_mxfp8, weight_scale = mxfp8_quantize(
        weight, sf_swizzle_layout=SfLayout.layout_linear
    )
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight_mxfp8, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        weight_scale.view(n, k // 32), requires_grad=False
    )
    kernel = kernel_cls(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)
    layer.quant_method = SimpleNamespace(kernel=kernel)
    layer._input_quant_key = kernel.input_quant_key()
    layer.params_dtype = torch.bfloat16

    act_fn = SiluAndMulWithClamp(swiglu_limit=7.0, alpha=1.702, beta=1.0)
    x = torch.randn(num_tokens, 2 * k, dtype=torch.bfloat16, device="cuda") * 4
    qa = maybe_fused_act_quant(act_fn, x, layer)
    assert isinstance(qa, QuantizedActivation)
    torch.testing.assert_close(
        kernel.apply_weights(layer, qa),
        kernel.apply_weights(layer, act_fn.forward_cuda(x)),
        atol=0,
        rtol=0,
    )
