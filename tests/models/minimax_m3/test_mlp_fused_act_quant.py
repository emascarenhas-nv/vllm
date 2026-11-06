# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMaxM3MLP and MiniMaxM3DSparkMLP feed a FlashInfer MXFP8 down_proj from the
fused clamped-SwiGLU + MXFP8 kernel, matching the unfused path bit for bit, and
otherwise fall back to the plain activation."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import set_default_torch_dtype

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_flashinfer()
    ),
    reason="The fused producer feeds FlashInfer's SM100 MXFP8 linears",
)

HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 768
# Below MiniMaxM3MLP's fused GEMM1 threshold.
NUM_TOKENS = 200


def _make_mlp(kind: str) -> torch.nn.Module:
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    config = MiniMaxM3TextConfig(
        hidden_size=HIDDEN_SIZE, dspark_ffn_hidden_size=INTERMEDIATE_SIZE
    )
    quant_config = ModelOptMxFp8Config.from_config(
        {
            "quant_method": "mxfp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, 32],
            "ignored_layers": [],
        }
    )
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        if kind == "target":
            from vllm.models.minimax_m3.nvidia.model import MiniMaxM3MLP

            mlp = MiniMaxM3MLP(
                config,
                INTERMEDIATE_SIZE,
                quant_config=quant_config,
                prefix="model.layers.0.mlp",
            )
        else:
            from vllm.models.minimax_m3.common.dspark import MiniMaxM3DSparkMLP

            mlp = MiniMaxM3DSparkMLP(config, quant_config, prefix="model.layers.0.mlp")
    for proj in (mlp.gate_up_proj, mlp.down_proj):
        weight, scale = _mxfp8_e4m3_quantize_torch(
            torch.randn(proj.weight.shape, device="cuda") / 16
        )
        proj.weight.data.copy_(weight)
        proj.weight_scale.data.copy_(scale)
        proj.quant_method.process_weights_after_loading(proj)
    return mlp


@pytest.mark.parametrize("kind", ["target", "draft"])
@pytest.mark.parametrize(
    "backend", ["flashinfer_cutedsl", "flashinfer_cutlass", "torch"]
)
@torch.inference_mode()
def test_mlp_fused_act_quant(dist_init, kind: str, backend: str) -> None:
    from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
        TorchMxFp8LinearKernel,
    )
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
        get_input_quant_key,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    kernel_cls = {
        "flashinfer_cutedsl": FlashInferCutedslMxfp8LinearKernel,
        "flashinfer_cutlass": FlashInferCutlassMxfp8LinearKernel,
        "torch": TorchMxFp8LinearKernel,
    }[backend]
    supported, reason = kernel_cls.is_supported()
    if not supported:
        pytest.skip(reason)
    assert hasattr(torch.ops._C, "silu_and_mul_with_clamp_mxfp8_quant")

    torch.manual_seed(0)
    config = VllmConfig(kernel_config=KernelConfig(linear_backend=backend))
    with set_current_vllm_config(config):
        mlp = _make_mlp(kind)
        assert type(mlp.down_proj.quant_method.kernel) is kernel_cls
        # All three advertise kMxfp8Dynamic; the producer feeds only FlashInfer's.
        assert get_input_quant_key(mlp.down_proj) == kMxfp8Dynamic
        x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
        gate_ups, down_inputs = [], []
        with (
            mlp.gate_up_proj.register_forward_hook(
                lambda _module, _inputs, output: gate_ups.append(output[0])
            ),
            mlp.down_proj.register_forward_pre_hook(
                lambda _module, inputs: down_inputs.append(inputs[0])
            ),
        ):
            out = mlp(x)
        (gate_up,), (down_input,) = gate_ups, down_inputs

        if backend == "torch":
            assert not isinstance(down_input, QuantizedActivation)
            act = mlp.act_fn(gate_up)
        else:
            assert isinstance(down_input, QuantizedActivation)
            # Bit-exact with the CUDA activation and the linear's own quantization.
            act = mlp.act_fn.forward_cuda(gate_up)
            q_ref, s_ref = mxfp8_e4m3_quantize(act, is_sf_swizzled_layout=True)
            torch.testing.assert_close(
                down_input.data.view(torch.uint8),
                q_ref.view(torch.uint8),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                down_input.scale.view(torch.uint8),
                s_ref.view(torch.uint8),
                atol=0,
                rtol=0,
            )
        expected, _ = mlp.down_proj(act)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.parametrize("unsupported", ["fp32", "non_contiguous", "1d"])
@torch.inference_mode()
def test_unsupported_input_falls_back(
    dist_init, default_vllm_config, unsupported: str
) -> None:
    from vllm.model_executor.layers.fusion.fused_act_quant import maybe_fused_act_quant
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

    mlp = _make_mlp("target")
    x = torch.randn(16, 2 * INTERMEDIATE_SIZE, device="cuda", dtype=torch.bfloat16)
    assert isinstance(
        maybe_fused_act_quant(mlp.act_fn, x, mlp.down_proj), QuantizedActivation
    )
    x = {
        "fp32": x.float(),
        "non_contiguous": x.t().contiguous().t(),
        "1d": x[0],
    }[unsupported]
    out = maybe_fused_act_quant(mlp.act_fn, x, mlp.down_proj)
    assert not isinstance(out, QuantizedActivation)
    torch.testing.assert_close(out, mlp.act_fn(x), atol=0, rtol=0)
