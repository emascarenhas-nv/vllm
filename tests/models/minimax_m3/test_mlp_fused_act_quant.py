# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMaxM3MLP and MiniMaxM3DSparkMLP feed their MXFP8 down_proj from the fused
clamped-SwiGLU + MXFP8 kernel and match the unfused path bit for bit."""

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
@pytest.mark.parametrize("num_tokens", [1, 8, 200])
@torch.inference_mode()
def test_mlp_fuses_act_quant(
    dist_init, default_vllm_config, monkeypatch, kind: str, num_tokens: int
) -> None:
    from vllm.model_executor.layers.activation import SiluAndMulWithClamp
    from vllm.model_executor.layers.fusion import fused_act_quant
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )

    key = (SiluAndMulWithClamp, kMxfp8Dynamic)
    producer = fused_act_quant._FUSED_ACT_QUANT.get(key)
    assert producer is not None, "silu_and_mul_with_clamp_mxfp8_quant not built"
    calls = []

    def spy(x, linear, act_fn):
        calls.append(tuple(x.shape))
        return producer(x, linear, act_fn)

    monkeypatch.setitem(fused_act_quant._FUSED_ACT_QUANT, key, spy)

    torch.manual_seed(0)
    mlp = _make_mlp(kind)
    x = torch.randn(num_tokens, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    out = mlp(x)
    assert calls == [(num_tokens, 2 * INTERMEDIATE_SIZE)]

    gate_up, _ = mlp.gate_up_proj(x)
    expected, _ = mlp.down_proj(mlp.act_fn.forward_cuda(gate_up))
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
