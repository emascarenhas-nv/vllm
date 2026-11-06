# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark silu_and_mul_with_clamp_mxfp8_quant against SiluAndMulWithClamp
(custom op or torch.compile) followed by mxfp8_e4m3_quantize, at MiniMax-M3's
per-rank intermediate sizes."""

import itertools

import torch

from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import set_random_seed

NUM_TOKENS = [1, 4, 16, 64, 256, 1024, 4096, 16384]
INTERMEDIATE_SIZES = [384, 768, 1536, 3072, 6144, 12288]
LIMIT, ALPHA, BETA = 7.0, 1.702, 1.0  # MiniMax-M3 SwiGLU-OAI


@default_vllm_config()
def benchmark(num_tokens: int, d: int, provider: str) -> tuple[float, float, float]:
    set_random_seed(0)
    x = torch.randn(num_tokens, 2 * d, dtype=torch.bfloat16, device="cuda") * 4
    act_fn = SiluAndMulWithClamp(LIMIT, ALPHA, BETA)
    if provider == "fused":
        q = torch.empty((num_tokens, d), dtype=torch.float8_e4m3fn, device="cuda")
        s = torch.empty(
            round_up(num_tokens, 128) * round_up(d // 32, 4),
            dtype=torch.uint8,
            device="cuda",
        )

        def fn():
            torch.ops._C.silu_and_mul_with_clamp_mxfp8_quant(
                q, s, x, LIMIT, ALPHA, BETA
            )

    elif provider == "unfused":

        def fn():
            mxfp8_e4m3_quantize(act_fn.forward_cuda(x), is_sf_swizzled_layout=True)

    else:
        # Compile per shape, so earlier shapes neither make dims dynamic nor use
        # up dynamo's recompile limit.
        torch._dynamo.reset()
        compiled = torch.compile(act_fn.forward_native, dynamic=False)

        def fn():
            mxfp8_e4m3_quantize(compiled(x), is_sf_swizzled_layout=True)

    ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
        fn, quantiles=[0.5, 0.2, 0.8]
    )
    return 1000 * ms, 1000 * min_ms, 1000 * max_ms


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="Benchmark silu_and_mul_with_clamp_mxfp8_quant."
    )
    parser.add_argument("--num-tokens", type=int, nargs="+", default=NUM_TOKENS)
    parser.add_argument(
        "--intermediate-sizes", type=int, nargs="+", default=INTERMEDIATE_SIZES
    )
    args = parser.parse_args()

    perf_report = triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["num_tokens", "d"],
            x_vals=list(itertools.product(args.num_tokens, args.intermediate_sizes)),
            line_arg="provider",
            line_vals=["fused", "unfused", "unfused-compiled"],
            line_names=["Fused", "Unfused (custom op)", "Unfused (torch.compile)"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="us",
            plot_name="silu_and_mul_clamp_mxfp8_quant",
            args={},
        )
    )
    perf_report(benchmark).run(print_data=True)
