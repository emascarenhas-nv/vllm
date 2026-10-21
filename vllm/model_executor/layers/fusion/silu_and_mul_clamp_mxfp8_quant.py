# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Clamped SwiGLU (SiluAndMulWithClamp) fused with dynamic MXFP8 quantization.

Producer for consumers that advertise kMxfp8Dynamic (the FlashInfer MXFP8
linear kernels, F8_128x4 swizzled scales): one launch replaces the activation
and the consumer's own mxfp8_e4m3_quantize of the bf16 activation. The
activation mirrors the CUDA silu_and_mul_with_clamp kernel's rounding (gate
clamped and rounded to bf16, IEEE divide, bf16 activation times the fp32
clamped up, rounded to bf16) and the quant mirrors mxfp8_e4m3_quantize, so the
result is bit-exact with the unfused path.
"""

import torch

from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op

# (BLOCK_M, BLOCK_N, num_warps), from B200 sweeps at d=768. Decode batches
# (<= 512 rows) are fastest with one 256-column row per program; from 1024 rows,
# 4 rows per program halves the time (16384 rows: 51 -> 22 us).
_SMALL_M_CONFIG = (1, 256, 2)
_LARGE_M_CONFIG = (4, 256, 4)
_LARGE_M_THRESHOLD = 1024


def _config(num_rows: int) -> tuple[int, int, int]:
    return _LARGE_M_CONFIG if num_rows >= _LARGE_M_THRESHOLD else _SMALL_M_CONFIG


@triton.jit
def _silu_and_mul_clamp_mxfp8_quant_kernel(
    x_ptr,  # bf16 [num_rows, 2 * d]: gate | up
    q_ptr,  # e4m3 [num_rows, d]
    s_ptr,  # uint8 UE8M0 scales, F8_128x4 swizzled, 1D
    num_rows,
    d,
    stride_x,
    limit,
    alpha,
    beta,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows < num_rows)[:, None] & (cols < d)[None, :]
    x_row = x_ptr + rows[:, None].to(tl.int64) * stride_x
    g = tl.load(x_row + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_row + d + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    # silu_and_mul_with_clamp: gate = bf16(min(gate, limit)), up stays fp32.
    g = tl.minimum(g, limit).to(tl.bfloat16).to(tl.float32)
    u = tl.minimum(tl.maximum(u, -limit), limit)
    act = tl.math.div_rn(g, 1.0 + tl.extra.cuda.libdevice.exp(-g * alpha))
    act = act.to(tl.bfloat16).to(tl.float32)
    y = (act * (u + beta)).to(tl.bfloat16).to(tl.float32)

    # MXFP8: UE8M0 = ceil_pow2(amax / 448) per 32 elements, saturated to [0, 254].
    NUM_GROUPS: tl.constexpr = BLOCK_N // 32
    yg = tl.reshape(y, (BLOCK_M, NUM_GROUPS, 32))
    amax = tl.max(tl.abs(yg), axis=2)
    norm = amax * (1.0 / 448.0)
    bits = norm.to(tl.uint32, bitcast=True)
    exp_biased = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    bump = tl.where(
        (mantissa != 0) & ~((exp_biased == 0) & (mantissa <= 0x400000)), 1, 0
    )
    sf = tl.minimum(exp_biased + bump, 254)
    sf = tl.where(norm <= 0.0, 0, sf)
    # Exact inverse scale 2^(127 - sf); 0 when sf == 0.
    inv_scale = tl.where(sf == 0, 0.0, ((254 - sf) << 23).to(tl.float32, bitcast=True))
    q = yg * inv_scale[:, :, None]
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    q = tl.reshape(q, (BLOCK_M, BLOCK_N))
    q_row = q_ptr + rows[:, None].to(tl.int64) * d
    tl.store(q_row + cols[None, :], q.to(q_ptr.dtype.element_ty), mask=mask)

    # F8_128x4: [row // 128, sf_col // 4, row % 32, row % 128 // 32, sf_col % 4].
    num_k_tiles = (d // 32 + 3) // 4
    sf_cols = pid_n * NUM_GROUPS + tl.arange(0, NUM_GROUPS)
    r = rows[:, None]
    c = sf_cols[None, :]
    sf_off = (
        (r // 128) * (num_k_tiles * 512)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (c % 4)
    )
    sf_mask = (rows < num_rows)[:, None] & (sf_cols < d // 32)[None, :]
    tl.store(s_ptr + sf_off, sf.to(tl.uint8), mask=sf_mask)


def _silu_and_mul_clamp_mxfp8_quant_impl(
    x: torch.Tensor,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
    limit: float,
    alpha: float,
    beta: float,
) -> None:
    num_rows, d = q_out.shape
    if num_rows == 0:
        return
    use_pdl = current_platform.is_arch_support_pdl()
    block_m, block_n, num_warps = _config(num_rows)
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(d, block_n))
    _silu_and_mul_clamp_mxfp8_quant_kernel[grid](
        x,
        q_out,
        scale_out,
        num_rows,
        d,
        x.stride(0),
        limit,
        alpha,
        beta,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        USE_PDL=use_pdl,
        num_warps=num_warps,
        **({"launch_pdl": True} if use_pdl else {}),
    )


def _silu_and_mul_clamp_mxfp8_quant_fake(
    x: torch.Tensor,
    q_out: torch.Tensor,
    scale_out: torch.Tensor,
    limit: float,
    alpha: float,
    beta: float,
) -> None:
    return None


direct_register_custom_op(
    op_name="silu_and_mul_clamp_mxfp8_quant",
    op_func=_silu_and_mul_clamp_mxfp8_quant_impl,
    mutates_args=["q_out", "scale_out"],
    fake_impl=_silu_and_mul_clamp_mxfp8_quant_fake,
)


def silu_and_mul_clamp_mxfp8_quant_supported(
    act_fn: torch.nn.Module, x: torch.Tensor, linear: LinearBase
) -> bool:
    """A consumer of F8_128x4 swizzled scales, and a bf16 2D row-major input
    whose d fills whole 128x4 scale tiles (no padding scale columns to zero)."""
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )

    # QuantKey does not encode scale layout (as DeepSeek-V4.1's
    # can_fuse_query_quant): only these kernels take F8_128x4 swizzled scales.
    kernel = getattr(getattr(linear, "quant_method", None), "kernel", None)
    return (
        type(kernel)
        in (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel)
        and isinstance(act_fn, SiluAndMulWithClamp)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.stride(-1) == 1
        and (x.shape[-1] // 2) % 128 == 0
    )


def silu_and_mul_clamp_mxfp8_quant(
    x: torch.Tensor, linear: LinearBase, act_fn: SiluAndMulWithClamp
) -> QuantizedActivation:
    """act_fn(x) as the dynamic MXFP8 input of ``linear``: e4m3 [num_rows, d]
    data and 1D F8_128x4 swizzled UE8M0 scales."""
    num_rows, d = x.shape[0], x.shape[-1] // 2
    q_out = torch.empty((num_rows, d), dtype=torch.float8_e4m3fn, device=x.device)
    # Rows padded to 128 and scale columns to 4, as mxfp8_e4m3_quantize returns.
    scale_out = torch.empty(
        (round_up(num_rows, 128) * round_up(d // 32, 4),),
        dtype=torch.uint8,
        device=x.device,
    )
    torch.ops.vllm.silu_and_mul_clamp_mxfp8_quant(
        x, q_out, scale_out, act_fn.swiglu_limit, act_fn.alpha, act_fn.beta
    )
    return QuantizedActivation(
        data=q_out,
        scale=scale_out,
        orig_dtype=x.dtype,
        orig_shape=q_out.shape,
        quant_key=kMxfp8Dynamic,
    )
