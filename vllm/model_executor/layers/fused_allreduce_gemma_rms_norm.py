# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual fusion of tensor-parallel all-reduce with the following GemmaRMSNorm.

Under tensor parallelism a ``RowParallelLinear`` (e.g. attention ``o_proj``)
produces a per-rank partial sum that is all-reduced, and the result is then fed
into a ``GemmaRMSNorm`` that adds the residual and normalizes. flashinfer ships a
kernel that fuses all-reduce + residual-add + RMSNorm into a single launch; this
helper drives it directly (no torch.compile pass) for models that run eager.

Scope: attention output. ``maybe_fused_allreduce_gemma_rms_norm_quant`` also
hands the next linear its pre-quantized MXFP8 input (a QuantizedActivation) from
the same kernel when that linear consumes one; it is the all-reduce counterpart
of ``maybe_fused_norm_quant``. When the flashinfer fast path is not applicable
(TP==1, flashinfer/NVSwitch unavailable, unsupported dtype, or an oversize
batch) it falls back to ``all_reduce`` + ``GemmaRMSNorm``, which is numerically
identical to the unfused model path.
"""

import torch

from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    get_input_quant_key,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

MiB = 1024 * 1024

# flashinfer fused all-reduce + RMSNorm is wired as a registered custom op in
# allreduce_rms_fusion; both that op and the workspace helpers only exist when
# flashinfer.comm.allreduce_fusion is importable.
try:
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        flashinfer_trtllm_fused_allreduce_norm,
    )
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        flashinfer_comm,
        get_fi_ar_workspace,
    )

    _AR_RESIDUAL_RMS_NORM = (
        flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNorm
        if flashinfer_comm is not None
        else None
    )
    # MXFP8 (F8_128x4 swizzled scales) output patterns of the trtllm kernel;
    # None with flashinfer builds that predate them.
    _fi_patterns = getattr(flashinfer_comm, "AllReduceFusionPattern", None)
    _AR_RESIDUAL_RMS_NORM_MXFP8 = getattr(
        _fi_patterns, "kARResidualRMSNormMXFP8Quant", None
    )
    _AR_RESIDUAL_RMS_NORM_OUT_MXFP8 = getattr(
        _fi_patterns, "kARResidualRMSNormOutMXFP8Quant", None
    )
except (ImportError, AttributeError):
    flashinfer_trtllm_fused_allreduce_norm = None  # type: ignore[assignment]
    get_fi_ar_workspace = None  # type: ignore[assignment]
    _AR_RESIDUAL_RMS_NORM = None
    _AR_RESIDUAL_RMS_NORM_MXFP8 = None
    _AR_RESIDUAL_RMS_NORM_OUT_MXFP8 = None


_FI_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _max_token_num(tp_size: int, hidden_size: int, dtype: torch.dtype) -> int | None:
    """Workspace token budget for flashinfer fused all-reduce, or None if the
    current world size / device is unsupported. Mirrors ``FlashInferAllReduce``."""
    from vllm.config.compilation import PassConfig

    max_size_mb = PassConfig.default_fi_allreduce_fusion_max_size_mb().get(tp_size)
    if not max_size_mb:
        return None
    element_size = torch.tensor([], dtype=dtype).element_size()
    return int(max_size_mb * MiB) // (hidden_size * element_size)


def _can_use_flashinfer(
    hidden_states: torch.Tensor, tp_size: int, require_trtllm: bool = False
) -> tuple[bool, int]:
    """Whether the flashinfer fused path applies; returns (ok, max_token_num).
    ``require_trtllm``: only the trtllm workspace backend (quantized patterns)."""
    if (
        flashinfer_trtllm_fused_allreduce_norm is None
        or get_fi_ar_workspace is None
        or _AR_RESIDUAL_RMS_NORM is None
    ):
        return False, 0
    if (
        not hidden_states.is_cuda
        or hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or hidden_states.dtype not in _FI_SUPPORTED_DTYPES
    ):
        return False, 0

    num_tokens, hidden_size = hidden_states.shape
    max_token_num = _max_token_num(tp_size, hidden_size, hidden_states.dtype)
    if max_token_num is None or num_tokens > max_token_num:
        return False, 0

    # Lazily create / fetch the (globally cached) workspace; returns None on
    # GPUs without NVSwitch, in which case we fall back gracefully.
    workspace = get_fi_ar_workspace(
        world_size=tp_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_token_num,
        hidden_dim=hidden_size,
        dtype=hidden_states.dtype,
        group=get_tp_group().cpu_group,
    )
    if workspace is None:
        return False, 0
    if require_trtllm and getattr(workspace, "backend", None) != "trtllm":
        return False, 0
    # The token-count bound above uses the whole workspace budget, but a backend
    # may use only a fraction of it per call (mnnvl rotates through three Lamport
    # buffers). Ask the workspace so we don't admit tensors that the kernel will
    # reject with "The buffer size in the given workspace is insufficient".
    if not workspace.is_buffer_size_sufficient(
        tp_size=tp_size,
        num_tokens=num_tokens,
        hidden_dim=hidden_size,
        dtype=hidden_states.dtype,
    ):
        return False, 0
    return True, max_token_num


def _can_use_aiter_fused_ar_rms(hidden_states: torch.Tensor) -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm._aiter_ops import rocm_aiter_ops

    if not rocm_aiter_ops.is_custom_all_reduce_enabled():
        return False
    if (
        hidden_states.dim() != 2
        or not hidden_states.is_contiguous()
        or hidden_states.dtype not in _FI_SUPPORTED_DTYPES
    ):
        return False
    aiter_ar = rocm_aiter_ops.get_aiter_allreduce()
    if aiter_ar is None or aiter_ar.disabled:
        return False
    total_bytes = hidden_states.numel() * hidden_states.element_size()
    if total_bytes > aiter_ar.effective_max_size():
        return False
    if not aiter_ar.should_custom_ar(hidden_states):
        return False
    return aiter_ar.use_1stage_fused_ar_rms(hidden_states)


def fused_allreduce_gemma_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce ``hidden_states`` + add ``residual`` + GemmaRMSNorm, fused.

    ``hidden_states`` is the per-rank *partial* (un-reduced) output of a
    row-parallel linear; ``norm`` is the GemmaRMSNorm applied right after.
    Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        # No all-reduce needed; identical to the unfused path.
        return norm(hidden_states, residual)

    ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
    if ok:
        norm_out = torch.empty_like(hidden_states)
        # With norm_out provided, the kernel writes the new residual
        # (all_reduce(hidden_states) + residual) into the hidden_states buffer
        # and the normalized result into norm_out, leaving `residual` untouched.
        flashinfer_trtllm_fused_allreduce_norm(
            allreduce_in=hidden_states,
            residual=residual,
            rms_gamma=norm.weight,
            rms_eps=norm.variance_epsilon,
            world_size=tp_size,
            weight_bias=1.0,  # GemmaRMSNorm-style
            launch_with_pdl=True,
            fp32_acc=True,
            max_token_num=max_token_num,
            pattern_code=_AR_RESIDUAL_RMS_NORM,
            norm_out=norm_out,
        )
        return norm_out, hidden_states

    if _can_use_aiter_fused_ar_rms(hidden_states):
        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()(
            input_=hidden_states,
            residual=residual,
            weight=norm.weight,
            epsilon=norm.variance_epsilon,
            gemma_norm=True,
        )

    # Fallback: explicit all-reduce + GemmaRMSNorm (matches the unfused model).
    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)


def _takes_swizzled_mxfp8_input(linear: torch.nn.Module) -> bool:
    """Whether ``linear`` consumes an MXFP8 QuantizedActivation with F8_128x4
    swizzled scales (as DeepSeek-V4.1's can_fuse_query_quant)."""
    if not current_platform.is_cuda():
        return False
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutedslMxfp8LinearKernel,
        FlashInferCutlassMxfp8LinearKernel,
    )

    # QuantKey does not encode scale layout; restrict the producer to the
    # consumers that accept F8_128x4 swizzled scales.
    return get_input_quant_key(linear) == kMxfp8Dynamic and type(
        getattr(getattr(linear, "quant_method", None), "kernel", None)
    ) in (FlashInferCutedslMxfp8LinearKernel, FlashInferCutlassMxfp8LinearKernel)


def _fused_allreduce_gemma_rms_norm_mxfp8(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: GemmaRMSNorm,
    linear: torch.nn.Module,
    need_norm_out: bool,
) -> tuple[torch.Tensor | None, QuantizedActivation, torch.Tensor] | None:
    """The trtllm fused all-reduce + residual + GemmaRMSNorm, also writing the
    MXFP8 input of ``linear``. None when that path does not apply."""
    tp_size = get_tensor_model_parallel_world_size()
    pattern = (
        _AR_RESIDUAL_RMS_NORM_OUT_MXFP8
        if need_norm_out
        else _AR_RESIDUAL_RMS_NORM_MXFP8
    )
    if (
        tp_size == 1
        or pattern is None
        or hidden_states.dim() != 2
        or hidden_states.shape[-1] % 32 != 0
        or not _takes_swizzled_mxfp8_input(linear)
    ):
        return None
    ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size, require_trtllm=True)
    if not ok:
        return None
    num_tokens, hidden_size = hidden_states.shape
    quant_out = torch.empty(
        (num_tokens, hidden_size),
        dtype=torch.float8_e4m3fn,
        device=hidden_states.device,
    )
    # F8_128x4: rows padded to 128, scale columns to 4.
    scale_out = torch.empty(
        (round_up(num_tokens, 128) * round_up(hidden_size // 32, 4),),
        dtype=torch.uint8,
        device=hidden_states.device,
    )
    norm_out = torch.empty_like(hidden_states) if need_norm_out else None
    flashinfer_trtllm_fused_allreduce_norm(
        allreduce_in=hidden_states,
        residual=residual,
        rms_gamma=norm.weight,
        rms_eps=norm.variance_epsilon,
        world_size=tp_size,
        weight_bias=1.0,  # GemmaRMSNorm-style
        launch_with_pdl=True,
        fp32_acc=True,
        max_token_num=max_token_num,
        pattern_code=pattern,
        norm_out=norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
    )
    logger.info_once(
        "Fused all-reduce + GemmaRMSNorm writes the MXFP8 input of %s.",
        type(linear).__name__,
    )
    quantized = QuantizedActivation(
        data=quant_out,
        scale=scale_out,
        orig_dtype=hidden_states.dtype,
        orig_shape=hidden_states.shape,
        quant_key=kMxfp8Dynamic,
    )
    # With norm_out the kernel writes the new residual into the hidden_states
    # buffer; without it, into ``residual`` in place.
    return norm_out, quantized, hidden_states if need_norm_out else residual


def maybe_fused_allreduce_gemma_rms_norm_quant(
    norm_fn: GemmaRMSNorm,
    x: torch.Tensor,
    linear: torch.nn.Module,
    residual: torch.Tensor,
) -> tuple[torch.Tensor | QuantizedActivation, torch.Tensor]:
    """``fused_allreduce_gemma_rms_norm`` for a normalized output whose only
    consumer is ``linear``: returns its MXFP8 QuantizedActivation from the same
    kernel when ``linear`` consumes one, else the bf16 output. Returns
    ``(hidden_states, new_residual)``, like ``maybe_fused_norm_quant``."""
    fused = _fused_allreduce_gemma_rms_norm_mxfp8(
        x, residual, norm_fn, linear, need_norm_out=False
    )
    if fused is not None:
        _, quantized, residual = fused
        return quantized, residual
    return fused_allreduce_gemma_rms_norm(x, residual, norm_fn)


def fused_allreduce_gemma_rms_norm_and_quant(
    norm_fn: GemmaRMSNorm,
    x: torch.Tensor,
    linear: torch.nn.Module,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, QuantizedActivation | None, torch.Tensor]:
    """``fused_allreduce_gemma_rms_norm`` whose bf16 output has other consumers
    besides ``linear``: returns ``(norm_out, quantized, new_residual)``, where
    ``quantized`` is ``linear``'s MXFP8 input from the same kernel, or None."""
    fused = _fused_allreduce_gemma_rms_norm_mxfp8(
        x, residual, norm_fn, linear, need_norm_out=True
    )
    if fused is not None:
        norm_out, quantized, residual = fused
        assert norm_out is not None
        return norm_out, quantized, residual
    norm_out, residual = fused_allreduce_gemma_rms_norm(x, residual, norm_fn)
    return norm_out, None, residual
