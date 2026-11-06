// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// silu_and_mul_with_clamp fused with dynamic MXFP8 quantization (e4m3 data,
// F8_128x4 swizzled UE8M0 scales). It repeats the roundings of
// silu_and_mul_with_clamp (csrc/activation_kernels.cu) and FlashInfer's
// mxfp8_quantize, so the output is bit-exact with the two ops back to back.

#include <cuda_fp8.h>

#include <algorithm>
#include <climits>
#include <map>
#include <mutex>

#include <torch/csrc/stable/tensor.h>

#include "../../cuda_vec_utils.cuh"
#include "../../dispatch_utils.h"
#include "../../torch_utils.h"

namespace vllm {
namespace mxfp8_act_quant {

// One UE8M0 scale per 32 elements; the F8_128x4 layout packs the scales of
// four consecutive blocks of a row (one 128-column k tile) into one 4B word.
constexpr int kSfVecSize = 32;
constexpr int kColsPerWord = 128;
constexpr int kMaxThreadsPerBlock = 256;
// The tile kernel takes inputs of up to this many (row, k tile) items;
// gridDim.y is its k-tile count.
constexpr int64_t kTileKernelMaxItems = 512;
static_assert(kTileKernelMaxItems <= 65535);
// Larger inputs: 4 elements per lane (one warp per item) up to k4EltsMaxItems
// items, 8 up to k8EltsMaxItems, then 16 with 256-bit loads (else 8).
constexpr int64_t k4EltsMaxItems = 4096;
constexpr int64_t k8EltsMaxItems = 8192;
constexpr float kE4m3Max = 448.0f;

// FlashInfer's float_to_ue8m0_fast: the biased exponent of v rounded toward
// +inf (subnormals up to 2^-127 stay at 0), clamped to 254, and 0 for v <= 0.
__device__ __forceinline__ uint32_t float_to_ue8m0(float v) {
  uint32_t const bits = __float_as_uint(v);
  uint32_t const exp_biased = (bits >> 23) & 0xFFu;
  uint32_t const mantissa = bits & 0x7FFFFFu;
  bool const round_up =
      mantissa != 0u && !(exp_biased == 0u && mantissa <= 0x400000u);
  uint32_t const code = min(exp_biased + (round_up ? 1u : 0u), 254u);
  return v <= 0.0f ? 0u : code;
}

// FlashInfer's ue8m0_to_inv_scale_fast: exactly 2^(127 - code), with 0 for
// code 0 and for code 254 (whose inverse would be subnormal).
__device__ __forceinline__ float ue8m0_to_inv_scale(uint32_t code) {
  int const exp = max(254 - static_cast<int>(code), 0);
  return code == 0u ? 0.0f : __uint_as_float(static_cast<uint32_t>(exp) << 23);
}

// Index of the F8_128x4 word holding the four scales of (row, k_tile):
// [row / 128][k_tile][row % 32][(row % 128) / 32].
__device__ __forceinline__ int64_t sf_word_offset(int row, int k_tile,
                                                  int num_k_tiles) {
  return (static_cast<int64_t>(row >> 7) * num_k_tiles + k_tile) * 128 +
         (row & 31) * 4 + ((row >> 5) & 3);
}

// out_sf index of word i of the last 128-row tile (sf_word_offset inverted)
// if it holds a padding row's scales, which FlashInfer zeroes; else -1.
__device__ __forceinline__ int padding_scale_word(unsigned const i,
                                                  int const num_rows,
                                                  int const num_padded_rows,
                                                  int const num_k_tiles) {
  int const tile_row = num_padded_rows - 128;
  int const row = tile_row + static_cast<int>((i % 4) * 32 + (i % 128) / 4);
  return i < num_k_tiles * 128u && row >= num_rows
             ? tile_row * num_k_tiles + static_cast<int>(i)
             : -1;
}

// silu_and_mul_with_clamp's packed path for kPairs pairs, with its roundings
// to the input dtype. All expf are issued before the IEEE divides, whose
// slow-path branches would otherwise serialize them.
template <typename packed_t, int kPairs>
__device__ __forceinline__ void clamped_swiglu(packed_t const* gate2,
                                               packed_t const* up2,
                                               float const limit,
                                               float const alpha,
                                               float const beta, float* y) {
  float2 g[kPairs], u[kPairs], e[kPairs];
#pragma unroll
  for (int i = 0; i < kPairs; ++i) {
    g[i] = cast_to_float2(gate2[i]);
    u[i] = cast_to_float2(up2[i]);
    g[i].x = fminf(g[i].x, limit);
    g[i].y = fminf(g[i].y, limit);
    u[i].x = fmaxf(fminf(u[i].x, limit), -limit);
    u[i].y = fmaxf(fminf(u[i].y, limit), -limit);
    g[i] = cast_to_float2(cast_to_packed<packed_t>(g[i]));
  }
#pragma unroll
  for (int i = 0; i < kPairs; ++i) {
    e[i] = make_float2(expf(-g[i].x * alpha), expf(-g[i].y * alpha));
  }
  packed_t silu[kPairs];
#pragma unroll
  for (int i = 0; i < kPairs; ++i) {
    silu[i] = cast_to_packed<packed_t>(
        make_float2(g[i].x / (1.0f + e[i].x), g[i].y / (1.0f + e[i].y)));
  }
#pragma unroll
  for (int i = 0; i < kPairs; ++i) {
    float2 v = cast_to_float2(silu[i]);
    v.x *= u[i].x + beta;
    v.y *= u[i].y + beta;
    v = cast_to_float2(cast_to_packed<packed_t>(v));
    y[2 * i] = v.x;
    y[2 * i + 1] = v.y;
  }
}

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t const v) {
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
    return __bfloat162float(v);
  } else {
    return __half2float(v);
  }
}

// v rounded to nearest even in the 16-bit dtype, as a float.
template <typename scalar_t>
__device__ __forceinline__ float round_to(float const v) {
  if constexpr (std::is_same_v<scalar_t, __nv_bfloat16>) {
    return __bfloat162float(__float2bfloat16_rn(v));
  } else {
    return __half2float(__float2half_rn(v));
  }
}

// clamped_swiglu for one element, with the same roundings.
template <typename scalar_t>
__device__ __forceinline__ float clamped_swiglu_one(float gate, float up,
                                                    float const limit,
                                                    float const alpha,
                                                    float const beta) {
  gate = round_to<scalar_t>(fminf(gate, limit));
  up = fmaxf(fminf(up, limit), -limit);
  float const silu = round_to<scalar_t>(gate / (1.0f + expf(-gate * alpha)));
  return round_to<scalar_t>(silu * (up + beta));
}

// kEltsPerThread contiguous input elements of one lane, as packed pairs.
template <typename scalar_t, int kEltsPerThread>
struct alignas(kEltsPerThread * sizeof(scalar_t)) LaneVec {
  typename PackedTypeConverter<scalar_t>::Type elts[kEltsPerThread / 2];
};

template <typename scalar_t, int kEltsPerThread>
__device__ __forceinline__ void load_lane(LaneVec<scalar_t, kEltsPerThread>& v,
                                          scalar_t const* ptr) {
  constexpr int kBytes = kEltsPerThread * sizeof(scalar_t);
  if constexpr (kBytes == 32) {
    ld256(reinterpret_cast<u32x8_t&>(v), reinterpret_cast<u32x8_t const*>(ptr));
  } else if constexpr (kBytes == 16) {
    *reinterpret_cast<int4*>(&v) = __ldg(reinterpret_cast<int4 const*>(ptr));
  } else {
    static_assert(kBytes == 8);
    *reinterpret_cast<uint2*>(&v) = __ldg(reinterpret_cast<uint2 const*>(ptr));
  }
}

// A group of kLanesPerGroup lanes handles one (row, k tile) item, looping
// over rows with stride row_slots. Thread i of the grid also zeroes padding
// scale word i (row_slots >= kEltsPerThread gives every word a thread).
template <typename scalar_t, int kEltsPerThread>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
    silu_and_mul_clamp_mxfp8_quant_kernel(
        uint8_t* __restrict__ out,           // [num_rows, d] e4m3
        uint32_t* __restrict__ out_sf,       // F8_128x4 UE8M0, 4 per word
        scalar_t const* __restrict__ input,  // [num_rows, 2 * d]
        int64_t const input_row_stride, int const num_rows,
        int const num_padded_rows, int const d, int const num_k_tiles,
        int const row_slots, uint32_t const row_magic, int const row_shift,
        float const limit, float const alpha, float const beta) {
  using packed_t = typename PackedTypeConverter<scalar_t>::Type;
  using vec_t = LaneVec<scalar_t, kEltsPerThread>;
  constexpr int kPacked = kEltsPerThread / 2;
  constexpr int kLanesPerSf = kSfVecSize / kEltsPerThread;
  constexpr int kLanesPerGroup = kColsPerWord / kEltsPerThread;
  static_assert(kLanesPerGroup == 4 * kLanesPerSf && 32 % kLanesPerGroup == 0);

  // Index math ahead of the PDL wait; only the store has to follow it.
  int const pad_word =
      padding_scale_word(blockIdx.x * blockDim.x + threadIdx.x, num_rows,
                         num_padded_rows, num_k_tiles);
  int const lane_in_group = threadIdx.x % kLanesPerGroup;
  int const group =
      blockIdx.x * (blockDim.x / kLanesPerGroup) + threadIdx.x / kLanesPerGroup;
  // Groups past the last row slot (in the final block) have no work.
  if (group >= row_slots * num_k_tiles) return;
  // group / num_k_tiles by a multiply-high and a shift (see launch).
  int const row_start =
      num_k_tiles == 1
          ? group
          : static_cast<int>(
                __umulhi(static_cast<uint32_t>(group), row_magic) >> row_shift);
  int const k_tile = group - row_start * num_k_tiles;
  int const col = k_tile * kColsPerWord + lane_in_group * kEltsPerThread;
  bool const valid_col = col < d;
  // Group-wide mask for the shuffles: groups of a warp may sit on different
  // rows and take different branches.
  unsigned const group_mask =
      (kLanesPerGroup == 32 ? 0xffffffffu : ((1u << kLanesPerGroup) - 1u))
      << (threadIdx.x % 32 / kLanesPerGroup * kLanesPerGroup);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // PDL: the GEMM that produces `input` must have finished.
  cudaGridDependencySynchronize();
#endif

  // After the wait: earlier kernels may still use out_sf's memory until then.
  if (pad_word >= 0) out_sf[pad_word] = 0u;

  for (int row = row_start; row < num_rows; row += row_slots) {
    vec_t gate, up;
    if (valid_col) {
      scalar_t const* gate_ptr = input + row * input_row_stride + col;
      load_lane(gate, gate_ptr);
      load_lane(up, gate_ptr + d);
    } else {
      // Columns past d (only when d % 128 != 0): zero, so they add nothing to
      // the block max and their scale stays 0.
#pragma unroll
      for (int i = 0; i < kPacked; ++i) {
        gate.elts[i] = cast_to_packed<packed_t>(make_float2(0.0f, 0.0f));
        up.elts[i] = gate.elts[i];
      }
    }

    float y[kEltsPerThread];
    // Four pairs (eight expf) at a time; eight pairs would lower occupancy.
    constexpr int kPairs = kPacked < 4 ? kPacked : 4;
#pragma unroll
    for (int i = 0; i < kPacked; i += kPairs) {
      clamped_swiglu<packed_t, kPairs>(gate.elts + i, up.elts + i, limit, alpha,
                                       beta, y + 2 * i);
    }

    // NaN-ignoring block max, NaN only if every element is NaN, as FlashInfer.
    float amax = fabsf(y[0]);
#pragma unroll
    for (int i = 1; i < kEltsPerThread; ++i) amax = fmaxf(amax, fabsf(y[i]));
#pragma unroll
    for (int offset = 1; offset < kLanesPerSf; offset <<= 1) {
      amax = fmaxf(amax,
                   __shfl_xor_sync(group_mask, amax, offset, kLanesPerGroup));
    }

    uint32_t const code = float_to_ue8m0(amax * (1.0f / kE4m3Max));
    float const inv_scale = ue8m0_to_inv_scale(code);

    if (valid_col) {
      uint32_t packed_out[kEltsPerThread / 4];
#pragma unroll
      for (int i = 0; i < kEltsPerThread / 4; ++i) {
        // Saturate to +-448 before the round-to-nearest cast, as FlashInfer.
        float q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          q[j] = fmaxf(fminf(y[4 * i + j] * inv_scale, kE4m3Max), -kE4m3Max);
        }
        uint32_t const lo = __nv_cvt_float2_to_fp8x2(make_float2(q[0], q[1]),
                                                     __NV_SATFINITE, __NV_E4M3);
        uint32_t const hi = __nv_cvt_float2_to_fp8x2(make_float2(q[2], q[3]),
                                                     __NV_SATFINITE, __NV_E4M3);
        packed_out[i] = lo | (hi << 16);
      }
      uint8_t* out_ptr = out + static_cast<int64_t>(row) * d + col;
      if constexpr (kEltsPerThread == 16) {
        *reinterpret_cast<uint4*>(out_ptr) = make_uint4(
            packed_out[0], packed_out[1], packed_out[2], packed_out[3]);
      } else if constexpr (kEltsPerThread == 8) {
        *reinterpret_cast<uint2*>(out_ptr) =
            make_uint2(packed_out[0], packed_out[1]);
      } else {
        *reinterpret_cast<uint32_t*>(out_ptr) = packed_out[0];
      }
    }

    // Gather the four block scales of this k tile into the group's first lane.
    uint32_t word = code;
#pragma unroll
    for (int i = 1; i < 4; ++i) {
      word |=
          __shfl_down_sync(group_mask, code, i * kLanesPerSf, kLanesPerGroup)
          << (8 * i);
    }
    if (lane_in_group == 0) {
      out_sf[sf_word_offset(row, k_tile, num_k_tiles)] = word;
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Small inputs: a block of four warps per (row, k tile), one element per
// thread, to spread a tile's expf and divides over the SM's four schedulers.
// Each warp is one MXFP8 block and writes its byte of the scale word.
template <typename scalar_t>
__global__ void __launch_bounds__(kColsPerWord)
    silu_and_mul_clamp_mxfp8_quant_tile_kernel(
        uint8_t* __restrict__ out,           // [num_rows, d] e4m3
        uint32_t* __restrict__ out_sf,       // F8_128x4 UE8M0, 4 per word
        scalar_t const* __restrict__ input,  // [num_rows, 2 * d]
        int64_t const input_row_stride, int const num_rows,
        int const num_padded_rows, int const d, int const num_k_tiles,
        float const limit, float const alpha, float const beta) {
  int const row = blockIdx.x;
  int const k_tile = blockIdx.y;
  int const col = k_tile * kColsPerWord + threadIdx.x;
  bool const valid_col = col < d;
  scalar_t const* gate_ptr = input + row * input_row_stride + col;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  // Columns past d (only when d % 128 != 0) are zero, as in the kernel above.
  float gate = 0.0f, up = 0.0f;
  if (valid_col) {
    gate = to_float(__ldg(gate_ptr));
    up = to_float(__ldg(gate_ptr + d));
  }
  float const y = clamped_swiglu_one<scalar_t>(gate, up, limit, alpha, beta);

  float amax = fabsf(y);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset));
  }
  uint32_t const code = float_to_ue8m0(amax * (1.0f / kE4m3Max));

  if (valid_col) {
    float const q =
        fmaxf(fminf(y * ue8m0_to_inv_scale(code), kE4m3Max), -kE4m3Max);
    out[static_cast<int64_t>(row) * d + col] =
        __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
  }
  if (threadIdx.x % 32 == 0) {
    reinterpret_cast<uint8_t*>(
        out_sf)[sf_word_offset(row, k_tile, num_k_tiles) * 4 +
                threadIdx.x / 32] = code;
  }

  // The last row's blocks zero the padding rows' scales, one word per thread.
  if (row == num_rows - 1) {
    int const pad_row = num_padded_rows - 128 + threadIdx.x;
    if (pad_row >= num_rows) {
      out_sf[sf_word_offset(pad_row, k_tile, num_k_tiles)] = 0u;
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// cudaLaunchKernelEx with programmatic dependent launch (PDL) on SM90+.
template <typename... Params, typename... Args>
void launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block,
                cudaStream_t stream, Args... args) {
  cudaLaunchConfig_t config;
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = get_device_prop()->major >= 9 ? 1 : 0;
  cudaError_t const err = cudaLaunchKernelEx(&config, kernel, args...);
  STD_TORCH_CHECK(err == cudaSuccess, "silu_and_mul_with_clamp_mxfp8_quant: ",
                  cudaGetErrorString(err));
}

template <typename scalar_t, int kEltsPerThread>
void launch(int device, uint8_t* out, uint32_t* out_sf, scalar_t const* input,
            int64_t input_row_stride, int num_rows, int num_padded_rows, int d,
            int num_k_tiles, float limit, float alpha, float beta,
            cudaStream_t stream) {
  constexpr int kLanesPerGroup = kColsPerWord / kEltsPerThread;
  auto const kernel =
      silu_and_mul_clamp_mxfp8_quant_kernel<scalar_t, kEltsPerThread>;
  int const num_sms = get_device_prop()->multiProcessorCount;

  // Enough row slots to fill the GPU once, then each group loops over rows.
  static std::mutex mutex;
  static std::map<int, int> max_groups_by_device;
  int max_groups;
  {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = max_groups_by_device.find(device);
    if (it == max_groups_by_device.end()) {
      int blocks_per_sm = 0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel,
                                                    kMaxThreadsPerBlock, 0);
      it = max_groups_by_device
               .emplace(device, std::max(blocks_per_sm, 1) * num_sms *
                                    (kMaxThreadsPerBlock / kLanesPerGroup))
               .first;
    }
    max_groups = it->second;
  }
  // At least kEltsPerThread row slots, for a thread per padding scale word
  // (slots past num_rows then only zero those).
  int const row_slots =
      std::max(kEltsPerThread, std::min(num_rows, max_groups / num_k_tiles));
  int const num_groups = row_slots * num_k_tiles;

  // Small launches use smaller blocks so the work spreads over more SMs.
  // Rounded up without overflow: the thread count may be just below 2^31.
  int const per_sm = (num_groups * kLanesPerGroup - 1) / num_sms + 1;
  int const threads =
      std::clamp((per_sm + 31) / 32 * 32, 32, kMaxThreadsPerBlock);
  int const groups_per_block = threads / kLanesPerGroup;

  // group / num_k_tiles as umulhi(group, row_magic) >> row_shift, exact for
  // groups below 2^31 (as CUTLASS's FastDivmod).
  int ceil_log2 = 0;
  while ((int64_t{1} << ceil_log2) < num_k_tiles) ++ceil_log2;
  uint32_t row_magic = 0;
  int row_shift = 0;
  if (num_k_tiles > 1) {
    row_magic = static_cast<uint32_t>(
        ((uint64_t{1} << (31 + ceil_log2)) + num_k_tiles - 1) / num_k_tiles);
    row_shift = ceil_log2 - 1;
  }

  launch_pdl(kernel,
             dim3((num_groups + groups_per_block - 1) / groups_per_block),
             dim3(threads), stream, out, out_sf, input, input_row_stride,
             num_rows, num_padded_rows, d, num_k_tiles, row_slots, row_magic,
             row_shift, limit, alpha, beta);
}

}  // namespace mxfp8_act_quant
}  // namespace vllm

// out = MXFP8(silu_and_mul_with_clamp(input, limit, alpha, beta)).
// input: [..., 2 * d] bf16/fp16 (gate | up), at least 2-D, d a positive
// multiple of 32; a 2-D input may be row-strided. out: [..., d]
// float8_e4m3fn, contiguous. out_scale: contiguous uint8 UE8M0 scales in
// FlashInfer's F8_128x4 swizzled layout of round_up(rows, 128) x
// round_up(d / 32, 4); padding rows' and columns' scales are written as 0.
// input (data pointer and row stride) and out must be 16-byte aligned,
// out_scale 4-byte aligned.
void silu_and_mul_with_clamp_mxfp8_quant(torch::stable::Tensor& out,
                                         torch::stable::Tensor& out_scale,
                                         torch::stable::Tensor const& input,
                                         double limit, double alpha,
                                         double beta) {
  namespace q = vllm::mxfp8_act_quant;
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::BFloat16 ||
          input.scalar_type() == torch::headeronly::ScalarType::Half,
      "silu_and_mul_with_clamp_mxfp8_quant: input must be bf16 or fp16");
  STD_TORCH_CHECK(
      out.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "silu_and_mul_with_clamp_mxfp8_quant: out must be float8_e4m3fn");
  STD_TORCH_CHECK(
      out_scale.scalar_type() == torch::headeronly::ScalarType::Byte,
      "silu_and_mul_with_clamp_mxfp8_quant: out_scale must be uint8");
  STD_TORCH_CHECK(input.dim() >= 2 && input.stride(-1) == 1,
                  "silu_and_mul_with_clamp_mxfp8_quant: input must be at least "
                  "2D with a contiguous last dim");
  STD_TORCH_CHECK(input.dim() == 2 || input.is_contiguous(),
                  "silu_and_mul_with_clamp_mxfp8_quant: only a 2D input may "
                  "have strided rows");

  int64_t const d = input.size(-1) / 2;
  STD_TORCH_CHECK(d > 0 && input.size(-1) == 2 * d && d % q::kSfVecSize == 0,
                  "silu_and_mul_with_clamp_mxfp8_quant: input last dim must "
                  "be 2 * d with d a positive multiple of 32, got ",
                  input.size(-1));
  int64_t const num_rows = input.numel() / input.size(-1);
  int64_t const input_row_stride = input.dim() == 2 ? input.stride(0) : 2 * d;
  int64_t const num_k_tiles = (d + q::kColsPerWord - 1) / q::kColsPerWord;
  int64_t const num_padded_rows = (num_rows + 127) / 128 * 128;
  STD_TORCH_CHECK(
      out.is_contiguous() && out.size(-1) == d && out.numel() == num_rows * d,
      "silu_and_mul_with_clamp_mxfp8_quant: out must be a "
      "contiguous [..., d] tensor");
  STD_TORCH_CHECK(
      out_scale.is_contiguous() &&
          out_scale.numel() == num_padded_rows * num_k_tiles * 4,
      "silu_and_mul_with_clamp_mxfp8_quant: out_scale must hold the F8_128x4 "
      "swizzled scales of ",
      num_padded_rows, " padded rows x ", num_k_tiles * 4, " scale columns");
  // Row and scale word counters are 32-bit, and the row loop computes row
  // indices up to about twice the row count before it stops.
  STD_TORCH_CHECK(num_padded_rows <= INT_MAX / 2 &&
                      num_padded_rows * num_k_tiles <= INT_MAX && d <= INT_MAX,
                  "silu_and_mul_with_clamp_mxfp8_quant: input too large");
  STD_TORCH_CHECK(input.get_device_index() == out.get_device_index() &&
                      input.get_device_index() == out_scale.get_device_index(),
                  "silu_and_mul_with_clamp_mxfp8_quant: tensors must be on "
                  "the same device");
  if (num_rows == 0) return;

  auto const addr = [](torch::stable::Tensor const& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr());
  };
  int64_t const row_bytes = input_row_stride * input.element_size();
  STD_TORCH_CHECK(addr(input) % 16 == 0 && row_bytes % 16 == 0 &&
                      addr(out) % 16 == 0 && addr(out_scale) % 4 == 0,
                  "silu_and_mul_with_clamp_mxfp8_quant: input rows and out "
                  "must be 16-byte aligned, out_scale 4-byte aligned");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream(input.get_device_index());

  // 256-bit loads (16 elements per lane) on SM100+ when rows are 32B aligned.
  bool const use_256b = CUDART_VERSION >= 12090 &&
                        get_device_prop()->major >= 10 &&
                        addr(input) % 32 == 0 && row_bytes % 32 == 0;
  int64_t const items = num_rows * num_k_tiles;
  bool const use_tile_kernel = items <= q::kTileKernelMaxItems;
  int elts_per_thread = 16;
  if (items <= q::k4EltsMaxItems) {
    elts_per_thread = 4;
  } else if (!use_256b || items <= q::k8EltsMaxItems) {
    elts_per_thread = 8;
  }

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "silu_and_mul_with_clamp_mxfp8_quant", [&] {
        using cuda_t = typename vllm::CUDATypeConverter<scalar_t>::Type;
        auto* out_ptr = static_cast<uint8_t*>(out.data_ptr());
        auto* out_sf_ptr = static_cast<uint32_t*>(out_scale.data_ptr());
        auto const* input_ptr = static_cast<cuda_t const*>(input.data_ptr());
        if (use_tile_kernel) {
          q::launch_pdl(q::silu_and_mul_clamp_mxfp8_quant_tile_kernel<cuda_t>,
                        dim3(num_rows, num_k_tiles), dim3(q::kColsPerWord),
                        stream, out_ptr, out_sf_ptr, input_ptr,
                        input_row_stride, num_rows, num_padded_rows, d,
                        num_k_tiles, limit, alpha, beta);
          return;
        }
        auto const run = [&](auto ept) {
          q::launch<cuda_t, decltype(ept)::value>(
              input.get_device_index(), out_ptr, out_sf_ptr, input_ptr,
              input_row_stride, num_rows, num_padded_rows, d, num_k_tiles,
              limit, alpha, beta, stream);
        };
        using std::integral_constant;
        if (elts_per_thread == 16) {
          run(integral_constant<int, 16>{});
        } else if (elts_per_thread == 8) {
          run(integral_constant<int, 8>{});
        } else {
          run(integral_constant<int, 4>{});
        }
      });
}
