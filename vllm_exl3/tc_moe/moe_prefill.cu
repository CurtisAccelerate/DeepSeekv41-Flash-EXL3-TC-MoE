// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Codec adapted from turboderp-org/exllamav3 (MIT); the MIT grant is in codec.cuh.
//
// EXL3 routed-expert MoE on the tensor cores for prefill-sized calls.
// DS V4.1 Flash port of Morrowmake vllm-cmp170hx@18305948ac: hidden 5120,
// intermediate 640 or 512, bits_x2 6 (K3) or 4 (K2), mul1 codebook (id 2).
// The host switches on (intermediate, bits_x2). Calls larger than
// VLLM_EXL3_TC_PREFILL_ROWS (default 2048) run as row sub-chunks on one
// workspace; rows are independent, so the result matches one call.
//
// Arithmetic: the input and activation stages and the combine's per-route
// transform are FP32; the combine sums the routes' weighted products in FP64
// and rounds once to FP32, then to BF16. The expert GEMMs run m16n8k16 FP16
// tensor-core MMAs on exactly decoded weights and a hi/lo FP16 split of the
// activations. Everything is fixed-order and a row never depends on other rows.

#include "moe_prefill.h"
#include "common.h"
#include "codec.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace exl3 {
namespace moe_prefill {

// Hidden is fixed. Intermediate and bits_x2 are the instance template arguments.
constexpr int kHidden = 5120;
constexpr int kCodebook = 2;
// Rows per route tile (expert GEMM block).
constexpr int kTileRows = 64;

// Integers the route kernel writes into the workspace tail each call:
//   [0] local assignment count, [1] route tile count, [2..3] unused,
//   sorted[assignments]: local assignment ids grouped by local expert,
//   tiles[assignments] (16-byte aligned): {expert, first sorted position, rows, 0},
//   position[assignments]: sorted position of each assignment (-1: not local).
struct Layout {
  static constexpr int64_t kHeader = 4;
  __host__ __device__ static constexpr int64_t sorted_offset() { return kHeader; }
  __host__ __device__ static constexpr int64_t tiles_offset(int64_t assignments) {
    return kHeader + (assignments + 3) / 4 * 4;
  }
  __host__ __device__ static constexpr int64_t position_offset(int64_t assignments) {
    return tiles_offset(assignments) + 4 * assignments;
  }
  __host__ __device__ static constexpr int64_t ints(int64_t assignments) {
    return position_offset(assignments) + assignments;
  }
};

namespace {

constexpr int kRouteThreads = 1024;
constexpr int kOperandWarps = 8;

// Expert GEMM block: kWarps warps, each owning Shape::kTPW adjacent n16 output
// tiles, over one route tile of up to kTileRows rows; K streamed in kKC chunks,
// kStages deep.
constexpr int kWarps = 8;
constexpr int kThreads = 32 * kWarps;
constexpr int kKC = 64;
constexpr int kSteps = kKC / 16;
constexpr int kStages = 3;

struct Routing {
  const void* ids;
  const void* expert_map;
  const int64_t* pointers;
  const int64_t* metadata;
  int64_t global_experts;
  int64_t local_experts;
  int64_t topk;
  bool ids64;
  bool map64;
};

__device__ __forceinline__ int64_t integer_at(const void* ptr, bool wide, int64_t i) {
  return wide ? static_cast<const int64_t*>(ptr)[i] : static_cast<const int32_t*>(ptr)[i];
}
__device__ __forceinline__ int64_t local_expert(const Routing& routing, int64_t assignment) {
  const int64_t global = integer_at(routing.ids, routing.ids64, assignment);
  if (global < 0 || global >= routing.global_experts) return -1;
  const int64_t local = integer_at(routing.expert_map, routing.map64, global);
  return local >= 0 && local < routing.local_experts ? local : -1;
}
__device__ __forceinline__ const half* expert_part(const Routing& routing, int64_t expert,
                                                  int projection, int part) {
  return reinterpret_cast<const half*>(routing.pointers[expert * 9 + projection * 3 + part]);
}

// Exclusive block-wide scan of two counters; also returns the totals.
__device__ __forceinline__ void block_scan2(int a, int b, int& a_before, int& b_before,
                                            int& a_total, int& b_total) {
  __shared__ int warp_a[32];
  __shared__ int warp_b[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  int x = a;
  int y = b;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int xu = __shfl_up_sync(0xffffffffu, x, d);
    const int yu = __shfl_up_sync(0xffffffffu, y, d);
    if (lane >= d) {
      x += xu;
      y += yu;
    }
  }
  if (lane == 31) {
    warp_a[warp] = x;
    warp_b[warp] = y;
  }
  __syncthreads();
  if (warp == 0) {
    int wx = lane < warps ? warp_a[lane] : 0;
    int wy = lane < warps ? warp_b[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int xu = __shfl_up_sync(0xffffffffu, wx, d);
      const int yu = __shfl_up_sync(0xffffffffu, wy, d);
      if (lane >= d) {
        wx += xu;
        wy += yu;
      }
    }
    warp_a[lane] = wx;
    warp_b[lane] = wy;
  }
  __syncthreads();
  a_before = x - a + (warp > 0 ? warp_a[warp - 1] : 0);
  b_before = y - b + (warp > 0 ? warp_b[warp - 1] : 0);
  a_total = warp_a[warps - 1];
  b_total = warp_b[warps - 1];
}

// Groups the local assignments by local expert into kTileRows-row tiles and
// records each assignment's sorted position (one block). Order inside an
// expert is arbitrary: a row's result never depends on its position. Traps
// on an expert outside the served format (the caller's host checks cover the
// shapes; formats and trellis alignment live in device metadata).
template <int kIntermediate, int kBitsX2>
__global__ void __launch_bounds__(kRouteThreads)
route(Routing routing, int64_t assignments, int* table) {
  extern __shared__ int route_shared[];
  const int experts = static_cast<int>(routing.local_experts);
  int* cursor = route_shared;
  int* row_begin = cursor + experts;
  int* tile_begin = row_begin + experts;   // experts + 1 entries
  for (int e = threadIdx.x; e < experts; e += blockDim.x) {
    cursor[e] = 0;
    for (int q = 0; q < 3; ++q) {
      const int64_t* meta = routing.metadata + int64_t(e) * 12 + q * 4;
      const int64_t in = q < 2 ? kHidden : kIntermediate;
      const int64_t out = q < 2 ? kIntermediate : kHidden;
      const uintptr_t trellis = static_cast<uintptr_t>(routing.pointers[int64_t(e) * 9 + q * 3]);
      if (meta[0] != in || meta[1] != out || meta[2] != kBitsX2 || meta[3] != kCodebook ||
          (trellis & 15) != 0) {
        __trap();
      }
    }
  }
  __syncthreads();
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    if (e >= 0) atomicAdd(&cursor[e], 1);
  }
  __syncthreads();
  const int per = (experts + blockDim.x - 1) / blockDim.x;
  const int e0 = min(experts, static_cast<int>(threadIdx.x) * per);
  const int e1 = min(experts, e0 + per);
  int rows = 0;
  int tiles = 0;
  for (int e = e0; e < e1; ++e) {
    rows += cursor[e];
    tiles += (cursor[e] + kTileRows - 1) / kTileRows;
  }
  int row_at, tile_at, row_total, tile_total;
  block_scan2(rows, tiles, row_at, tile_at, row_total, tile_total);
  for (int e = e0; e < e1; ++e) {
    const int count = cursor[e];
    row_begin[e] = row_at;
    tile_begin[e] = tile_at;
    cursor[e] = row_at;
    row_at += count;
    tile_at += (count + kTileRows - 1) / kTileRows;
  }
  if (threadIdx.x == 0) {
    tile_begin[experts] = tile_total;
    table[0] = row_total;
    table[1] = tile_total;
  }
  __syncthreads();
  int4* tile_table = reinterpret_cast<int4*>(table + Layout::tiles_offset(assignments));
  for (int t = threadIdx.x; t < tile_total; t += blockDim.x) {
    // Last expert whose first tile is at or before t (empty experts share
    // their successor's start, so the search lands on the owner).
    int lo = 0;
    int hi = experts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (tile_begin[mid] <= t) lo = mid; else hi = mid;
    }
    const int first = row_begin[lo] + (t - tile_begin[lo]) * kTileRows;
    const int end = lo + 1 < experts ? row_begin[lo + 1] : row_total;
    tile_table[t] = make_int4(lo, first, min(kTileRows, end - first), 0);
  }
  int* sorted = table + Layout::sorted_offset();
  int* position = table + Layout::position_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    int at = -1;
    if (e >= 0) {
      at = atomicAdd(&cursor[e], 1);
      sorted[at] = static_cast<int>(a);
    }
    position[a] = at;
  }
}

// --------------------------------------------------------------------------
// Tensor-core operands.

// Exponent e of the power of two that puts max |v| 2^e in [2^14, 2^15), with
// 2^e and 2^-e normal FP32 (peak 0: e = 0).
__device__ __forceinline__ int split_exponent(float peak) {
  const int biased = static_cast<int>(__float_as_uint(peak) >> 23);
  return peak == 0.0f ? 0 : min(141 - biased, 126);
}
__device__ __forceinline__ float pow2(int e) { return __uint_as_float(uint32_t(127 + e) << 23); }

// H128 over one 128-value block held by the 8 lanes of an aligned lane group
// (l = lane % 8): v[8 h + j] is element 64 h + 8 l + j. Exactly the per-element
// operations and stage order of hadamard128_warp (distances 1..64 in
// increasing order, lower element v + other, upper other - v, then the
// 1/sqrt(128) scale), so the results are bitwise equal; distances 1, 2, 4 and
// 64 run in registers, 8, 16, 32 by shuffles inside the group (3 shuffle
// stages instead of 5). Every lane of the warp must call it.
__device__ __forceinline__ void h128_register_stage(float (&v)[16], int distance) {
  float next[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float other = v[i ^ distance];
    next[i] = (i & distance) ? __fsub_rn(other, v[i]) : __fadd_rn(v[i], other);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = next[i];
}
__device__ __forceinline__ void hadamard128_group(float (&v)[16], int lane) {
  h128_register_stage(v, 1);
  h128_register_stage(v, 2);
  h128_register_stage(v, 4);
#pragma unroll
  for (int d = 1; d < 8; d <<= 1) {   // element distance 8 d
    const bool upper = (lane & d) != 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float other = __shfl_xor_sync(0xffffffffu, v[i], d);
      v[i] = upper ? __fsub_rn(other, v[i]) : __fadd_rn(v[i], other);
    }
  }
  h128_register_stage(v, 8);           // element distance 64
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = __fmul_rn(v[i], kH128Scale);
}

// Eight consecutive values as FP32 (exact widening); one 16-byte load when
// the row base is 16-byte aligned (`vec`), element loads otherwise.
__device__ __forceinline__ void load8(const __nv_bfloat16* p, bool vec, float* f) {
  if (vec) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      f[2 * q] = __uint_as_float(w[q] << 16);
      f[2 * q + 1] = __uint_as_float(w[q] & 0xffff0000u);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = __bfloat162float(p[j]);
  }
}
__device__ __forceinline__ void load8(const half* p, bool vec, float* f) {
  if (vec) {
    const uint4 u = *reinterpret_cast<const uint4*>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&w[q]));
      f[2 * q] = pair.x;
      f[2 * q + 1] = pair.y;
    }
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) f[j] = __half2float(p[j]);
  }
}
// FP32 workspace rows: always 16-byte aligned (checked workspace base, row and
// column offsets multiples of 8 floats).
__device__ __forceinline__ void load8(const float* p, float* f) {
  const float4 a = *reinterpret_cast<const float4*>(p);
  const float4 b = *reinterpret_cast<const float4*>(p + 4);
  f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w;
  f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
__device__ __forceinline__ bool aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & 15) == 0;
}

template <int kVals>
__device__ __forceinline__ float lane_peak(const float (&v)[kVals]) {
  float peak = 0.0f;
#pragma unroll
  for (int i = 0; i < kVals; ++i) peak = fmaxf(peak, fabsf(v[i]));
  return peak;
}

__device__ __forceinline__ uint32_t pack_halves(__half a, __half b) {
  return uint32_t(__half_as_ushort(a)) | (uint32_t(__half_as_ushort(b)) << 16);
}
// Eight consecutive operand values y = v 2^e as the FP16 high halves at `hi`
// and the FP16-rounded residuals at `lo` (both 16-byte aligned).
__device__ __forceinline__ void store8_split(const float* v, float scale, __half* hi,
                                             __half* lo) {
  uint32_t h[4];
  uint32_t l[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float y0 = __fmul_rn(v[2 * q], scale);
    const float y1 = __fmul_rn(v[2 * q + 1], scale);
    const __half h0 = __float2half_rn(y0);
    const __half h1 = __float2half_rn(y1);
    h[q] = pack_halves(h0, h1);
    l[q] = pack_halves(__float2half_rn(__fsub_rn(y0, __half2float(h0))),
                       __float2half_rn(__fsub_rn(y1, __half2float(h1))));
  }
  *reinterpret_cast<uint4*>(hi) = make_uint4(h[0], h[1], h[2], h[3]);
  *reinterpret_cast<uint4*>(lo) = make_uint4(l[0], l[1], l[2], l[3]);
}

// One 512-column iteration of x * suh and H128. Recomputed for the store so
// the kernel does not keep v[10][16] live (that spills). The second pass is
// the same operations, so the stored halves match a single pass.
__device__ __forceinline__ void gate_up_block(const __nv_bfloat16* row, const half* suh, bool vec,
                                             int lane, int lane_col, int it, float (&v)[16]) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int col = 512 * it + 64 * h + lane_col;
    float xf[8];
    float sf[8];
    load8(row + col, vec, xf);
    load8(suh + col, vec, sf);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[8 * h + j] = __fmul_rn(xf[j], sf[j]);
  }
  hadamard128_group(v, lane);
}

// One warp per (sorted position, gate/up projection): x * suh and H128 (the
// shipped FP32 operations), then the split operand row (K halves high, then
// K halves low) and its 2^-e. Lane group g of iteration it holds 128-block
// 4 it + g. Hidden 5120 is 10 iterations. The peak pass and the store pass
// recompute each block; the result is bitwise the held-v form.
__global__ void __launch_bounds__(32 * kOperandWarps)
gate_up_operand(const __nv_bfloat16* x, Routing routing, const int* table, __half* operand,
                float* unscale) {
  constexpr int kIters = kHidden / 512;
  static_assert(kIters == 10, "hidden 5120");
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* sorted = table + Layout::sorted_offset();
  const int64_t tasks = int64_t(table[0]) * 2;
  for (int64_t task = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5); task < tasks;
       task += int64_t(gridDim.x) * kOperandWarps) {
    const int64_t pos = task >> 1;
    const int projection = static_cast<int>(task & 1);
    const int64_t assignment = sorted[pos];
    const half* suh_row = expert_part(routing, local_expert(routing, assignment), projection, 1);
    const __nv_bfloat16* row = x + (assignment / routing.topk) * kHidden;
    const bool vec = aligned16(row) && aligned16(suh_row);
    float peak = 0.0f;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      float v[16];
      gate_up_block(row, suh_row, vec, lane, lane_col, it, v);
      peak = fmaxf(peak, lane_peak(v));
    }
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
    const int e = split_exponent(peak);
    const float scale = pow2(e);
    __half* out = operand + task * 2 * kHidden;
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      float v[16];
      gate_up_block(row, suh_row, vec, lane, lane_col, it, v);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = 512 * it + 64 * h + lane_col;
        store8_split(&v[8 * h], scale, out + col, out + kHidden + col);
      }
    }
    if (lane == 0) unscale[pos * 3 + projection] = pow2(-e);
  }
}

// Down transform of one value: gate and up output transforms already applied
// (H128 done), then svh, SiLU(gate) * up and down suh (the shipped FP32
// operations).
__device__ __forceinline__ float down_input(float g, float u, float gate_svh, float up_svh,
                                            float down_suh, float limit) {
  float gv = __fmul_rn(g, gate_svh);
  float uv = __fmul_rn(u, up_svh);
  if (limit > 0.0f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  const float silu = gv / __fadd_rn(1.0f, expf(-gv));
  return __fmul_rn(__fmul_rn(silu, uv), down_suh);
}

// One warp per sorted position: gate and up output transforms (H128, svh),
// SiLU(gate) * up, down suh, down H128, then the split down operand row and
// its 2^-e. I = 640 is four lane-group blocks plus a 128-column warp tail.
// I = 512 is four lane-group blocks and no tail. The peak covers both.
template <int kIntermediate>
__global__ void __launch_bounds__(32 * kOperandWarps)
down_operand(Routing routing, const int* table, const float* gate_up, __half* operand,
             float* unscale, float limit) {
  static_assert(kIntermediate == 640 || kIntermediate == 512, "down intermediate");
  constexpr int kTail = 512;
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* sorted = table + Layout::sorted_offset();
  const int64_t tasks = table[0];
  for (int64_t pos = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5); pos < tasks;
       pos += int64_t(gridDim.x) * kOperandWarps) {
    const int64_t expert = local_expert(routing, sorted[pos]);
    const half* gate_svh = expert_part(routing, expert, 0, 2);
    const half* up_svh = expert_part(routing, expert, 1, 2);
    const half* down_suh = expert_part(routing, expert, 2, 1);
    const bool vec = aligned16(gate_svh) && aligned16(up_svh) && aligned16(down_suh);
    const float* gate = gate_up + pos * 2 * kIntermediate;
    const float* up = gate + kIntermediate;

    float v[16];
    {
      float g[16];
      float u[16];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        load8(gate + lane_col + 64 * h, &g[8 * h]);
        load8(up + lane_col + 64 * h, &u[8 * h]);
      }
      hadamard128_group(g, lane);
      hadamard128_group(u, lane);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int col = lane_col + 64 * h;
        float gs[8];
        float us[8];
        float ds[8];
        load8(gate_svh + col, vec, gs);
        load8(up_svh + col, vec, us);
        load8(down_suh + col, vec, ds);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          v[8 * h + j] = down_input(g[8 * h + j], u[8 * h + j], gs[j], us[j], ds[j], limit);
        }
      }
      hadamard128_group(v, lane);
    }
    float w[4] = {};
    if constexpr (kIntermediate == 640) {
      float g[4];
      float u[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        g[i] = gate[kTail + lane + 32 * i];
        u[i] = up[kTail + lane + 32 * i];
      }
      hadamard128_warp(g, lane);
      hadamard128_warp(u, lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int col = kTail + lane + 32 * i;
        w[i] = down_input(g[i], u[i], __half2float(gate_svh[col]), __half2float(up_svh[col]),
                          __half2float(down_suh[col]), limit);
      }
      hadamard128_warp(w, lane);
    }

    float peak = lane_peak(v);
    if constexpr (kIntermediate == 640) peak = fmaxf(peak, lane_peak(w));
#pragma unroll
    for (int d = 16; d >= 1; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
    const int e = split_exponent(peak);
    const float scale = pow2(e);
    __half* out = operand + pos * 2 * kIntermediate;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int col = lane_col + 64 * h;
      store8_split(&v[8 * h], scale, out + col, out + kIntermediate + col);
    }
    if constexpr (kIntermediate == 640) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int k = kTail + lane + 32 * i;
        const float y = __fmul_rn(w[i], scale);
        const __half h = __float2half_rn(y);
        out[k] = h;
        out[kIntermediate + k] = __float2half_rn(__fsub_rn(y, __half2float(h)));
      }
    }
    if (lane == 0) unscale[pos * 3 + 2] = pow2(-e);
  }
}

// --------------------------------------------------------------------------
// Expert GEMM.

__device__ __forceinline__ uint32_t smem_address(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 16-byte cp.async to a shared-memory address; src_size 0 zero-fills.
__device__ __forceinline__ void copy16_to(uint32_t dst, const void* src, int src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src),
               "r"(src_size));
}
__device__ __forceinline__ void copy_commit() { asm volatile("cp.async.commit_group;"); }
template <int kPending>
__device__ __forceinline__ void copy_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending));
}
// d = A B + c (FP16 in, FP32 accumulate).
__device__ __forceinline__ void mma(const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                    const float (&c)[4], float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]),
        "f"(c[2]), "f"(c[3]));
}

// Two mul1 codebook values as an FP16 pair (low half = s0): exactly
// codebook_pair<2> before its widening to FP32.
__device__ __forceinline__ uint32_t mul1_half2(uint32_t s0, uint32_t s1) {
  const uint32_t sum0 = __dp4a(s0 * 0x83DCD12Du, 0x01010101u, 0x6400u);
  const uint32_t sum1 = __dp4a(s1 * 0x83DCD12Du, 0x01010101u, 0x6400u);
  const uint32_t packed = __byte_perm(sum0, sum1, 0x5410);
  const uint32_t scale = 0x1eee1eeeu;
  const uint32_t bias = 0xc931c931u;
  const __half2 value = __hfma2(*reinterpret_cast<const __half2*>(&packed),
                                *reinterpret_cast<const __half2*>(&scale),
                                *reinterpret_cast<const __half2*>(&bias));
  return *reinterpret_cast<const uint32_t*>(&value);
}

// A lane's eight decoded values as the m16n8k16 A fragment of W^T (fragment
// row = output column, fragment column = k): value pairs (0,1), (4,5), (2,3),
// (6,7) of decode_lane's order.
template <int kBitsX2>
__device__ __forceinline__ void decode_fragment(const uint32_t (&words)[TileRing<kBitsX2>::kLoadWords],
                                                int shift, uint32_t (&a)[4]) {
  using Ring = TileRing<kBitsX2>;
  uint32_t u[Ring::kWindowWords];
#pragma unroll
  for (int i = 0; i < Ring::kWindowWords; ++i) {
    const uint32_t next = i + 1 < Ring::kLoadWords ? words[i + 1] : 0u;
    u[i] = __funnelshift_l(next, words[i], shift);
  }
  uint32_t s[8];
  s[0] = window_state<Ring::rel(0), Ring::kWindowWords>(u);
  s[1] = window_state<Ring::rel(1), Ring::kWindowWords>(u);
  s[2] = window_state<Ring::rel(2), Ring::kWindowWords>(u);
  s[3] = window_state<Ring::rel(3), Ring::kWindowWords>(u);
  s[4] = window_state<Ring::rel(4), Ring::kWindowWords>(u);
  s[5] = window_state<Ring::rel(5), Ring::kWindowWords>(u);
  s[6] = window_state<Ring::rel(6), Ring::kWindowWords>(u);
  s[7] = window_state<Ring::rel(7), Ring::kWindowWords>(u);
  a[0] = mul1_half2(s[0], s[1]);
  a[1] = mul1_half2(s[4], s[5]);
  a[2] = mul1_half2(s[2], s[3]);
  a[3] = mul1_half2(s[6], s[7]);
}

struct Gemm {
  const int* table;
  int64_t assignments;     // routing table capacity
  const int64_t* pointers;
  const __half* operand;   // [pos][projection][hi, lo][k]
  const float* unscale;    // [pos][3]
  float* out;              // [pos][projection][n]
};

// Compile-time GEMM shape: K, N, projections per launch (2: gate and up; 1:
// down), the first projection's index in the expert's 3 matrices and the n16
// output tiles per warp (each B fragment feeds TPW decoded A fragments).
//
// One stage of the K pipeline: the hi/lo operand rows of the route tile (row
// = 64 halves in eight 16-byte chunks, chunk c stored at c ^ (row & 7)) at
// offset 0, then the block's kBlockTiles adjacent trellis tiles of each k16
// step at kStageW.
template <int K, int N, int Projections, int First, int TPW, int Bits>
struct Shape {
  static constexpr int kK = K;
  static constexpr int kN = N;
  static constexpr int kProjections = Projections;
  static constexpr int kFirst = First;
  static constexpr int kTPW = TPW;
  static constexpr int kBitsX2 = Bits;
  static constexpr int kTileBytes = 16 * Bits;
  static constexpr int kBlockTiles = kWarps * TPW;
  static constexpr int kChunks = K / kKC;
  static constexpr int kColBlocks = N / (16 * kBlockTiles);
  static constexpr int kTilesN = N / 16;
  static constexpr int64_t kRowHalves = int64_t(Projections) * 2 * K;
  static constexpr int64_t kWChunk = int64_t(kSteps) * kTilesN * kTileBytes;
  static constexpr uint32_t kStageW = 2 * kTileRows * kKC * 2;
  static constexpr uint32_t kStageBytes = kStageW + kSteps * kBlockTiles * kTileBytes;
  static constexpr int kSmemBytes = kStages * kStageBytes;
  static_assert(K % kKC == 0 && N % (16 * kBlockTiles) == 0 && kChunks >= kStages - 1,
                "served shape");
  static_assert(kStageBytes % 16 == 0, "16-byte aligned stages");
};

template <class S>
__device__ __forceinline__ uint32_t next_stage(uint32_t offset) {
  return offset + S::kStageBytes == kStages * S::kStageBytes ? 0u : offset + S::kStageBytes;
}
__device__ __forceinline__ void ldmatrix4_at(uint32_t address, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(address));
}
__device__ __forceinline__ uint32_t lds32(uint32_t address) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(address));
  return v;
}

// The K loop and epilogue of one block item for a route tile of kG 8-row
// groups starting at group kGroup0 (rows in 8 (kGroup0 + kG) - 7 .. 8 kGroup0):
// straight-line code, so the group MMAs of a k16 step overlap (a runtime group
// bound made every group a separately serialised branch). The ldmatrix of
// group g + 1 is issued before the MMAs of group g; each B fragment serves the
// warp's S::kTPW n16 tiles. Shared addresses are per-lane constants plus the
// stage offset; `load` issues the next chunk's copies into its own rotating
// stage. Groups are independent rows, so two sequential passes are bitwise
// one pass as long as each group still sums in increasing k.
template <int kG, int kGroup0, class S, class Load>
__device__ __forceinline__ void run_tile(const Gemm& p, uint32_t smem, Load& load, int rows,
                                         int first, int pi, int cb) {
  constexpr int TPW = S::kTPW;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  using Ring = TileRing<S::kBitsX2>;
  float sum[TPW][kG][4];
#pragma unroll
  for (int j = 0; j < TPW; ++j) {
#pragma unroll
    for (int g = 0; g < kG; ++g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) sum[j][g][i] = 0.0f;
    }
  }
  const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // ldmatrix source of this lane: matrix lane / 8 = (hi k0..7, hi k8..15,
  // lo k0..7, lo k8..15) of rows 8 g + lane % 8; for k16 step s the 16-byte
  // vector (2 s + lane / 8 % 2) ^ (lane % 8) = 2 s ^ x.
  const int x = ((lane >> 3) & 1) ^ (lane & 7);
  const uint32_t b_lane = smem + (lane >> 4) * (kTileRows * kKC * 2) + (lane & 7) * (kKC * 2);
  // Trellis words of this lane's window in the warp's first tile of step 0.
  uint32_t w_lane[Ring::kLoadWords];
  int shift;
  {
    const LaneWindow<S::kBitsX2> window(lane);
    shift = window.shift;
#pragma unroll
    for (int i = 0; i < Ring::kLoadWords; ++i) {
      w_lane[i] = smem + S::kStageW + TPW * warp * S::kTileBytes + 4 * window.word[i];
    }
  }

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    load();
    copy_commit();
  }
  uint32_t read = 0;
  for (int chunk = 0; chunk < S::kChunks; ++chunk) {
    copy_wait<kStages - 2>();
    __syncthreads();
    if (chunk + kStages - 1 < S::kChunks) load();
    copy_commit();
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
      uint32_t a[TPW][4];
#pragma unroll
      for (int j = 0; j < TPW; ++j) {
        uint32_t words[Ring::kLoadWords];
#pragma unroll
        for (int i = 0; i < Ring::kLoadWords; ++i) {
          words[i] = lds32(w_lane[i] + read + step * (S::kBlockTiles * S::kTileBytes) +
                           j * S::kTileBytes);
        }
        decode_fragment<S::kBitsX2>(words, shift, a[j]);
      }
      const uint32_t bsrc = b_lane + read + ((x ^ (2 * step)) * 16);
      constexpr uint32_t kGroupBytes = 8 * kKC * 2;
      uint32_t b[2][4];
      ldmatrix4_at(bsrc + kGroup0 * kGroupBytes, b[0]);
#pragma unroll
      for (int g = 0; g < kG; ++g) {
        if (g + 1 < kG) {
          ldmatrix4_at(bsrc + (kGroup0 + g + 1) * kGroupBytes, b[(g + 1) & 1]);
        }
        const uint32_t(&bg)[4] = b[g & 1];
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
          // The k16 low products (C = 0, lo magnitude) seed the high MMA, so
          // one truncating accumulate at hi magnitude per k16; the k16
          // partial is then summed in FP32 (RN) in increasing k.
          float d[4];
          mma(a[j], bg[2], bg[3], zero, d);
          mma(a[j], bg[0], bg[1], d, d);
#pragma unroll
          for (int i = 0; i < 4; ++i) sum[j][g][i] = __fadd_rn(sum[j][g][i], d[i]);
        }
      }
    }
    read = next_stage<S>(read);
  }

  // Lane holds output columns 16 tile + lane / 4 (+ 8) of rows 8 g + 2 (lane % 4) (+ 1).
  const int t = lane & 3;
#pragma unroll
  for (int j = 0; j < TPW; ++j) {
    const int col = (cb * S::kBlockTiles + TPW * warp + j) * 16 + (lane >> 2);
#pragma unroll
    for (int g = 0; g < kG; ++g) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = 8 * (kGroup0 + g) + 2 * t + c;
        if (r < rows) {
          const int64_t pos = first + r;
          const float scale = p.unscale[pos * 3 + S::kFirst + pi];
          float* out = p.out + (pos * S::kProjections + pi) * S::kN + col;
          out[0] = __fmul_rn(sum[j][g][c], scale);
          out[8] = __fmul_rn(sum[j][g][2 + c], scale);
        }
      }
    }
  }
}

// Block items: (route tile, projection, column block of 16 kBlockTiles), column
// blocks fastest so concurrently running blocks share the tile's operand rows in L2.
template <class S>
__global__ void __launch_bounds__(kThreads, 1) gemm(Gemm p) {
  extern __shared__ __align__(16) unsigned char gemm_shared[];
  const uint32_t smem = smem_address(gemm_shared);
  const int64_t items = int64_t(p.table[1]) * S::kProjections * S::kColBlocks;
  const int64_t item = blockIdx.x;
  if (item >= items) return;
  const int64_t tile_index = item / (S::kProjections * S::kColBlocks);
  const int rest = static_cast<int>(item % (S::kProjections * S::kColBlocks));
  const int pi = rest / S::kColBlocks;
  const int cb = rest % S::kColBlocks;
  const int4 tile = reinterpret_cast<const int4*>(p.table + Layout::tiles_offset(p.assignments))[tile_index];
  const int rows = tile.z;
  const unsigned char* trellis = reinterpret_cast<const unsigned char*>(
      p.pointers[int64_t(tile.x) * 9 + (S::kFirst + pi) * 3]);
  const __half* operand =
      p.operand + int64_t(tile.y) * S::kRowHalves + int64_t(pi) * 2 * S::kK;

  // Per-thread cp.async work of a stage: operand vectors v = tid + kThreads i
  // (16 bytes: chunk v % 8 of row v / 8 % 64, part v / 512), i.e. rows r0 and
  // r0 + 32 of both parts at shared offsets 0 / 4096 (row + 32) / 8192 (part
  // 1), and trellis vectors tid + kThreads j of the block's tiles. Sources
  // advance by one chunk per load. Rows past the tile's 8-row groups are
  // never read; rows past the tile inside them are zero-filled from row 0.
  const int tid = threadIdx.x;
  static_assert(kThreads == 256 && kTileRows == 64 && kKC == 64, "operand copy layout");
  constexpr int kStepVectors = S::kBlockTiles * S::kTileBytes / 16;
  constexpr int kWVectors = kSteps * kStepVectors;
  constexpr int kWCopies = (kWVectors + kThreads - 1) / kThreads;
  const int r0 = tid >> 3;
  const int ck = tid & 7;
  const int padded_rows = (rows + 7) & ~7;
  const bool live0 = r0 < padded_rows;
  const bool live1 = r0 + 32 < padded_rows;
  const int size0 = r0 < rows ? 16 : 0;
  const int size1 = r0 + 32 < rows ? 16 : 0;
  const __half* src0 = operand + int64_t(r0 < rows ? r0 : 0) * S::kRowHalves + ck * 8;
  const __half* src1 = operand + int64_t(r0 + 32 < rows ? r0 + 32 : 0) * S::kRowHalves + ck * 8;
  const uint32_t b_dst = smem + r0 * (kKC * 2) + (ck ^ (r0 & 7)) * 16;
  // The last trellis copy covers only threads tid < kWVectors - (kWCopies - 1) kThreads.
  const bool w_last_live = tid + (kWCopies - 1) * kThreads < kWVectors;
  const unsigned char* w_src[kWCopies];
#pragma unroll
  for (int j = 0; j < kWCopies; ++j) {
    const int w = j + 1 < kWCopies || w_last_live ? tid + j * kThreads : tid;
    w_src[j] = trellis +
               (int64_t(w / kStepVectors) * S::kTilesN + int64_t(cb) * S::kBlockTiles) * S::kTileBytes +
               (w % kStepVectors) * 16;
  }
  const uint32_t w_dst = smem + S::kStageW + tid * 16;
  uint32_t stage = 0;

  auto load = [&]() {
    if (live0) {
      copy16_to(b_dst + stage, src0, size0);
      copy16_to(b_dst + stage + 8192, src0 + S::kK, size0);
    }
    if (live1) {
      copy16_to(b_dst + stage + 4096, src1, size1);
      copy16_to(b_dst + stage + 12288, src1 + S::kK, size1);
    }
#pragma unroll
    for (int j = 0; j < kWCopies; ++j) {
      if (j + 1 < kWCopies || w_last_live) {
        copy16_to(w_dst + stage + j * kThreads * 16, w_src[j], 16);
      }
      w_src[j] += S::kWChunk;
    }
    src0 += kKC;
    src1 += kKC;
    stage = next_stage<S>(stage);
  };

  // Down K2 (TPW 2, bits_x2 4) keeps sum[2][8][4] and spills 8 bytes at the
  // 255-register cap. Groups are independent rows, so a tile wider than 4
  // groups runs as two passes of at most 4. Each group still sums in
  // increasing k, which is bitwise the single pass. The other shapes fit in
  // one pass; their source snapshots stay out of this function so they do not
  // pick up the extra live pointers.
  const int groups = (rows + 7) >> 3;
  if constexpr (S::kTPW == 2 && S::kBitsX2 == 4) {
    const __half* const src0_base = src0;
    const __half* const src1_base = src1;
    const unsigned char* w_src_base[kWCopies];
#pragma unroll
    for (int j = 0; j < kWCopies; ++j) w_src_base[j] = w_src[j];
    auto rewind = [&]() {
      copy_wait<0>();
      __syncthreads();
      src0 = src0_base;
      src1 = src1_base;
#pragma unroll
      for (int j = 0; j < kWCopies; ++j) w_src[j] = w_src_base[j];
      stage = 0;
    };
    switch (groups) {
      case 1: run_tile<1, 0, S>(p, smem, load, rows, tile.y, pi, cb); break;
      case 2: run_tile<2, 0, S>(p, smem, load, rows, tile.y, pi, cb); break;
      case 3: run_tile<3, 0, S>(p, smem, load, rows, tile.y, pi, cb); break;
      case 4: run_tile<4, 0, S>(p, smem, load, rows, tile.y, pi, cb); break;
      case 5:
        run_tile<4, 0, S>(p, smem, load, rows, tile.y, pi, cb);
        rewind();
        run_tile<1, 4, S>(p, smem, load, rows, tile.y, pi, cb);
        break;
      case 6:
        run_tile<4, 0, S>(p, smem, load, rows, tile.y, pi, cb);
        rewind();
        run_tile<2, 4, S>(p, smem, load, rows, tile.y, pi, cb);
        break;
      case 7:
        run_tile<4, 0, S>(p, smem, load, rows, tile.y, pi, cb);
        rewind();
        run_tile<3, 4, S>(p, smem, load, rows, tile.y, pi, cb);
        break;
      case 8:
        run_tile<4, 0, S>(p, smem, load, rows, tile.y, pi, cb);
        rewind();
        run_tile<4, 4, S>(p, smem, load, rows, tile.y, pi, cb);
        break;
      default: break;
    }
  } else {
    switch (groups) {
#define EXL3_GROUPS(G) \
      case G: run_tile<G, 0, S>(p, smem, load, rows, tile.y, pi, cb); break;
      EXL3_GROUPS(1) EXL3_GROUPS(2) EXL3_GROUPS(3) EXL3_GROUPS(4)
      EXL3_GROUPS(5) EXL3_GROUPS(6) EXL3_GROUPS(7) EXL3_GROUPS(8)
#undef EXL3_GROUPS
      default: break;
    }
  }
}

// Eight consecutive sums: one 16-byte store per 8 BF16 when the output base
// is 16-byte aligned, element stores otherwise.
__device__ __forceinline__ void store8_sum(__nv_bfloat16* out, bool vec, const float* sum) {
  if (vec) {
    uint32_t w[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      w[q] = uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[2 * q]))) |
             (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(sum[2 * q + 1]))) << 16);
    }
    *reinterpret_cast<uint4*>(out) = make_uint4(w[0], w[1], w[2], w[3]);
  } else {
#pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = __float2bfloat16_rn(sum[j]);
  }
}
// One warp per (token, 512-column span; lane group g holds 128-block g of the
// span): each local route's down output transform (H128, svh) in the shipped
// FP32 operations, then its routing-weight product summed in FP64 in route
// order and rounded to FP32 once, then to BF16 (round to nearest even). The
// route loop is warp-uniform.
template <typename Out>
__global__ void combine(const float* down, const float* weights, Routing routing,
                        const int* table, int64_t assignments, int64_t rows, Out* out) {
  constexpr int kSpans = kHidden / 512;
  static_assert(kSpans == 10, "hidden 5120 is ten 512-column spans");
  const int lane = threadIdx.x & 31;
  const int lane_col = 128 * (lane >> 3) + 8 * (lane & 7);
  const int* position = table + Layout::position_offset(assignments);
  const bool out_vec = aligned16(out);
  // One task per warp (no grid-stride loop), so the shuffles are in provably
  // converged code (a task loop makes ptxas call a WARPSYNC stub per shuffle).
  {
    const int64_t task = int64_t(blockIdx.x) * kOperandWarps + (threadIdx.x >> 5);
    if (task >= rows * kSpans) return;
    const int64_t row = task / kSpans;
    const int base = static_cast<int>(task % kSpans) * 512 + lane_col;
    double sum[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) sum[i] = 0.0;
    for (int64_t route = 0; route < routing.topk; ++route) {
      const int64_t assignment = row * routing.topk + route;
      const int pos = position[assignment];
      if (pos < 0) continue;
      const half* svh = expert_part(routing, local_expert(routing, assignment), 2, 2);
      const bool vec = aligned16(svh);
      const double weight = weights[assignment];
      const float* src = down + int64_t(pos) * kHidden + base;
      float v[16];
      load8(src, v);
      load8(src + 64, v + 8);
      hadamard128_group(v, lane);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float s[8];
        load8(svh + base + 64 * h, vec, s);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const double value = __fmul_rn(v[8 * h + j], s[j]);
          sum[8 * h + j] = fma(value, weight, sum[8 * h + j]);
        }
      }
    }
    float rounded[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) rounded[i] = __double2float_rn(sum[i]);
    Out* dst = out + row * kHidden + base;
    store8_sum(dst, out_vec, rounded);
    store8_sum(dst + 64, out_vec, rounded + 8);
  }
}

unsigned int warp_grid(int64_t tasks) {
  return static_cast<unsigned int>(
      std::min<int64_t>((tasks + kOperandWarps - 1) / kOperandWarps, 65535));
}

template <class S>
void launch_gemm(const Gemm& p, int64_t max_items, cudaStream_t stream) {
  static const bool configured = [] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gemm<S>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        S::kSmemBytes));
    return true;
  }();
  (void)configured;
  TORCH_CHECK(max_items <= std::numeric_limits<int>::max(), "EXL3 MoE prefill grid too large");
  gemm<S><<<static_cast<unsigned int>(max_items), kThreads, S::kSmemBytes, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

bool served_shape(int64_t hidden, int64_t intermediate) {
  return hidden == kHidden && (intermediate == 640 || intermediate == 512);
}

template <int kIntermediate>
int64_t workspace_count(int64_t assignments) {
  const int64_t operand = assignments * 2 * 2 * kHidden / 2;   // FP16 gate/up rows (later down)
  const int64_t gate_up = assignments * 2 * kIntermediate;
  const int64_t unscale = (assignments * 3 + 3) / 4 * 4;
  return operand + gate_up + unscale + (Layout::ints(assignments) + 3) / 4 * 4;
}

int64_t prefill_row_cap() {
  const char* env = std::getenv("VLLM_EXL3_TC_PREFILL_ROWS");
  if (env == nullptr || env[0] == '\0') return 2048;
  char* end = nullptr;
  const long value = std::strtol(env, &end, 10);
  if (end == env || *end != '\0' || value < 1) return 2048;
  return value;
}

// One row sub-chunk. Rows do not depend on each other, so several chunks on
// the same workspace match one call.
template <int kIntermediate, int kBitsX2>
void run_chunk(const __nv_bfloat16* x, const void* ids, bool ids64, const float* weights,
               const void* expert_map, bool map64, int64_t global_experts,
               const int64_t* pointers, const int64_t* metadata, int64_t local_experts,
               int64_t rows, int64_t topk, float limit, __nv_bfloat16* out, float* workspace,
               cudaStream_t stream) {
  using GateUpShape = Shape<kHidden, kIntermediate, 2, 0, 1, kBitsX2>;
  using DownShape = Shape<kIntermediate, kHidden, 1, 2, 2, kBitsX2>;
  const int64_t assignments = rows * topk;
  Routing routing{ids, expert_map, pointers, metadata, global_experts, local_experts, topk,
                  ids64, map64};
  __half* operand = reinterpret_cast<__half*>(workspace);
  float* down_result = workspace + assignments * kIntermediate;
  float* gate_up = workspace + assignments * 2 * kHidden;
  float* unscale = gate_up + assignments * 2 * kIntermediate;
  int* table = reinterpret_cast<int*>(unscale + (assignments * 3 + 3) / 4 * 4);
  route<kIntermediate, kBitsX2>
      <<<1, kRouteThreads, (3 * local_experts + 1) * sizeof(int), stream>>>(routing, assignments,
                                                                           table);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int64_t max_tiles = std::min<int64_t>(
      assignments, assignments / kTileRows + std::min<int64_t>(local_experts, assignments) + 1);
  gate_up_operand<<<warp_grid(2 * assignments), 32 * kOperandWarps, 0, stream>>>(
      x, routing, table, operand, unscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  Gemm p{};
  p.table = table;
  p.assignments = assignments;
  p.pointers = pointers;
  p.operand = operand;
  p.unscale = unscale;
  p.out = gate_up;
  launch_gemm<GateUpShape>(p, max_tiles * GateUpShape::kProjections * GateUpShape::kColBlocks,
                           stream);
  down_operand<kIntermediate><<<warp_grid(assignments), 32 * kOperandWarps, 0, stream>>>(
      routing, table, gate_up, operand, unscale, limit);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  p.out = down_result;
  launch_gemm<DownShape>(p, max_tiles * DownShape::kProjections * DownShape::kColBlocks, stream);
  const int64_t combine_blocks = (rows * (kHidden / 512) + kOperandWarps - 1) / kOperandWarps;
  TORCH_CHECK(combine_blocks <= std::numeric_limits<int>::max(), "EXL3 MoE prefill grid too large");
  const unsigned int grid = static_cast<unsigned int>(combine_blocks);
  combine<<<grid, 32 * kOperandWarps, 0, stream>>>(down_result, weights, routing, table,
                                                   assignments, rows, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace moe_prefill

int64_t moe_workspace_size_prefill(int64_t max_rows, int64_t hidden_size,
                                   int64_t intermediate_size, int64_t topk,
                                   int64_t local_experts) {
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  TORCH_CHECK(moe_prefill::served_shape(hidden_size, intermediate_size),
              "EXL3 MoE prefill covers hidden 5120 and intermediate 640 or 512 only");
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  TORCH_CHECK(assignments <= std::numeric_limits<int>::max(),
              "EXL3 MoE has too many routed assignments");
  const int64_t count = intermediate_size == 640
                            ? moe_prefill::workspace_count<640>(assignments)
                            : moe_prefill::workspace_count<512>(assignments);
  return workspace_elements(count);
}

void moe_out_prefill(const torch::Tensor& x, const torch::Tensor& topk_ids,
                     const torch::Tensor& topk_weights,
                     const torch::Tensor& packed_ptrs,
                     const torch::Tensor& packed_meta,
                     const torch::Tensor& expert_map, int64_t intermediate_size,
                     int64_t bits_x2, double limit,
                     torch::Tensor& out, torch::Tensor& workspace) {
  TORCH_CHECK(x.is_cuda(), "EXL3 x must be CUDA");
  check_tensor(x, x, at::kBFloat16, "x");
  check_integer(topk_ids, x, "topk_ids");
  check_integer(expert_map, x, "expert_map");
  check_tensor(topk_weights, x, at::kFloat, "topk_weights");
  check_tensor(packed_ptrs, x, at::kLong, "packed_ptrs");
  check_tensor(packed_meta, x, at::kLong, "packed_meta");
  check_tensor(out, x, at::kBFloat16, "out");
  check_tensor(workspace, x, at::kFloat, "workspace");
  TORCH_CHECK(x.dim() == 2 && out.dim() == 2 && topk_ids.dim() == 2 &&
                  topk_weights.dim() == 2 && expert_map.dim() == 1 &&
                  packed_ptrs.dim() == 3 && packed_meta.dim() == 3,
              "EXL3 MoE tensor ranks mismatch");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = packed_ptrs.size(0);
  TORCH_CHECK(topk_ids.size(0) == rows && topk_weights.sizes() == topk_ids.sizes() &&
                  out.sizes() == x.sizes(), "EXL3 MoE row/topk/output geometry mismatch");
  TORCH_CHECK(packed_ptrs.size(1) == 3 && packed_ptrs.size(2) == 3 &&
                  packed_meta.size(0) == experts && packed_meta.size(1) == 3 &&
                  packed_meta.size(2) == 4, "EXL3 MoE pointer/metadata geometry mismatch");
  TORCH_CHECK(experts > 0 && experts <= kMoePrefillMaxExperts,
              "EXL3 MoE prefill needs 1..", kMoePrefillMaxExperts, " local experts");
  TORCH_CHECK(rows == 0 || rows >= kMoePrefillMinRows,
              "EXL3 MoE prefill serves calls of at least ", kMoePrefillMinRows, " rows");
  TORCH_CHECK((intermediate_size == 640 || intermediate_size == 512) &&
                  (bits_x2 == 6 || bits_x2 == 4) && hidden == 5120,
              "EXL3 TC MoE prefill supports hidden 5120 and (intermediate, bits_x2) in "
              "{(640,6),(512,6),(640,4),(512,4)} only");
  const int64_t cap = moe_prefill::prefill_row_cap();
  const int64_t chunk_rows = rows == 0 ? 0 : std::min(rows, cap);
  const int64_t required =
      moe_workspace_size_prefill(chunk_rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
  check_mutations(out, workspace,
                  {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  const bool ids64 = topk_ids.scalar_type() == at::kLong;
  const bool map64 = expert_map.scalar_type() == at::kLong;
  const auto* x_base = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
  const char* ids_base = static_cast<const char*>(topk_ids.data_ptr());
  const size_t id_bytes = static_cast<size_t>(topk_ids.element_size());
  const float* weight_base = topk_weights.data_ptr<float>();
  auto* out_base = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  const auto launch = [&](auto intermediate, auto bits) {
    constexpr int kI = decltype(intermediate)::value;
    constexpr int kBits = decltype(bits)::value;
    for (int64_t done = 0; done < rows;) {
      const int64_t n = std::min(cap, rows - done);
      moe_prefill::run_chunk<kI, kBits>(
          x_base + done * hidden, ids_base + done * topk * id_bytes, ids64,
          weight_base + done * topk, expert_map.data_ptr(), map64, expert_map.numel(),
          packed_ptrs.data_ptr<int64_t>(), packed_meta.data_ptr<int64_t>(), experts, n, topk,
          static_cast<float>(limit), out_base + done * hidden, workspace.data_ptr<float>(),
          stream);
      done += n;
    }
  };
  if (intermediate_size == 640 && bits_x2 == 6) {
    launch(std::integral_constant<int, 640>{}, std::integral_constant<int, 6>{});
  } else if (intermediate_size == 512 && bits_x2 == 6) {
    launch(std::integral_constant<int, 512>{}, std::integral_constant<int, 6>{});
  } else if (intermediate_size == 640 && bits_x2 == 4) {
    launch(std::integral_constant<int, 640>{}, std::integral_constant<int, 4>{});
  } else {
    launch(std::integral_constant<int, 512>{}, std::integral_constant<int, 4>{});
  }
}

}  // namespace exl3
