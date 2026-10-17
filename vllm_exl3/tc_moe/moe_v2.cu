// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Codec adapted from turboderp-org/exllamav3 (MIT); the MIT grant is in codec.cuh.
//
// EXL3 routed-expert MoE on the tensor cores for decode-sized calls.
// DS V4.1 Flash port of Morrowmake vllm-cmp170hx@18305948ac: hidden 5120,
// intermediate 640 or 512, bits_x2 6 (K3) or 4 (K2), mul1 codebook (id 2).
// The host switches on (intermediate, bits_x2); any other pair is a
// TORCH_CHECK failure. Arithmetic is the pinned kernel's: m16n8k16 FP16 MMAs
// on exactly decoded weights, a hi/lo FP16 split of the scaled activations,
// and FP64 input, activation and combine stages. Results are deterministic
// and a row never depends on other rows.

#include "moe_v2.h"
#include "common.h"
#include "codec_moe_v2.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>
#include <type_traits>

namespace exl3 {
namespace moe_v2 {
namespace {

// The route + gate/up operand launch (block 0 routes with these threads; the
// other blocks run one warp per H128 block of a row).
constexpr int kRouteInputThreads = 640;
// Route kernel shared memory: three int arrays per local expert.
constexpr int64_t kMaxRouteExperts = 4096;

// Integers the route kernel writes into the workspace each call:
//   [0] local assignment count, [1] row-tile count, [2..3] unused,
//   sorted[assignments]: local assignment ids grouped by local expert,
//   tiles[assignments] (16-byte aligned): {expert, first sorted index, rows, 0},
//   experts[assignments]: the local expert of each sorted entry,
//   by_assignment[assignments]: the local expert of each assignment (-1: none),
//   tile_trellis[assignments][3] (int64), tile_columns[assignments][3]: per row
//   tile and projection the trellis address and padded_out / 16.
struct RouteLayout {
  static constexpr int64_t kHeader = 4;
  __host__ __device__ static constexpr int64_t sorted_offset() { return kHeader; }
  __host__ __device__ static constexpr int64_t tiles_offset(int64_t assignments) {
    return kHeader + (assignments + 3) / 4 * 4;
  }
  __host__ __device__ static constexpr int64_t experts_offset(int64_t assignments) {
    return tiles_offset(assignments) + 4 * assignments;
  }
  __host__ __device__ static constexpr int64_t by_assignment_offset(int64_t assignments) {
    return experts_offset(assignments) + assignments;
  }
  // Per row tile and projection: the expert's trellis address (int64, 8-byte
  // aligned) and its tile columns (padded_out / 16).
  __host__ __device__ static constexpr int64_t tile_trellis_offset(int64_t assignments) {
    return (by_assignment_offset(assignments) + assignments + 1) / 2 * 2;
  }
  __host__ __device__ static constexpr int64_t tile_columns_offset(int64_t assignments) {
    return tile_trellis_offset(assignments) + 2 * 3 * assignments;
  }
  __host__ __device__ static constexpr int64_t ints(int64_t assignments) {
    return tile_columns_offset(assignments) + 3 * assignments;
  }
};

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
  return wide ? static_cast<const int64_t*>(ptr)[i]
              : static_cast<const int32_t*>(ptr)[i];
}
__device__ __forceinline__ int64_t local_expert(const Routing& routing, int64_t assignment) {
  const int64_t global = integer_at(routing.ids, routing.ids64, assignment);
  if (global < 0 || global >= routing.global_experts) return -1;
  const int64_t local = integer_at(routing.expert_map, routing.map64, global);
  return local >= 0 && local < routing.local_experts ? local : -1;
}
// [trellis, suh, svh] of one projection of one local expert.
__device__ __forceinline__ const int64_t* expert_pointers(const Routing& routing,
                                                          int64_t expert, int projection) {
  return routing.pointers + expert * 9 + projection * 3;
}
__device__ __forceinline__ const half* expert_suh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[1]);
}
__device__ __forceinline__ const half* expert_svh(const Routing& routing, int64_t expert,
                                                  int projection) {
  return reinterpret_cast<const half*>(expert_pointers(routing, expert, projection)[2]);
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

// Groups the local assignments by local expert into tiles of up to
// block_rows rows (one block, every call), records each sorted entry's
// expert and each assignment's expert, and each row tile's trellis address
// and tile columns per projection. Order inside an expert is arbitrary: each
// assignment's result depends only on its own row, never on its position or
// neighbours.
template <int kIntermediate, int kBitsX2>
__device__ __forceinline__ void route_block(const Routing& routing, int64_t assignments,
                                            int block_rows, int* table) {
  extern __shared__ int route_shared[];
  const int experts = static_cast<int>(routing.local_experts);
  int* cursor = route_shared;              // counts, then fill cursors
  int* row_begin = cursor + experts;
  int* tile_begin = row_begin + experts;   // experts + 1 entries
  for (int e = threadIdx.x; e < experts; e += blockDim.x) cursor[e] = 0;
  __syncthreads();
  int* by_assignment = table + RouteLayout::by_assignment_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    by_assignment[a] = static_cast<int>(e);
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
    tiles += (cursor[e] + block_rows - 1) / block_rows;
  }
  int row_at, tile_at, row_total, tile_total;
  block_scan2(rows, tiles, row_at, tile_at, row_total, tile_total);
  for (int e = e0; e < e1; ++e) {
    const int count = cursor[e];
    row_begin[e] = row_at;
    tile_begin[e] = tile_at;
    cursor[e] = row_at;
    row_at += count;
    tile_at += (count + block_rows - 1) / block_rows;
  }
  if (threadIdx.x == 0) {
    tile_begin[experts] = tile_total;
    table[0] = row_total;
    table[1] = tile_total;
  }
  __syncthreads();
  int4* tile_table = reinterpret_cast<int4*>(table + RouteLayout::tiles_offset(assignments));
  for (int t = threadIdx.x; t < tile_total; t += blockDim.x) {
    // Last expert whose first tile is at or before t (empty experts share
    // their successor's start, so the search lands on the owner).
    int lo = 0;
    int hi = experts;
    while (hi - lo > 1) {
      const int mid = (lo + hi) >> 1;
      if (tile_begin[mid] <= t) lo = mid; else hi = mid;
    }
    const int first = row_begin[lo] + (t - tile_begin[lo]) * block_rows;
    const int end = lo + 1 < experts ? row_begin[lo + 1] : row_total;
    tile_table[t] = make_int4(lo, first, min(block_rows, end - first), 0);
    int64_t* tile_trellis = reinterpret_cast<int64_t*>(
        table + RouteLayout::tile_trellis_offset(assignments));
    int* tile_columns = table + RouteLayout::tile_columns_offset(assignments);
    for (int q = 0; q < 3; ++q) {
      const int64_t* meta = routing.metadata + int64_t(lo) * 12 + q * 4;
      // Served shape only. Gate/up are (hidden, I); down is (I, hidden).
      // bits_x2 is the instance; codebook 2 is mul1. Anything else traps.
      constexpr int64_t kHidden = 5120;
      const int64_t in = q < 2 ? kHidden : kIntermediate;
      const int64_t out = q < 2 ? kIntermediate : kHidden;
      if (meta[0] != in || meta[1] != out || meta[2] != kBitsX2 || meta[3] != 2) __trap();
      tile_trellis[t * 3 + q] = routing.pointers[int64_t(lo) * 9 + q * 3];
      tile_columns[t * 3 + q] = static_cast<int>(meta[1] / 16);
    }
  }
  int* sorted = table + RouteLayout::sorted_offset();
  int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  for (int64_t a = threadIdx.x; a < assignments; a += blockDim.x) {
    const int64_t e = local_expert(routing, a);
    if (e >= 0) {
      const int at = atomicAdd(&cursor[e], 1);
      sorted[at] = static_cast<int>(a);
      sorted_experts[at] = static_cast<int>(e);
    }
  }
}

__device__ __forceinline__ double silu(double g) { return g / __dadd_rn(1.0, exp(-g)); }

// Lane order of a K-long row (MMA B operand): k step k / 16, then lane t's k
// values 2t, 2t + 1, 2t + 8, 2t + 9 (t = (k % 8) / 2).
__device__ __forceinline__ int64_t lane_position(int64_t k) {
  const int kk = static_cast<int>(k & 15);
  return (k & ~int64_t(15)) + ((kk & 7) >> 1) * 4 + ((kk >> 3) << 1) + (kk & 1);
}

// The tensor-core B operand of one row: e per scale group (the peak over the
// group's warps) puts max |v| 2^e in [2^14, 2^15) (2^e and 2^-e stay normal
// FP32), hi = f16(v 2^e) (exact scaling), lo = f16(v 2^e - hi), stored at
// lane_position(k) of the row's high and low FP16 rows.
template <int kVals>
__device__ __forceinline__ float warp_peak(const float (&v)[kVals]) {
  float peak = 0.0f;
#pragma unroll
  for (int i = 0; i < kVals; ++i) peak = fmaxf(peak, fabsf(v[i]));
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, d));
  return peak;
}
__device__ __forceinline__ int halves_exponent(float peak) {
  const int biased = static_cast<int>(__float_as_uint(peak) >> 23);
  return peak == 0.0f ? 0 : min(141 - biased, 126);
}
// The lane's values at k = first + lane + 32 i, scaled by 2^e.
template <int kVals>
__device__ __forceinline__ void store_scaled_halves(const float (&v)[kVals], int lane, int first,
                                                    int e, __half* hi, __half* lo) {
  const float scale = __uint_as_float(uint32_t(127 + e) << 23);
#pragma unroll
  for (int i = 0; i < kVals; ++i) {
    const int64_t at = lane_position(first + lane + 32 * i);
    const float y = __fmul_rn(v[i], scale);
    const __half h = __float2half_rn(y);
    hi[at] = h;
    lo[at] = __float2half_rn(__fsub_rn(y, __half2float(h)));
  }
}

// Where the down GEMM wants its input: per assignment [hi | lo] FP16 rows of
// ipad halves, and one 2^-e per (assignment, down K slice) (the same value
// for every slice: one exponent per row).
struct DownHalves {
  __half* halves;
  float* unscale;
};

// One task: the routing entry at sorted position pos, 128-column block
// `block`; returns false past the routed count, else the transformed values
// (lane holds columns block * 128 + lane + 32 i) and the assignment. Gate and
// up arrive as kSlices K-slice partials (slice_stride apart), summed in
// increasing slice order; then the gate and up output transforms (H128, svh),
// SiLU(gate) * up, down suh and the down H128, all in FP64. Three dependent
// memory round trips: the routing table (count, sorted entry and its expert
// together), then all partials with the expert's pointers, then the svh/suh
// values.
template <int kSlices>
__device__ __forceinline__ bool activation_block(const Routing& routing, const int* table,
                                                 int64_t assignments, int64_t intermediate,
                                                 int64_t ipad, const float* gate, const float* up,
                                                 int64_t slice_stride, int64_t pos, int64_t block,
                                                 int lane, double limit, double (&v)[4],
                                                 int64_t& assignment) {
  const int* sorted = table + RouteLayout::sorted_offset();
  const int* sorted_experts = table + RouteLayout::experts_offset(assignments);
  const int count = __ldg(table);
  assignment = __ldg(sorted + pos);
  const int64_t expert = __ldg(sorted_experts + pos);
  if (pos >= count) return false;
  const int64_t base = block * 128 + lane;
  const half* gate_svh = expert_svh(routing, expert, 0);
  const half* up_svh = expert_svh(routing, expert, 1);
  const half* down_suh = expert_suh(routing, expert, 2);
  float gate_scale[4];
  float up_scale[4];
  float down_scale[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    gate_scale[i] = __half2float(gate_svh[base + 32 * i]);
    up_scale[i] = __half2float(up_svh[base + 32 * i]);
    down_scale[i] = __half2float(down_suh[base + 32 * i]);
  }
  // Sum K-slice partials as they load, increasing slice order. Adding +0
  // first is bitwise the same as starting from slice 0, and it keeps the
  // 16-slice case from holding gs/us[16][4].
  double g[4] = {0.0, 0.0, 0.0, 0.0};
  double u[4] = {0.0, 0.0, 0.0, 0.0};
#pragma unroll
  for (int s = 0; s < kSlices; ++s) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      g[i] = __dadd_rn(g[i], double(gate[s * slice_stride + assignment * ipad + base + 32 * i]));
      u[i] = __dadd_rn(u[i], double(up[s * slice_stride + assignment * ipad + base + 32 * i]));
    }
  }
  hadamard128_warp_f64(g, lane);
  hadamard128_warp_f64(u, lane);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t col = base + 32 * i;
    double gv = __dmul_rn(g[i], double(gate_scale[i]));
    double uv = __dmul_rn(u[i], double(up_scale[i]));
    if (limit > 0.0) {
      gv = fmin(gv, limit);
      uv = fmin(fmax(uv, -limit), limit);
    }
    v[i] = 0.0;
    if (col < intermediate) {
      v[i] = __dmul_rn(__dmul_rn(silu(gv), uv), double(down_scale[i]));
    }
  }
  hadamard128_warp_f64(v, lane);
  return true;
}

// The activation stage: rounded once to FP32 and written as the down GEMM's
// tensor-core operand with one 2^e per row (down_slices copies of 2^-e). One
// thread block per sorted position (kRowBlocks warps = ipad / 128, one per
// H128 block), the row peak from the warps' peaks in shared memory.
template <int kSlices, int kRowBlocks>
__global__ void __launch_bounds__(32 * kRowBlocks)
activate_down_row(Routing routing, const int* table, int64_t assignments, int64_t intermediate,
                  int64_t ipad, const float* gate, const float* up, int64_t slice_stride,
                  DownHalves halves, int down_slices, double limit) {
  __shared__ float warp_peaks[kRowBlocks];
  const int lane = threadIdx.x & 31;
  const int block = threadIdx.x >> 5;
  // Uniform trip count per thread block (barriers inside).
  for (int64_t pos = blockIdx.x; pos < assignments; pos += gridDim.x) {
    double v[4];
    int64_t assignment;
    if (!activation_block<kSlices>(routing, table, assignments, intermediate, ipad, gate, up,
                                   slice_stride, pos, block, lane, limit, v, assignment)) {
      break;
    }
    float f[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __double2float_rn(v[i]);
    const float peak = warp_peak(f);
    if (lane == 0) warp_peaks[block] = peak;
    __syncthreads();
    float row_peak = 0.0f;
#pragma unroll
    for (int w = 0; w < kRowBlocks; ++w) row_peak = fmaxf(row_peak, warp_peaks[w]);
    const int e = halves_exponent(row_peak);
    __half* hi = halves.halves + assignment * 2 * ipad;
    store_scaled_halves(f, lane, block * 128, e, hi, hi + ipad);
    if (block == 0 && lane < down_slices) {
      halves.unscale[assignment * down_slices + lane] = __uint_as_float(uint32_t(127 - e) << 23);
    }
    __syncthreads();
  }
}

// One block per (token, 128-column block), one warp per route (blockDim =
// 32 * min(topk, 32), at least 4 warps). Warp w handles routes w, w + warps,
// ...: the local route's down output (the sum of kSlices K-slice partials,
// slice_stride apart, in increasing slice order) transform (H128, svh) in
// FP64 into shared memory, every route's loads in flight at once. Then
// threads 0..127 apply exactly one routing-weight FMA per local route in
// route order (FP64) and round to FP32, then to BF16. Local experts come from
// the routing table's by-assignment array. Dynamic shared memory:
// combine_shared_bytes(topk).
constexpr int kCombineMaxThreads = 1024;
size_t combine_shared_bytes(int64_t topk) {
  return size_t(topk) * (128 * sizeof(double) + sizeof(float) + sizeof(int));
}
template <int kSlices>
__global__ void __launch_bounds__(kCombineMaxThreads)
combine_routes(const float* down, const float* weights, Routing routing, const int* table,
               int64_t rows, int64_t hidden, int64_t hpad, int64_t slice_stride,
               __nv_bfloat16* out) {
  extern __shared__ __align__(16) unsigned char combine_shared[];
  const int64_t topk = routing.topk;
  double* values = reinterpret_cast<double*>(combine_shared);      // [topk][128]
  float* route_weight = reinterpret_cast<float*>(values + topk * 128);
  int* route_local = reinterpret_cast<int*>(route_weight + topk);
  const int* by_assignment = table + RouteLayout::by_assignment_offset(rows * topk);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t blocks = hpad / 128;
  for (int64_t task = blockIdx.x; task < rows * blocks; task += gridDim.x) {
    const int64_t row = task / blocks;
    const int64_t col0 = (task % blocks) * 128;
    for (int64_t r = warp; r < topk; r += blockDim.x / 32) {
      const int64_t assignment = row * topk + r;
      const int64_t expert = __ldg(by_assignment + assignment);
      if (lane == 0) {
        route_local[r] = expert >= 0;
        route_weight[r] = weights[assignment];
      }
      if (expert < 0) continue;
      const half* svh = expert_svh(routing, expert, 2);
      float v[kSlices][4];
#pragma unroll
      for (int s = 0; s < kSlices; ++s) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          v[s][i] = down[s * slice_stride + assignment * hpad + col0 + lane + 32 * i];
        }
      }
      float scale[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) scale[i] = __half2float(svh[col0 + lane + 32 * i]);
      double x[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        x[i] = v[0][i];
#pragma unroll
        for (int s = 1; s < kSlices; ++s) x[i] = __dadd_rn(x[i], double(v[s][i]));
      }
      hadamard128_warp_f64(x, lane);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[r * 128 + lane + 32 * i] = __dmul_rn(x[i], double(scale[i]));
      }
    }
    __syncthreads();
    if (threadIdx.x < 128) {
      double sum = 0.0;
      for (int64_t r = 0; r < topk; ++r) {
        if (route_local[r]) {
          sum = __fma_rn(values[r * 128 + threadIdx.x], double(route_weight[r]), sum);
        }
      }
      const int64_t col = col0 + threadIdx.x;
      if (col < hidden) out[row * hidden + col] = __float2bfloat16_rn(__double2float_rn(sum));
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Lane-tile expert GEMM (bits_x2 10, codebook 2).
//
// An item is one warp's kNT output n16 tiles x one K slice of one route tile;
// no shared memory and no block barriers. The warp decodes its own 16x16
// trellis tiles straight into m16n8k16 A fragments (A = W^T: 16 output
// columns x 16 k; decode_lane_fragment_mul1) and multiplies them on the tensor
// cores (FP16 in, FP32 out, C = 0 per k16 step) with B = 16 k x 8 columns: per
// group of four route rows, columns 0..3 hold the rows' high FP16 parts and
// 4..7 the low parts of their activations scaled by a power of two per row
// and scale group (store_scaled_halves; produced once per call by
// route_and_gate_up_input and activate_down_row, read straight from global
// memory). The k16 results are summed in FP32 in increasing k per K slice,
// high and low apart; the slice partial (hi + lo) 2^-e goes to the workspace
// and the consumer kernel sums the slices in increasing slice order (FP64).
// Everything is fixed-order: a row's result never depends on other rows.
constexpr int kLaneRows = 8;                    // route rows per tile
constexpr int64_t kLaneMaxAssignments = 1024;   // routed rows the workspace layout covers
constexpr int kLaneThreads = 256;
constexpr int kLaneAhead = 3;                   // k steps prefetched to L2

struct LaneProblem {
  const int* table;
  int64_t assignments;       // routing table capacity
  int k_tiles;               // K / 16
  int n_tiles;               // N / 16
  int projections;           // 2: gate (out0) and up (out1); 1: down (out0)
  int first_projection;
  float* out0;               // [slice][assignment][N] partials
  float* out1;
  int64_t slice_stride;
  // B operand: per (assignment, projection) row [hi | lo] FP16 rows of K
  // halves in lane order, and 2^-e per (assignment, projection, K slice).
  const __half* halves;
  const float* unscale;
};

__device__ __forceinline__ void mma_f16_f32(const uint32_t (&a)[4], uint2 b, float (&d)[4]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b.x), "r"(b.y), "f"(0.0f));
}

// One warp: kNT output n16 tiles x kKT k16 steps of one route tile with
// kGroups n8 row groups (row_of: lane r holds tile row r's assignment), k
// outer: the kNT tiles of a k step are adjacent in the trellis (immediate
// offsets from three per-lane word pointers); the next k step's tiles and B
// fragments are in flight while this one decodes.
template <int kGroups, int kNT, int kKT, int kBitsX2, bool kAligned>
__device__ __forceinline__ void lane_tiles(const LaneProblem& p, int pi, int slice, int row_of,
                                           int rows, const uint16_t* trellis, int64_t tiles_n,
                                           int n0, int lane, const LaneWindow<kBitsX2>& window,
                                           float* out) {
  using Ring = TileRing<kBitsX2>;
  constexpr int kTileWords = Ring::kTileHalfwords / 2;
  const int kt0 = slice * kKT;
  const int slices = p.k_tiles / kKT;
  const int64_t row_halves = 2 * int64_t(p.k_tiles) * 16;
  const int t = lane & 3;
  const int gq = lane >> 2;
  // B source of this lane: column gq of each group = hi (gq < 4) or lo of
  // row 4g + gq % 4, k values 2t, 2t + 1, 2t + 8, 2t + 9 of each k step.
  const uint2* bsrc[kGroups];
  bool bok[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    const int r = 4 * g + (gq & 3);
    bok[g] = r < rows;
    const int64_t assignment = __shfl_sync(0xffffffffu, row_of, bok[g] ? r : 0);
    bsrc[g] = reinterpret_cast<const uint2*>(
                  p.halves + (assignment * p.projections + pi) * row_halves +
                  (gq >> 2) * (row_halves / 2) + kt0 * 16) + t;
  }
  // 2^-e of the rows this lane stores: 4g + 2t + c (lanes t < 2).
  float unscale[kGroups][2];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 4 * g + 2 * (t & 1) + c;
      const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
      unscale[g][c] = __ldg(p.unscale + (assignment * p.projections + pi) * slices + slice);
    }
  }
  const uint16_t* step = trellis + (int64_t(kt0) * tiles_n + n0) * Ring::kTileHalfwords;
  const int64_t stride = tiles_n * Ring::kTileHalfwords;
  const uint32_t* wp[Ring::kLoadWords];
#pragma unroll
  for (int i = 0; i < Ring::kLoadWords; ++i) {
    wp[i] = reinterpret_cast<const uint32_t*>(step) + window.word[i];
  }
  uint32_t words[kNT][Ring::kLoadWords];
  auto fetch = [&](int j) {
    if constexpr (kAligned) {
#pragma unroll
      for (int i = 0; i < Ring::kLoadWords; ++i) words[j][i] = __ldg(wp[i] + j * kTileWords);
    } else {
      load_lane_words<kBitsX2>(step + j * Ring::kTileHalfwords, window, false, words[j]);
    }
  };
#pragma unroll
  for (int j = 0; j < kNT; ++j) fetch(j);
  uint2 bnext[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) bnext[g] = bok[g] ? __ldg(bsrc[g]) : make_uint2(0u, 0u);
  float acc[kNT][kGroups][4];
#pragma unroll
  for (int j = 0; j < kNT; ++j) {
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[j][g][i] = 0.0f;
    }
  }
  // L2 prefetch kLaneAhead k steps ahead: lane = 128-byte line of the step.
  constexpr int kStepBytes = kNT * Ring::kTileHalfwords * 2;
  const char* ahead = reinterpret_cast<const char*>(step + kLaneAhead * stride) +
                      min(lane * 128, kStepBytes - 1);
#pragma unroll 1
  for (int kt = 0; kt < kKT; ++kt) {
    if (lane * 128 < kStepBytes + 127 && kt + kLaneAhead < kKT) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(ahead));
    }
    ahead += stride * 2;
    const bool more = kt + 1 < kKT;
    uint2 b[kGroups];
#pragma unroll
    for (int g = 0; g < kGroups; ++g) {
      b[g] = bnext[g];
      if (more && bok[g]) bnext[g] = __ldg(bsrc[g] + 4 * (kt + 1));
    }
    if constexpr (kAligned) {
#pragma unroll
      for (int i = 0; i < Ring::kLoadWords; ++i) wp[i] += stride / 2;
    } else {
      step += stride;
    }
#pragma unroll
    for (int j = 0; j < kNT; ++j) {
      uint32_t a[4];
      decode_lane_fragment_mul1<kBitsX2>(words[j], window.shift, a);
      if (more) fetch(j);
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        float d[4];
        mma_f16_f32(a, b[g], d);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][g][i] = __fadd_rn(acc[j][g][i], d[i]);
      }
    }
  }
  // Lanes t < 2 hold the high sums of rows 4g + 2t + {0, 1}; lane t + 2 the
  // low sums of the same rows and columns.
  const int64_t n_stride = int64_t(p.n_tiles) * 16;
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 4 * g + 2 * (t & 1) + c;
      const int64_t assignment = __shfl_sync(0xffffffffu, row_of, r < rows ? r : 0);
#pragma unroll
      for (int j = 0; j < kNT; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int i = 2 * h + c;
          const float low = __shfl_xor_sync(0xffffffffu, acc[j][g][i], 2);
          if (t < 2 && r < rows) {
            out[assignment * n_stride + (n0 + j) * 16 + gq + 8 * h] =
                __fmul_rn(__fadd_rn(acc[j][g][i], low), unscale[g][c]);
          }
        }
      }
    }
  }
}

// The unaligned-trellis variant, out of line: inlining it beside the aligned
// one costs the hot loops about 3 % (register allocation and code layout).
template <int kGroups, int kNT, int kKT, int kBitsX2>
__device__ __noinline__ void lane_tiles_unaligned(LaneProblem p, int pi, int slice, int row_of,
                                                  int rows, const uint16_t* trellis,
                                                  int64_t tiles_n, int n0, int lane,
                                                  float* out) {
  const LaneWindow<kBitsX2> window(lane);
  lane_tiles<kGroups, kNT, kKT, kBitsX2, false>(p, pi, slice, row_of, rows, trellis, tiles_n, n0,
                                               lane, window, out);
}

// Warp items: (route tile, projection, K slice, column group of kNT n16
// tiles), column groups fastest so a block's warps share B rows in L1.
template <int kNT, int kKT, int kBitsX2>
__global__ void __launch_bounds__(kLaneThreads, 1)
moe_lane_kernel(LaneProblem p) {
  const int lane = threadIdx.x & 31;
  const LaneWindow<kBitsX2> window(lane);
  const int slices = p.k_tiles / kKT;
  const int groups_n = p.n_tiles / kNT;
  const int64_t per_tile = int64_t(p.projections) * slices * groups_n;
  const int64_t items = int64_t(p.table[1]) * per_tile;
  const int* sorted = p.table + RouteLayout::sorted_offset();
  const int4* tiles = reinterpret_cast<const int4*>(p.table + RouteLayout::tiles_offset(p.assignments));
  const long long* tile_trellis = reinterpret_cast<const long long*>(
      p.table + RouteLayout::tile_trellis_offset(p.assignments));
  const int* tile_columns = p.table + RouteLayout::tile_columns_offset(p.assignments);
  const int64_t warps = int64_t(gridDim.x) * (kLaneThreads / 32);
  for (int64_t item = int64_t(blockIdx.x) * (kLaneThreads / 32) + (threadIdx.x >> 5);
       item < items; item += warps) {
    const int64_t tile_index = item / per_tile;
    int rest = static_cast<int>(item % per_tile);
    const int column_group = rest % groups_n;
    rest /= groups_n;
    const int slice = rest % slices;
    const int pi = rest / slices;
    const int projection = p.first_projection + pi;
    // One round trip for the tile, its trellis address and columns (written
    // by the route kernel, which also checked the format).
    const int4 tile = tiles[tile_index];
    const uint16_t* trellis = reinterpret_cast<const uint16_t*>(
        __ldg(tile_trellis + tile_index * 3 + projection));
    const int64_t tiles_n = __ldg(tile_columns + tile_index * 3 + projection);
    const int rows = tile.z;
    const int row_of = lane < rows ? sorted[tile.y + lane] : 0;
    const int n0 = column_group * kNT;
    const bool aligned = (reinterpret_cast<uintptr_t>(trellis) & 3) == 0;
    float* out = (pi == 0 ? p.out0 : p.out1) + slice * p.slice_stride;
#define EXL3_LANE_GROUPS(G)                                                                        \
    if (aligned) {                                                                                 \
      lane_tiles<G, kNT, kKT, kBitsX2, true>(p, pi, slice, row_of, rows, trellis, tiles_n, n0,     \
                                            lane, window, out);                                    \
    } else {                                                                                       \
      lane_tiles_unaligned<G, kNT, kKT, kBitsX2>(p, pi, slice, row_of, rows, trellis, tiles_n, n0, \
                                                lane, out);                                        \
    }
    if (rows <= 4) {
      EXL3_LANE_GROUPS(1);
    } else {
      EXL3_LANE_GROUPS(2);
    }
#undef EXL3_LANE_GROUPS
  }
}

// One launch: block 0 routes (route_block) while the other blocks form the
// gate/up tensor-core B operand straight from topk_ids (independent of the
// routing table). Hidden 5120 is 40 H128 blocks. 20 warps (640 threads) each
// own two H128 blocks: the short path's one 2^e per row reduces over all of
// them, and a long-path 1280-column group is blocks 2w and 2w+1 (kSplit 2).
// x * suh and H128 stay in FP64, rounded once to FP32. A group spans kSplit
// GEMM K slices, each given its own copy of 2^-e.
template <int kScaleK, int kSplit, int kIntermediate, int kBitsX2>
__global__ void __launch_bounds__(kRouteInputThreads)
route_and_gate_up_input(Routing routing, int64_t assignments, int block_rows, int* table,
                        const __nv_bfloat16* x, int64_t hidden, int64_t hpad, __half* halves,
                        float* unscale) {
  if (blockIdx.x == 0) {
    route_block<kIntermediate, kBitsX2>(routing, assignments, block_rows, table);
    return;
  }
  constexpr int kH128PerWarp = 2;
  constexpr int kBlocks = kScaleK / 128;
  constexpr int kWarps = kRouteInputThreads / 32;
  constexpr int kWarpsPerGroup = kBlocks / kH128PerWarp;
  static_assert(kBlocks % kH128PerWarp == 0, "a scale group holds whole H128 pairs");
  static_assert(kWarps % kWarpsPerGroup == 0 && kSplit <= 32,
                "input warps must tile whole groups");
  __shared__ float block_peaks[2][kWarps][kH128PerWarp];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int group = warp / kWarpsPerGroup;
  const int local = warp - group * kWarpsPerGroup;
  const int64_t slices = hpad / kScaleK * kSplit;
  // Uniform trip count per thread block (barriers inside).
  for (int64_t assignment = blockIdx.x - 1; assignment < assignments;
       assignment += gridDim.x - 1) {
    const __nv_bfloat16* row = x + (assignment / routing.topk) * hidden;
    const int64_t expert = local_expert(routing, assignment);
    float saved[2][kH128PerWarp][4];
#pragma unroll
    for (int sub = 0; sub < kH128PerWarp; ++sub) {
      const int64_t base = (int64_t(warp) * kH128PerWarp + sub) * 128;
      float xs[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t col = base + lane + 32 * i;
        xs[i] = col < hidden ? __bfloat162float(row[col]) : 0.0f;
      }
      float v[2][4] = {};
      if (expert >= 0) {
        const half* suh[2] = {expert_suh(routing, expert, 0), expert_suh(routing, expert, 1)};
        float scale[2][4];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int64_t col = base + lane + 32 * i;
            scale[q][i] = col < hidden ? __half2float(suh[q][col]) : 0.0f;
          }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          double block[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) block[i] = __dmul_rn(double(xs[i]), double(scale[q][i]));
          hadamard128_warp_f64(block, lane);
#pragma unroll
          for (int i = 0; i < 4; ++i) v[q][i] = __double2float_rn(block[i]);
        }
      }
#pragma unroll
      for (int q = 0; q < 2; ++q) {
#pragma unroll
        for (int i = 0; i < 4; ++i) saved[q][sub][i] = v[q][i];
        const float peak = warp_peak(v[q]);
        if (lane == 0) block_peaks[q][warp][sub] = peak;
      }
    }
    __syncthreads();
    if (expert >= 0) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        float group_peak = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarpsPerGroup; ++w) {
#pragma unroll
          for (int s = 0; s < kH128PerWarp; ++s) {
            group_peak = fmaxf(group_peak, block_peaks[q][group * kWarpsPerGroup + w][s]);
          }
        }
        const int e = halves_exponent(group_peak);
#pragma unroll
        for (int sub = 0; sub < kH128PerWarp; ++sub) {
          __half* hi = halves + (assignment * 2 + q) * 2 * hpad + int64_t(group) * kScaleK;
          store_scaled_halves(saved[q][sub], lane, (local * kH128PerWarp + sub) * 128, e, hi,
                              hi + hpad);
        }
        if (local == 0 && lane < kSplit) {
          unscale[(assignment * 2 + q) * slices + group * kSplit + lane] =
              __uint_as_float(uint32_t(127 - e) << 23);
        }
      }
    }
    __syncthreads();
  }
}

// DS V4.1 shape. Hidden is 5120. Column groups are 4 n16 tiles so they divide
// 40 (I = 640), 32 (I = 512) and 320 (down N). Gate/up K is 320 n16 tiles.
// Short calls: 20-tile slices (16 slices) and one 2^e over the whole row
// (kSplit 16). Long calls, from 160 assignments: 40-tile slices (8 slices)
// and one 2^e per 1280 columns (kSplit 2). Down K is I/16 tiles. Short is 20
// tiles at I = 640 and 16 at I = 512 (two slices); long is the whole K (one
// slice). The 160-assignment switch is unchanged.
template <int kIntermediate, int kBitsX2>
struct LaneShape {
  static constexpr int64_t kHidden = 5120;
  static constexpr int64_t kInter = kIntermediate;
  static constexpr int kBits = kBitsX2;
  static constexpr int kColumnTiles = 4;
  static constexpr int kGateUpShortScaleK = 5120;
  static constexpr int kGateUpShortSliceTiles = 20;
  static constexpr int kGateUpLongScaleK = 1280;
  static constexpr int kGateUpLongSliceTiles = 40;
  static constexpr int kDownShortSliceTiles = kIntermediate == 640 ? 20 : 16;
  static constexpr int kDownLongSliceTiles = kIntermediate / 16;
  static constexpr int64_t kLongSliceAssignments = 160;
  static constexpr int64_t kLongDownAssignments = 160;
  static constexpr int kGateUpShortSlices = int(kHidden / (16 * kGateUpShortSliceTiles));
  static constexpr int kGateUpLongSlices = int(kHidden / (16 * kGateUpLongSliceTiles));
  static constexpr int kDownShortSlices = int(kInter / (16 * kDownShortSliceTiles));
  static constexpr int kDownLongSlices = int(kInter / (16 * kDownLongSliceTiles));
  static constexpr int kGateUpSlices = kGateUpShortSlices;
  static constexpr int kDownSlices = kDownShortSlices;
  static constexpr int kActivateBlocks = kIntermediate / 128;
  static constexpr int kActivateThreads = kIntermediate / 4;
  static_assert(kBits == 4 || kBits == 6, "bits_x2 instance");
  static_assert(kInter == 512 || kInter == 640, "intermediate instance");
  static_assert(kGateUpShortSlices == 16 && kGateUpLongSlices == 8, "gate/up slice counts");
  static_assert(kDownShortSlices == 2 && kDownLongSlices == 1, "down slice counts");
  static_assert(kGateUpShortScaleK / (16 * kGateUpShortSliceTiles) == 16, "short kSplit");
  static_assert(kGateUpLongScaleK / (16 * kGateUpLongSliceTiles) == 2, "long kSplit");
  static_assert(kGateUpShortScaleK / (16 * kGateUpShortSliceTiles) <= 32, "kSplit bound");
  static_assert(kHidden % kGateUpShortScaleK == 0 && kHidden % kGateUpLongScaleK == 0,
                "scale groups divide K");
  static_assert(kHidden % (16 * kColumnTiles) == 0 && kInter % (16 * kColumnTiles) == 0,
                "column groups divide N");
  static_assert(kInter % (16 * kDownShortSliceTiles) == 0, "down short slices");
  static_assert(kInter % (16 * kDownLongSliceTiles) == 0, "down long slices");
  static_assert(kGateUpShortSlices >= kGateUpLongSlices && kDownShortSlices >= kDownLongSlices &&
                    kDownShortSlices <= 32,
                "workspace covers the short path");
  static_assert(kRouteInputThreads / 32 * 2 == kHidden / 128, "two H128 blocks per warp");
};

// Both intermediates use the same slice counts, so the workspace formula is
// shared. ipad still scales the gate/up partials.
constexpr int kGateUpSlices = LaneShape<640, 6>::kGateUpSlices;
constexpr int kDownSlices = LaneShape<640, 6>::kDownSlices;
static_assert(kGateUpSlices == LaneShape<512, 4>::kGateUpSlices, "shared gate/up workspace");
static_assert(kDownSlices == LaneShape<512, 4>::kDownSlices, "shared down workspace");

template <int kI, int kBits>
void check_shape(int64_t hpad, int64_t ipad, int64_t assignments) {
  using S = LaneShape<kI, kBits>;
  TORCH_CHECK(hpad == S::kHidden && ipad == S::kInter,
              "EXL3 MoE v2 supports padded hidden 5120 and intermediate 640 or 512 only");
  TORCH_CHECK(assignments <= kLaneMaxAssignments,
              "EXL3 MoE v2 supports at most 1024 routed assignments per call");
}

template <int kBits, int kKT>
void launch_lane(const LaneProblem& p, int64_t max_tiles, cudaStream_t stream) {
  constexpr int kColumnTiles = 4;
  auto* kernel = &moe_lane_kernel<kColumnTiles, kKT, kBits>;
  static const int64_t resident = [kernel] {
    int device = 0;
    int sms = 0;
    int per_sm = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel,
                                                                 kLaneThreads, 0));
    return int64_t(std::max(sms, 1)) * std::max(per_sm, 1);
  }();
  const int64_t per_tile = int64_t(p.projections) * (p.k_tiles / kKT) *
                           (p.n_tiles / kColumnTiles);
  constexpr int64_t kWarps = kLaneThreads / 32;
  const int64_t grid = std::min((max_tiles * per_tile + kWarps - 1) / kWarps, resident);
  if (grid <= 0) return;
  kernel<<<static_cast<unsigned int>(grid), kLaneThreads, 0, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// FP32 workspace elements before the routing table: gate, up and down slice
// partials; the gate/up and down B operands (FP16 hi and lo rows) and their
// 2^-e per (row, slice), rounded up to whole 16-byte groups.
int64_t lane_workspace(int64_t assignments, int64_t hpad, int64_t ipad) {
  const int64_t gate_up = checked_mul(2 * kGateUpSlices, ipad);
  const int64_t down = checked_mul(kDownSlices, hpad);
  const int64_t operands = checked_add(checked_mul(2, hpad), ipad);   // FP16 hi + lo
  const int64_t unscale = (assignments * (2 * kGateUpSlices + kDownSlices) + 3) / 4 * 4;
  return checked_add(
      checked_mul(assignments, checked_add(checked_add(gate_up, down), operands)), unscale);
}

}  // namespace
}  // namespace moe_v2

int64_t moe_workspace_size_v2(int64_t max_rows, int64_t hidden_size,
                              int64_t intermediate_size, int64_t topk,
                              int64_t local_experts) {
  using namespace moe_v2;
  TORCH_CHECK(max_rows >= 0 && topk >= 0 && local_experts >= 0,
              "EXL3 rows/topk/local_experts must be nonnegative");
  const int64_t hpad = padded_size(hidden_size);
  const int64_t ipad = padded_size(intermediate_size);
  TORCH_CHECK(hpad == 5120 && (ipad == 640 || ipad == 512),
              "EXL3 MoE v2 supports hidden 5120 and intermediate 640 or 512 only");
  if (local_experts == 0) return 0;
  const int64_t assignments = checked_mul(max_rows, topk);
  TORCH_CHECK(assignments <= kLaneMaxAssignments,
              "EXL3 MoE v2 supports at most 1024 routed assignments per call");
  return workspace_elements(
      checked_add(lane_workspace(assignments, hpad, ipad), RouteLayout::ints(assignments)));
}

template <int kI, int kBits>
void moe_out_v2_shaped(const torch::Tensor& x, const torch::Tensor& topk_ids,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& packed_ptrs,
                       const torch::Tensor& packed_meta,
                       const torch::Tensor& expert_map, double limit,
                       torch::Tensor& out, torch::Tensor& workspace) {
  using namespace moe_v2;
  using S = LaneShape<kI, kBits>;
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t experts = packed_ptrs.size(0);
  const int64_t hpad = padded_size(hidden);
  const int64_t ipad = padded_size(kI);
  const int64_t assignments = checked_mul(rows, topk);
  check_shape<kI, kBits>(hpad, ipad, assignments);
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  Routing routing{topk_ids.data_ptr(), expert_map.data_ptr(),
                  packed_ptrs.data_ptr<int64_t>(), packed_meta.data_ptr<int64_t>(),
                  expert_map.numel(), experts, topk,
                  topk_ids.scalar_type() == at::kLong, expert_map.scalar_type() == at::kLong};
  float* input = workspace.data_ptr<float>();
  const float* down_out = input;
  int64_t down_stride = 0;
  const int* route_table = nullptr;
  if (experts > 0 && assignments > 0) {
    const int64_t gate_up_stride = assignments * ipad;
    down_stride = assignments * hpad;
    float* gate = input;
    float* up = gate + S::kGateUpSlices * gate_up_stride;
    float* down = up + S::kGateUpSlices * gate_up_stride;
    __half* gate_up_halves = reinterpret_cast<__half*>(down + S::kDownSlices * down_stride);
    __half* down_halves = gate_up_halves + assignments * 2 * 2 * hpad;
    float* gate_up_unscale = reinterpret_cast<float*>(down_halves + assignments * 2 * ipad);
    float* down_unscale = gate_up_unscale + assignments * 2 * S::kGateUpSlices;
    int* table = reinterpret_cast<int*>(
        gate_up_unscale + (assignments * (2 * S::kGateUpSlices + S::kDownSlices) + 3) / 4 * 4);
    route_table = table;
    down_out = down;
    const auto gate_up = [&](auto scale_k, auto slice_tiles) {
      constexpr int kScaleK = decltype(scale_k)::value;
      constexpr int kSplit = kScaleK / (16 * decltype(slice_tiles)::value);
      route_and_gate_up_input<kScaleK, kSplit, kI, kBits>
          <<<launch_blocks(1 + assignments),
             kRouteInputThreads, (3 * experts + 1) * sizeof(int), stream>>>(
              routing, assignments, kLaneRows, table,
              reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), hidden, hpad,
              gate_up_halves, gate_up_unscale);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    const bool long_slices = assignments >= S::kLongSliceAssignments;
    if (long_slices) {
      gate_up(std::integral_constant<int, S::kGateUpLongScaleK>{},
              std::integral_constant<int, S::kGateUpLongSliceTiles>{});
    } else {
      gate_up(std::integral_constant<int, S::kGateUpShortScaleK>{},
              std::integral_constant<int, S::kGateUpShortSliceTiles>{});
    }
    const int64_t max_tiles = std::min<int64_t>(
        assignments, assignments / kLaneRows + std::min<int64_t>(experts, assignments) + 1);
    LaneProblem p{};
    p.table = table;
    p.assignments = assignments;
    p.k_tiles = static_cast<int>(hpad / 16);
    p.n_tiles = static_cast<int>(ipad / 16);
    p.projections = 2;
    p.first_projection = 0;
    p.out0 = gate;
    p.out1 = up;
    p.slice_stride = gate_up_stride;
    p.halves = gate_up_halves;
    p.unscale = gate_up_unscale;
    const bool long_down = assignments >= S::kLongDownAssignments;
    const int down_used = long_down ? S::kDownLongSlices : S::kDownShortSlices;
    const auto activate = [&](auto slices) {
      activate_down_row<decltype(slices)::value, S::kActivateBlocks>
          <<<launch_blocks(assignments), S::kActivateThreads, 0, stream>>>(
              routing, table, assignments, kI, ipad, gate, up, gate_up_stride,
              DownHalves{down_halves, down_unscale}, down_used, limit);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (long_slices) {
      launch_lane<kBits, S::kGateUpLongSliceTiles>(p, max_tiles, stream);
      activate(std::integral_constant<int, S::kGateUpLongSlices>{});
    } else {
      launch_lane<kBits, S::kGateUpShortSliceTiles>(p, max_tiles, stream);
      activate(std::integral_constant<int, S::kGateUpShortSlices>{});
    }
    p.k_tiles = static_cast<int>(ipad / 16);
    p.n_tiles = static_cast<int>(hpad / 16);
    p.projections = 1;
    p.first_projection = 2;
    p.out0 = down;
    p.out1 = nullptr;
    p.slice_stride = down_stride;
    p.halves = down_halves;
    p.unscale = down_unscale;
    if (long_down) {
      launch_lane<kBits, S::kDownLongSliceTiles>(p, max_tiles, stream);
    } else {
      launch_lane<kBits, S::kDownShortSliceTiles>(p, max_tiles, stream);
    }
  } else {
    routing.topk = 0;
  }
  const size_t shared = combine_shared_bytes(routing.topk);
  TORCH_CHECK(shared <= 48 * 1024, "EXL3 MoE topk too large for the combine stage");
  const auto combine = [&](auto slices) {
    combine_routes<decltype(slices)::value>
        <<<launch_blocks(rows * (hpad / 128)),
           32 * static_cast<int>(std::min<int64_t>(std::max<int64_t>(routing.topk, 4), 32)),
           shared, stream>>>(
            down_out, topk_weights.data_ptr<float>(), routing, route_table, rows, hidden, hpad,
            down_stride, reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()));
  };
  if (assignments >= S::kLongDownAssignments) {
    combine(std::integral_constant<int, S::kDownLongSlices>{});
  } else {
    combine(std::integral_constant<int, S::kDownSlices>{});
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_out_v2(const torch::Tensor& x, const torch::Tensor& topk_ids,
                const torch::Tensor& topk_weights,
                const torch::Tensor& packed_ptrs,
                const torch::Tensor& packed_meta,
                const torch::Tensor& expert_map, int64_t intermediate_size,
                int64_t bits_x2, double limit,
                torch::Tensor& out, torch::Tensor& workspace) {
  using namespace moe_v2;
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
  TORCH_CHECK(experts <= kMaxRouteExperts, "EXL3 MoE has too many local experts");
  const int64_t required =
      moe_workspace_size_v2(rows, hidden, intermediate_size, topk, experts);
  TORCH_CHECK(workspace.numel() >= required, "EXL3 MoE workspace is too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "EXL3 MoE workspace must be 16-byte aligned");
  check_mutations(out, workspace,
                  {&x, &topk_ids, &topk_weights, &packed_ptrs, &packed_meta, &expert_map});
  if (rows == 0) return;
  if (intermediate_size == 640 && bits_x2 == 6) {
    moe_out_v2_shaped<640, 6>(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
                              limit, out, workspace);
  } else if (intermediate_size == 512 && bits_x2 == 6) {
    moe_out_v2_shaped<512, 6>(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
                              limit, out, workspace);
  } else if (intermediate_size == 640 && bits_x2 == 4) {
    moe_out_v2_shaped<640, 4>(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
                              limit, out, workspace);
  } else if (intermediate_size == 512 && bits_x2 == 4) {
    moe_out_v2_shaped<512, 4>(x, topk_ids, topk_weights, packed_ptrs, packed_meta, expert_map,
                              limit, out, workspace);
  } else {
    TORCH_CHECK(false,
                "EXL3 TC MoE v2 supports (intermediate, bits_x2) in "
                "{(640,6),(512,6),(640,4),(512,4)} only");
  }
}


}  // namespace exl3
