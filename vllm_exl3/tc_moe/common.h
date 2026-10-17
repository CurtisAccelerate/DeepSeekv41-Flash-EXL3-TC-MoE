// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <ATen/MemoryOverlap.h>
#include <torch/types.h>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace exl3 {

inline int64_t checked_add(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 && a <= std::numeric_limits<int64_t>::max() - b,
              "EXL3 workspace size overflow");
  return a + b;
}
inline int64_t checked_mul(int64_t a, int64_t b) {
  TORCH_CHECK(a >= 0 && b >= 0 &&
                  (b == 0 || a <= std::numeric_limits<int64_t>::max() / b),
              "EXL3 workspace size overflow");
  return a * b;
}
inline int64_t workspace_elements(int64_t n) {
  TORCH_CHECK(n <= std::numeric_limits<int64_t>::max() / sizeof(float),
              "EXL3 workspace byte size overflow");
  return n;
}
inline int64_t padded_size(int64_t n) {
  TORCH_CHECK(n > 0 && n <= std::numeric_limits<int>::max() - 127,
              "EXL3 logical dimensions must be positive and fit int32 padding");
  return (n + 127) / 128 * 128;
}
inline void check_padded(int64_t n) {
  TORCH_CHECK(n > 0 && n % 128 == 0 &&
                  n <= std::numeric_limits<int>::max(),
              "EXL3 padded dimensions must be positive H128 multiples fitting int32");
}
inline void check_tensor(const torch::Tensor& t, const torch::Tensor& x,
                         at::ScalarType dtype, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == x.device(), name,
              " must be on the input CUDA device");
  TORCH_CHECK(t.scalar_type() == dtype, name, " has incorrect dtype");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}
inline void check_integer(const torch::Tensor& t, const torch::Tensor& x,
                          const char* name) {
  TORCH_CHECK(t.scalar_type() == at::kInt || t.scalar_type() == at::kLong,
              name, " must have int32 or int64 dtype");
  check_tensor(t, x, t.scalar_type(), name);
}
inline void check_mutations(const torch::Tensor& out,
                            const torch::Tensor& workspace,
                            std::initializer_list<const torch::Tensor*> inputs) {
  at::assert_no_internal_overlap(out);
  at::assert_no_internal_overlap(workspace);
  at::assert_no_overlap(out, workspace);
  for (const auto& input : inputs) {
    at::assert_no_overlap(out, *input);
    at::assert_no_overlap(workspace, *input);
  }
}
inline unsigned int launch_blocks(int64_t tasks) {
  return static_cast<unsigned int>(std::min<int64_t>(tasks, 65535));
}

}  // namespace exl3
