// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/types.h>

#include <cstdint>

namespace exl3 {

constexpr int64_t kMoePrefillMinRows = 65;
constexpr int64_t kMoePrefillMaxExperts = 4096;

int64_t moe_workspace_size_prefill(int64_t max_rows, int64_t hidden_size,
                                   int64_t intermediate_size, int64_t topk,
                                   int64_t local_experts);

void moe_out_prefill(const torch::Tensor& x, const torch::Tensor& topk_ids,
                     const torch::Tensor& topk_weights,
                     const torch::Tensor& packed_ptrs,
                     const torch::Tensor& packed_meta,
                     const torch::Tensor& expert_map, int64_t intermediate_size,
                     int64_t bits_x2, double limit, torch::Tensor& out,
                     torch::Tensor& workspace);

}  // namespace exl3
