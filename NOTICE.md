# Attribution

The DeepSeek V4.1 Flash port, the row dispatch, the shared workspace, the
EXL3 hook, and the CMP 170HX measurements in this repository are by
CurtisAccelerate (https://github.com/CurtisAccelerate).

The kernels and the host-staged all-reduce started from other people's
code. This repository does not include their weights.

- Tensor-core EXL3 MoE kernels (`moe_v2`, `moe_prefill`) and the host-staged
  all-reduce are ported from [Morrowmake/vllm-cmp170hx](https://github.com/Morrowmake/vllm-cmp170hx)
  at `18305948acba3f4c4d912cdd56f2202d2ebd94e8`, Apache-2.0.
  The published CMP 170HX recipe those numbers are compared with is
  [Morrowmake/qwen38-flash-next-cmp170hx-recipe](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)
  release 1.0.0.
- The EXL3 codec is adapted from [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)
  revision `d3739fd393337b1ff4d6c2a342b12f0c87a9592f` (MIT). The MIT grant is
  in `vllm_exl3/tc_moe/codec.cuh`. Copyright (c) 2025 Turboderp.
- The surrounding engine is [vLLM](https://github.com/vllm-project/vllm), Apache-2.0.
- Serving measurements used the checkpoint
  [Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw)
  revision `c5534b90602b4090980a1c3ded1eb3b4d99a38d5` (MIT). Those weights are
  not in this repository. On the layers that were measured, the trellis last
  dimension is 48, which is bits_x2 6 (K3), including layers 18 and 22.
- The model is DeepSeek V4.1 Flash from DeepSeek.

## Cooperative-decode experiment

The separate `experiments/exl3-cooperative-decode/` patch modifies ExLlamaV3
at `da68d57de61781e5b74ad68698ac4caf36408d7a`, Copyright (c) 2025 Turboderp,
under the [included MIT license](experiments/exl3-cooperative-decode/EXLLAMAV3-LICENSE.txt).
Its [scope and provenance notice](experiments/exl3-cooperative-decode/exllamav3-six-row-transpose-NOTICE.txt)
records the measured artifact and qualification limits. CurtisAccelerate performed the experiment and measurements. This component's
MIT notice is separate from the repository's Apache-2.0 license.

## October 22 experiments

The six-row host all-reduce patch adapts launch selection in Morrowmake's
host-staged collective at the existing pinned revision; its CUDA reduction
source is unchanged and remains Apache-2.0. Hybrid P2P and resident CUDA graph
pages report separate experiments, without shipping deployment controllers,
driver patches, model weights or binary modules. NVIDIA's documented graph
update API underlies the technique; no claim of inventing that API is made.
