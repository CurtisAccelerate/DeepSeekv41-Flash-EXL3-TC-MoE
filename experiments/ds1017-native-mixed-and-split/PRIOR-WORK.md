# Attention prior work and contribution boundaries

The October 25 DS1017 optimization is CurtisAccelerate's hardware-specific
adaptation, tuning and serving integration of existing sparse-attention
primitives, combined with a captured vocabulary-sharded drafter and native
expert specialization. The upstream primitives are substantial contributions
to this result and are credited below. Performance and quality numbers in the
1017 report are CurtisAccelerate's measurements on 4x CMP 170HX.

## Direct attention source lineage

| Contribution used | Prior work and attribution |
|---|---|
| DeepSeek sparse attention in Triton | Hexiang Wang and the vLLM ROCm contributors, [flash sparse MLA implementation, PR #41812](https://github.com/vllm-project/vllm/commit/7863fff6e591161a03a7df68b0879b9fa61cfc0b). |
| Split-K/flash-decode partial attention and stable combination of partial softmax state | Fangzhou Ai and vLLM contributors, [split-K decode, PR #44899](https://github.com/vllm-project/vllm/commit/fcf5115c45b9acfe3a77052ddbb7dfb0f4d5ef18). These are the central primitives our attention wrapper calls. |
| Subsequent upstream split-path work | Tuukka Sarvi's [gfx942 split-decode enablement, PR #46275](https://github.com/vllm-project/vllm/commit/b8168e33e000df5c60d63d7ca244ba2ca2c34079), and Fangzhou Ai's [reduction-occupancy change, PR #48788](https://github.com/vllm-project/vllm/commit/efed8a1e8345de2b4398a48243301d6a5b1a361e). These are part of the retained implementation's lineage; the AMD-specific dispatch policies are not our measured SM80 policy. |
| Ampere/SM80 compatibility and software FP8 conversion | haosdent, with co-author thomaslwang and the contributors acknowledged in [PixelML/sm80vllm commit f8ea5bb](https://github.com/PixelML/sm80vllm/commit/f8ea5bb163c161ef38b401d055cc5fd4a934091a). This adapts the ROCm Triton implementation to SM8x and supplies the FP8/BF16 conversion helpers and lookup table used here. |

The wrapper directly calls `_sparse_attn_decode_partial_kernel` and
`_sparse_attn_decode_reduce_kernel` from the retained
`rocm_aiter_mla_sparse.py`, together with the helpers in `fp8_sm80.py`.
The split-K algorithm and those kernel bodies are upstream work. Their
Apache-2.0 SPDX/copyright notices and the [license](retained/LICENSE) are
preserved. The retained file also credits DeepSeek's
[DeepGEMM attention test helpers](https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py)
for `fp8_paged_mqa_logits_torch` and `fp8_mqa_logits_torch`; those comments
remain intact. Those reference helpers are separate from our split decode launch.

## CurtisAccelerate's contribution

- Identified and measured the exposed compressed-attention decode work in
  the DS1017 EXL3 serving configuration, then adapted dispatch to use the
  existing split primitives for the eligible SM80 target graph.
- Implemented the DS1017-specific layer, cache-layout, shape and dtype guards;
  preallocated workspaces; startup preparation; and capture/fallback integration.
  The measured path covers compressed main-model layers 2-39 at six target
  rows / one request, with the other attention shapes retaining their reference path.
- Swept and selected the **8 splits / 8 core warps / 4 reducer warps / output
  copy** launch geometry for this configuration. The reduction primitive is
  upstream; selecting, integrating and measuring this configuration is our work.
- Implemented the captured vocabulary-sharded K5 drafter path, including
  local projection and maximum-value/token exchange, loaded-weight parity
  checks, output ownership and fallback handling. This builds on the existing
  DSpark model and serving APIs; target sampling and verification stay unchanged.
- Integrated these components with the guarded native-mixed expert kernels,
  prepared selectable resident configurations for comparisons without another
  model reload, and performed numerical, replay, performance and matched
  tool-using task evaluations. The public extraction includes components,
  fixtures and integration guidance; the server-specific coordinator is not included.

These are implementation and validation contributions beyond choosing a flag.
They do not establish a new attention algorithm or universal quality equivalence.
The experimental attention path changes floating-point reduction order; its
numerical and task-quality limitations remain in the performance report.

## Reproducible source identification

The measured dependency snapshots are retained in this directory:

| File | SHA-256 |
|---|---|
| `retained/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | `a36c0e13cf63a77c804a5d5d60ec2f6d3e62bec51de3afac4f28619f91f5f8fc` |
| `retained/vllm/v1/attention/ops/fp8_sm80.py` | `4ad3bd77073051c0cb2db2ef3d307ff5047e9a29dab3714fcc65843e87a7eb6e` |

An attribution audit compared these snapshots with PixelML commit `f8ea5bb`:
the partial-attention function and all eight FP8 helper functions have identical
Python ASTs. The retained reducer adds adaptive-split branches; our launch fixes
`ADAPTIVE_SPLITS=False`, retaining the upstream nonadaptive reduction path.
The complete files include integration differences and are **not** described as
byte-identical copies of that commit. Use the recorded snapshot hashes for
compatibility, rather than assuming a current upstream checkout is interchangeable.

## DeepSeek build, expert kernels and loading: separate contributions

**CurtisAccelerate implemented and published the DeepSeek V4.1 Flash EXL3 MoE
port and serving build, followed by the 1017 weight recipe and subsequent
optimizations.** The original six-row transpose changes, native-mixed K2/K3
specialization, Engram retention integration and resident optimization workflow
are CurtisAccelerate's work on the respective upstream foundations. They should
not be attributed wholesale to an upstream engine or kernel author.

| Component | Upstream foundation | CurtisAccelerate's implementation |
|---|---|---|
| Earlier tensor-core/prefill MoE path in [`vllm_exl3/tc_moe`](../../vllm_exl3/tc_moe/) | Morrowmake's [`moe_v2.cu` and `moe_prefill.cu`](https://github.com/Morrowmake/vllm-cmp170hx/tree/18305948acba3f4c4d912cdd56f2202d2ebd94e8/csrc/exl3), originally specialized to hidden 2560, intermediate 640 and bits_x2 10; EXL3 codec from Turboderp. | DeepSeek port to hidden 5120, intermediate 512/640 and K2/K3; register-spill fixes; row dispatch; shared workspace; EXL3 hook and validation. The DeepSeek port is ours, while the retained tensor-core kernel foundations remain credited to Morrowmake. |
| [Six-row cooperative transpose](../exl3-cooperative-decode/) and [native-mixed expert specialization](native/instantiate.cu) | Turboderp's ExLlamaV3 cooperative expert kernels at `da68d57de61781e5b74ad68698ac4caf36408d7a`. | Our six-row transpose/layout changes and later guarded K2 / specialized K3 implementation, shape guards, integration and bit-exact fixture validation. This is a separate lineage from Morrowmake's earlier tensor-core/prefill operators. |
| [Six-row host all-reduce tuning](../host-allreduce-six-row/) | Morrowmake's host-staged collective. | Our six-row launch policy, shape guard and paired measurements; the underlying CUDA reduction source is unchanged. |
| [Resident CUDA-graph optimization](../../docs/resident-cuda-graph-optimization.md) | NVIDIA's graph-update APIs and the serving engine's graphs. | Our retained graph/node ownership, distributed switching/rollback integration and no-reload A/B workflow. |

**Engram/n-gram retention is also our loading integration.** We implemented
persistent shared host backing and worker reattachment so compatible restarts
can reuse the already staged Engram payload instead of rereading/rebuilding it.
The four-rank full-payload attachment proof verified bit-exact native gathers,
unchanged payload hashes and backing retained after child exit. This is separate
from the EXL3 on-disk arena cache and from retaining GPU weights during same-worker
graph updates. It requires the correct Engram payload and surviving backing;
it is not a claim of GPU-weight retention across worker replacement or reboot.
The deployment-specific retention/controller implementation is not bundled as
a standalone installer in this publication.

DeepSeek supplies the model, Engram tables and DSpark architecture; LibertAI
supplies the measured routed-expert quantization. See [NOTICE.md](NOTICE.md)
and the [repository notice](../../NOTICE.md) for licenses and revision pins.
