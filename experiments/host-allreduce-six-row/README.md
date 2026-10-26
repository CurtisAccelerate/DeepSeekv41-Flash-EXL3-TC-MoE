# Six-row host all-reduce tuning

**About 2% faster controlled single-stream counting decode**, on top of the
EXL transpose experiment. No weights, quantization or CUDA reduction source
change. This is a launch-policy optimization of Morrowmake's host all-reduce.

For TP4 BF16 `(6, 5120)` activations (61,440 bytes), the measured change was
**one block / 1,024 threads / one-shot → one block / 512 threads / two-shot**.
Two-shot reduces host-memory traffic but adds a barrier; the result is specific
to the tested PCIe Gen2 ×16 dual-socket Broadwell system with 4× CMP 170HX.
One block and fixed-rank FP32 reduction are retained. A four-block candidate
was rejected because mixed sizes could desynchronize sequence/parity counters.

| Controlled test | Previous decode | Tuned decode | Paired gain |
|---|---:|---:|---:|
| Eight alternating pairs, fixed 400-token counting | 184.21 tok/s | 188.13 tok/s | +2.21% geometric mean |
| Later four-round sweep, same counting task family | 183.42 tok/s | 187.05 tok/s | +1.74% median |

Every pair in these two rows matched prompt/output token IDs, output lengths,
cache counts and speculative-work summaries and passed the recorded quietness
criterion. Summary parity does not prove identical per-step ordering. These
results support a small workload-specific gain, not a general prose or C8 gain.
The earlier transpose change is separate; do not add percentages from separate
campaigns. [Full measurements](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-10-22/README.md).

## Source and use

[host-allreduce-six-row.patch](host-allreduce-six-row.patch) applies to
`vllm/sm80/distributed/host_shm_all_reduce.py` in [Morrowmake/vllm-cmp170hx at `18305948`](https://github.com/Morrowmake/vllm-cmp170hx/blob/18305948acba3f4c4d912cdd56f2202d2ebd94e8/vllm/sm80/distributed/host_shm_all_reduce.py).
The pinned source matches the measured baseline after final-newline normalization.
The embedded `CUDA_SRC` remains identical; identities are recorded in
[source-identities.json](source-identities.json).

This publication packages the measured selection as an **opt-in source patch**
with narrower explicit shape/SM80/TP4 guards. It is not the exact resident
controller used for the measurements. Patch application, Python syntax and
unchanged CUDA source are checked; this packaging has not received an additional
GPU run. Existing launch scripts and defaults are unchanged.

Apply in a separate checkout of that pin, then set
`VLLM_HOST_ALLREDUCE_SIX_ROW=1` on every worker **before initial capture**.
Use an absolute path to the patch:

```bash
git apply --check /path/to/host-allreduce-six-row.patch
git apply /path/to/host-allreduce-six-row.patch
```

Unmatched shapes keep upstream dispatch. Setting an environment variable after
capture does not modify a captured graph. Our separate
[resident CUDA graph technique](../../docs/resident-cuda-graph-optimization.md)
made A/B switching possible on an already prepared owner. Mixed-size collective
correctness, graph replay, numerical checks and a paired serving benchmark on
the target topology remain necessary before adopting the source adaptation.

## Attribution

Original host-staged all-reduce: Morrowmake and vLLM contributors, Apache-2.0.
Six-row launch-policy implementation, experiment and measurements:
CurtisAccelerate, with AI assistance.
The repository [Apache-2.0 license](../../LICENSE) and [NOTICE](../../NOTICE.md)
apply. The separate EXL transpose component retains its MIT notice.
