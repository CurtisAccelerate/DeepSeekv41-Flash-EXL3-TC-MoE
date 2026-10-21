# ExLlamaV3 cooperative-decode experiment

This directory is the canonical home for the six-row transpose source
experiment measured on 4× CMP 170HX on October 21, 2026. The 1017 repository
keeps the model recipe and serving benchmarks. This component is separate
from this repository's `vllm_exl3/tc_moe` implementation and is not enabled
by its launch scripts or hook.

## Change and measured scope

The [measured patch](exllamav3-six-row-transpose.patch) transposes MMA
operands: decoded weights occupy M16 and tokens N8. This uses one instead
of two warp MMA instructions per decoded 16×16 tile. Weight bytes, EXL3
decoder, FOLD4 cadence, reductions and epilogues are retained. The change
does not halve total expert latency or total serving work.

The measured serving guard required SM80, six rows, hidden 5120, top-k 6,
384 local experts in [0,384), uniform gate/up/down K2 or K3, mul1 (cb2),
intermediate 512 or 640, gated SiLU with limit 10, no biases/shared extras
or mixed codebooks, and matching contiguous CUDA tensor types, scratch
and counters. The binary used wide gate/down entries with 512 threads,
WNT2/WN4/WKK4, 128 columns per block, Ksplit1 and an unchanged rot entry.
Other calls fell back to the original extension. Concurrent candidate
coverage was not instrumented.

**The header template has broader reach than that guard.** This is a source
experiment, not a generic replacement or a ready-to-install integration.
The private resident graph controller, CUDA binaries and weight fixtures
are not included. See the [scope and attribution notice](exllamav3-six-row-transpose-NOTICE.txt).

## Validation and results

The existing GPU evidence comprises 18 fixtures on each of four cards:
72/72 primary bitwise comparisons passed, with independent FP32 sanity
gates on real cases. Tested expert latency fell about 4.8–6.7%. These
fixtures do not establish full-model bitwise equivalence or task quality.
The controlled serving baseline was rebuilt with CUDA 12.8 and is not
proven identical to the historical CUDA 13 stock binary.

The model-specific [two sweep tables](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/e560d05caa54969b5fc664056202bea863ca9513/benchmarks/2026-10-21/kernel-sweeps.md)
record the measured decode improvement and its limits. The separate
[four-Spark comparison](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/e560d05caa54969b5fc664056202bea863ca9513/benchmarks/2026-10-21/spark-comparison.md)
is a comparison of different systems and quants, not the effect of this
patch. Benchmark tables remain maintained in the 1017 repository; these
links pin the evidence accompanying this version of the experiment.

## Reproduce the CPU and source checks

From the root of this repository, with Python and NumPy available:

```bash
python3 experiments/exl3-cooperative-decode/verify_mapping_cpu.py
```

This checks K2/K3 decoded fragment coordinates, a single writer per shared
reduction cell for 1–8 rows, and the logical FOLD4 rounding/folding model.
It imports no torch or CUDA and requires no model. It does not emulate
hardware MMA internal accumulation or prove GPU equivalence; the broader
CPU coordinate coverage does not expand the qualified GPU dispatch scope.

For source compatibility, use a separate checkout at the exact upstream pin:

```bash
git clone https://github.com/turboderp-org/exllamav3.git exllamav3-review
git -C exllamav3-review checkout --detach da68d57de61781e5b74ad68698ac4caf36408d7a
git -C exllamav3-review apply --check ../experiments/exl3-cooperative-decode/exllamav3-six-row-transpose.patch
```

The measured baseline differs from that upstream header only by one final
blank line. The patch is LF-normalized; measured header hashes preserve
the original bytes. [Source identities](source-identities.json) retain
the header, cubin and published patch hashes. A successful apply check
establishes source compatibility, not CUDA compilation or safe dispatch.

## Before broader integration

Add an explicit guard for the demonstrated geometry and retain fallback
outside it. Extend upstream `tests/kernels/moe/test_moe_coop.py` with
eligible real and synthetic K2/K3 cases, zero/foreign/duplicate routes,
partial rows, graph replay and fallback checks. Rebuild both arms with
the same toolkit and flags, rerun numerical and performance gates, and
measure actual candidate dispatch coverage. Wider architecture, HALF,
codebook or row support requires corresponding validation.

The publication and repository move perform no new GPU validation or
deployment. Loading arbitrary new binaries into a live serving process
is outside this source artifact's qualification.

## Attribution

ExLlamaV3 cooperative source: Turboderp, Copyright (c) 2025,
[MIT](EXLLAMAV3-LICENSE.txt). Experiment and measurements: Curtis White /
CurtisAccelerate. Serving integration: vLLM and Morrowmake. Model/quant:
DeepSeek and LibertAI. The included MIT license governs the inherited
component; the repository's other tensor-core work retains its own notices.
