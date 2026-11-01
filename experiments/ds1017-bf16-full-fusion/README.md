# DS1017: BF16 cache, qualified A/B/C decode and full prefill fusion

**Measured experimental source export, November 1, 2026. Defaults remain off.**
This directory contains the tested runtime, native source and qualification
tests from the V6 build, with site-specific launch records, historical raw logs
and stale staging documents removed. It does not change the repository's normal
launch or provide a portable replacement base image.

The retained optimized stack is the base. Selected GNmac target/drafter kernels
were integrated into its guarded A/B/C graph bank; the new prefill path then
combines integer index preparation and ragged packing for **all 40 target layers**.
It keeps the existing floating-point attention kernel and Marlin dense prefill.
The validated cache type is **bfloat16**, not FP16 and not FP8.

## Measured result

The matched C-fixed, BF16-cache prefill comparison improved wall time by
**4.3–6.7% beyond our MoE-only prefill baseline** over 2K–128K. At 128K:
**91.585 → 87.615 seconds**. These are full request walls, not kernel timers.
The previous partial fusion covered 20 ratio-1 layers. This path also covers
18 ratio-2 and 2 sliding-window-only layers. It directly builds the same int32
indices and row pointers, preserving invalid slots, order and bounds.

The same-boot A/C SPARK suite completed 1,680 requests, three passes per arm.
At one/eight users C reached 112.02/281.48 full-wall tokens/s versus A's
88.66/245.44. C's absolute rates are close to previous C results; the larger
relative A/C gap partly reflects a slower A reference. Do not attribute that
whole gap to this prefill kernel. The model-recipe repository contains the
complete tables, previous baselines, likelihood results and caveats.

## What the labels mean

| Label | Meaning |
|---|---|
| A | Original optimized target/decode paths |
| B | Selected GN-derived target paths, retaining our eligible expert/attention routes |
| C | B target plus FP8 draft-head and fused Markov selection, with proposal/RNG guards |
| REFERENCE | Original prefill |
| MOE_TRY | Qualified expert-prefill schedule, the isolated prefill baseline |
| MOE_COMBINE | Prior partial integer fusion, 20 ratio-1 layers |
| MOE_RAGGED | Full integer fusion plus MoE prefill; Marlin dense prefill retained |

All tested comparisons here use BF16 KV. That does not make the expert weights
or all target operations BF16. C retains the qualified short-row FP8 target
paths, while the target vocabulary head remains BF16.

The common speed/accuracy recommendation is BF16 KV + `MOE_RAGGED` + Marlin
dense prefill. There is no demonstrated decode arm that wins every quality
and speed measure: A leads long-history/critical likelihood; B effectively
ties A overall; C is the fastest measured A/C option and leads short likelihood.
Optional dense/cuBLAS variants passed startup numerical gates but lack a
matched current end-to-end speed/quality comparison. The prior dense experiment
worsened likelihood; it is not promoted as a speed-only override.

Start with the [review map](REVIEW-MAP.md) to separate the new integration from
retained vLLM/A/B/C dependencies. `CHANGE-MAP.json` records that classification.

## Source and build boundaries

`source/package/` preserves inference runtime/native source byte for byte. Six
offline helpers have privacy-only endpoint/socket substitutions, with original
and published hashes recorded in `SOURCE-MANIFEST.json`. See
[publication privacy cleanup](PUBLICATION-PRIVACY.md). Check the published bytes with:

```bash
python experiments/ds1017-bf16-full-fusion/check_sources.py
python -m unittest discover -s experiments/ds1017-bf16-full-fusion/source/package/prefill-ragged-tests
```

The Dockerfile is for the **retained exact local base image**, whose digest is
in [QUALIFICATION.md](QUALIFICATION.md). Its base verifier rejects other bytes.
No public download of that site-local image is implied. A build compiles only
the new MoE prefill operator and runs the CPU qualification; it does not load a
model, stop a serve or prove end-to-end GPU qualification on another machine.
Use a locally verified alias for the pinned base:

```bash
docker build --network=none --build-arg BASE_IMAGE=YOUR_VERIFIED_LOCAL_ALIAS \
  -t ds1017-prefill-candidate experiments/ds1017-bf16-full-fusion/source/package
```

Site-specific launch JSON is intentionally excluded. Deployment still needs
the retained model/Engram mounts, ABI, graph-bank integration and source pins.
Reusing a Python file without rebuilding/recapturing does not update CUDA graphs.
Neither this documentation nor the source export enables public admission.

The published package was rechecked in the exact V6 image with CUDA hidden,
the NVIDIA runtime disabled and networking disabled: **96 CPU tests, no skips,
12 native host ABI queries, no CUDA initialization**. Historical GPU and full
engine receipts are summarized in [QUALIFICATION.md](QUALIFICATION.md).

## Drafter scope and remaining latency work

C exercised the GN-derived FP8 draft head and fused Markov selection. The main
run retained **five draft tokens, gate off**. It did not test GNmac's full
adaptive 3/5 policy or 0.10 gate. Only A/B/C and their oracle controls were
published by this loaded bank; optional source code is not evidence of a
qualified runtime arm. See the concrete [next ablation](DRAFTING-ABLATION.md).

Cold latency exceptions are preserved. Eight-user count took C 7.250s versus
A 5.166s, and the first seeded short delivery probe paused for 0.399s. Verbose
logs correlate those windows with shared engine/draft and rejection/recovery
compilation. This does not quantify the whole delay or prove Hermes buffering.
Broader pre-admission engine warmup is a supported next direction, not a fix
implemented or qualified by this export.

Read [NOTICE.md](NOTICE.md) for the separate GNmac, Morrowmake, vLLM, ExLlamaV3
and CurtisAccelerate contribution boundaries. No model weights or compiled
binary modules are committed.
