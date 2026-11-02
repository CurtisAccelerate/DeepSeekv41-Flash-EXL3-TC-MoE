# Published BF16-projection option and reader choice

We extended our optimized DeepSeek serving stack with prefill fusion across all 40 target layers, faster target execution and integrated draft sampling, while preserving our specialized expert shapes, split attention and communication paths. We compared FP8 and BF16 output projection with BF16 KV, measuring decode speed, full request wall time and perplexity. The packaged option uses BF16 output projection and fixed-five verification.

Special thanks to [GNmac](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) for the FP8 and draft-sampling work adapted in this update; see the [existing attribution document](NOTICE.md).

The [November 2 headline comparison](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-11-02/README.md) leads with the tentative speed winner (earlier FP8 projection, 143.06 prose tok/s), then the tentative quality winner in the latest fixture (original target, PPL 1.166442 versus 1.168604 for this option). The packaged option measured 135.18 prose tok/s in its latest confirmation. Users choose their tradeoff; archived default/snapshot names identify this packaged option, not proof of an overall winner.

[Current full report](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-11-02/README.md) · [Packaged BF16 recipe](defaults/DEFAULT-RECIPE.json) · [Supported private client](defaults/balanced_default_client.py) · [Attribution](NOTICE.md).

Use the client only against an already-qualified dedicated owner:

```sh
python defaults/balanced_default_client.py --root /path/to/qualified-stage --request request.json --out response.json
```

The request file is one streaming API body or a list of one through eight bodies. The client selects the recipe, records actual wall/TTFT/gaps and restores the reference/private paused state on exit. `--greedy-proposals` is explicit opt-in with the mixed speed and seeded-repeat limitations in the report. This is not a general public serving launcher or Hermes promotion.

The original target is the current small-fixture numerical accuracy reference. The packaged BF16 recipe has tangible wall-time gains over the original and retains precision safeguards; general accuracy gain/equivalence is not established. It is not the demonstrated quality winner or a universal recommendation. Narrow/adaptive/gated source is retained for reproducibility, not selected or silently admitted for sampled workloads.

Source updates are in place, with all inference/native bytes pinned to the measured image. Privacy-only offline-helper changes have both tested/published hashes in SOURCE-MANIFEST.json. Run `python check_sources.py`.

## Historical November1 integration and qualification

The following dated material describes the prior stage, not current default or latest quality ranking.

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


## Verified Docker packaging

Local versioned tag `star-ds1017:2026.11.02-balanced-k5`, immutable image `sha256:16d47d8d6059f8d4abfa159268bc9f3d65f9fc3b65311a481cd15991c572ee7c`. The derivative passed219-file source manifest verification,121 CPU tests plus1,164 draft-plan cases, native ABI/pin checks and a no-CUDA check. It packages the unchanged GPU-qualified runtime; a fresh GPU boot of the derivative is not claimed. It has not been pushed to a container registry. The inherited base must already be available locally.

Build a context outside this repository, then build on a host with that exact pinned base available:

```sh
python release/build_context.py --experiment . --out /tmp/ds1017-balanced-context
docker tag sha256:7ecbcf725f31d1b5bc05fc6863a16a94ef9c1c6cf7441e0a5c257fcd430e4d94 star-release-runtime:7ecbcf725f31
docker build --network=none --pull=false --build-arg BASE_IMAGE=star-release-runtime:7ecbcf725f31 -t star-ds1017:2026.11.02-balanced-k5 /tmp/ds1017-balanced-context
```

This is a private source/runtime artifact, not a complete model-weight image or a repointed public service. The build materializes existing pinned host-overlay code, without copying model data, and keeps two filesystem layers to fit the deeply layered retained base. Initial missing-path/layer-depth build failures are preserved in the internal evidence. See release/DOCKER-BUILD-VERIFIED.json and DEFAULT-SNAPSHOT.json.
