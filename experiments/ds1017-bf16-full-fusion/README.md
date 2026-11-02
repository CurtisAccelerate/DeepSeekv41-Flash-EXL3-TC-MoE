# Speed-default Docker and configurable projection/quality settings

We extended our optimized DeepSeek serving stack with prefill fusion across all 40 target layers, faster target execution and integrated draft sampling, while preserving our specialized expert shapes, split attention and communication paths. We compared FP8 and BF16 output projection with BF16 KV, measuring decode speed, full request wall time and perplexity. The new Docker defaults to the speed option with short-row FP8 output projection and fixed-five verification. BF16 KV is retained in every lane.

Special thanks to [GNmac](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx) for the FP8 and draft-sampling work adapted in this update; see the [existing attribution document](NOTICE.md).

The [November 2 headline comparison](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-11-02/README.md) leads with the tentative speed winner (earlier FP8 projection, 143.06 prose tok/s), then the tentative quality winner in the latest fixture (original target, PPL 1.166442 versus 1.168604 for this option). The BF16-projection option measured 135.18 prose tok/s in its latest confirmation. The current Docker default selects the speed recipe. The original target remains the latest small-fixture numerical likelihood reference.

[Current full report](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-11-02/README.md) · [Speed-default recipe and options](defaults/DEFAULT-RECIPE.json) · [Supported private client](defaults/balanced_default_client.py) · [Attribution](NOTICE.md).

[Download the public Docker archive](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/releases/tag/2026.11.02-speed-k5). Its flattened filesystem excludes private build history and preserves every pinned inference/native/source file. Download instructions and checksums are attached to the release.

## Docker default and easy settings

The new default is `speed`; set `DS1017_LANE=bf16` at cold load for BF16 output projection, or select `quality` for the original target/reference prefill. All lanes retain BF16 KV and our specialized expert/attention/communication paths. `quality` is the latest small-fixture numerical reference, not a general accuracy guarantee.

```sh
docker run --rm --runtime=runc --network=none IMAGE describe
docker run --rm --runtime=runc --network=none -e DS1017_LANE=bf16 IMAGE describe
python defaults/configure_launch.py --spec /path/to/fresh-stage/launch.json --image sha256:IMAGE_ID
python defaults/check_ready.py --root /path/to/stage
python defaults/lane_client.py --root /path/to/stage --request request.json --out result.json
```

`describe` inspects settings without loading the model. The configure command defaults to speed and operates only on a fresh frozen stage; rerun its mounted-source preflight and retained all-rank GPU startup checks before requests. Select `--lane bf16` when preparing a fresh BF16-projection stage and when using its request client. The original `--lane quality` request path works in either qualified bank. Projection-bank changes require a cold load; a wrong-scope request fails instead of falling back. Model/data mounts and the existing private admission protocol are retained.

The entry point and client use existing runtime profiles. The new image build tests settings/profile compatibility without initializing CUDA. It does not imply a newly measured143tok/s result or a newly qualified speed-bank cold load. See [complete settings and validation scope](release/speed-default/README.md). The former BF16-only recipe/client and build records remain archived; they are not the new default.

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


## Archived BF16-only Docker packaging

Local versioned tag `star-ds1017:2026.11.02-balanced-k5`, immutable image `sha256:16d47d8d6059f8d4abfa159268bc9f3d65f9fc3b65311a481cd15991c572ee7c`. The derivative passed219-file source manifest verification,121 CPU tests plus1,164 draft-plan cases, native ABI/pin checks and a no-CUDA check. It packages the unchanged GPU-qualified runtime; a fresh GPU boot of the derivative is not claimed. It has not been pushed to a container registry. The inherited base must already be available locally.

Build a context outside this repository, then build on a host with that exact pinned base available:

```sh
python release/build_context.py --experiment . --out /tmp/ds1017-balanced-context
docker tag sha256:7ecbcf725f31d1b5bc05fc6863a16a94ef9c1c6cf7441e0a5c257fcd430e4d94 star-release-runtime:7ecbcf725f31
docker build --network=none --pull=false --build-arg BASE_IMAGE=star-release-runtime:7ecbcf725f31 -t star-ds1017:2026.11.02-balanced-k5 /tmp/ds1017-balanced-context
```

This is a private source/runtime artifact, not a complete model-weight image or a repointed public service. The build materializes existing pinned host-overlay code, without copying model data, and keeps two filesystem layers to fit the deeply layered retained base. Initial missing-path/layer-depth build failures are preserved in the internal evidence. See release/bf16-20261102/DOCKER-BUILD-VERIFIED.json and release/bf16-20261102/DEFAULT-SNAPSHOT.json.
