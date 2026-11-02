# Speed-default Docker with explicit BF16 and quality settings

The default is **speed**: BF16 KV, short-row FP8 output projection/dense paths,
our retained specialized expert/attention/communication paths, full 40-layer
integer prefill fusion, Marlin prefill, fast draft head/sampler, fixed-five and
gateoff. This is the faster recipe family; the earlier repeated prose result
was 143.06 tok/s. Packaging does not constitute a new speed measurement.

| Setting | Output projection | Target profile | Prefill | Purpose |
| --- | --- | --- | --- | --- |
| `speed` (default) | Short-row FP8 | Fast integrated target | Full integer fusion | Speed choice |
| `bf16` | BF16 | Integrated target with BF16 projection | Full integer fusion | Precision-conservative option |
| `quality` | BF16 | Original target | Original reference | Latest small-fixture numerical quality leader |

All three retain **BF16 KV**, including the speed default. The indexer retains
its original quantized storage; these are not fully BF16 model weights.
The latest likelihood fixture favored the original target 1.166442 PPL versus
the BF16-projection option 1.168604, within repeat variation. BF16 projection
is an option, not a proved universal quality improvement.

## Select a boot lane

The Docker entry point maps `DS1017_LANE` to the existing runtime capture
scope. It does not add or replace an inference kernel.

```sh
docker run --rm --runtime=runc --network=none IMAGE describe
docker run --rm --runtime=runc --network=none -e DS1017_LANE=bf16 IMAGE describe
docker run --rm --runtime=runc --network=none -e DS1017_LANE=quality IMAGE describe
```

These `describe` commands inspect settings and do not load a model. Model
weights, the existing Engram backing, the frozen mounts and the existing
preflight/server command remain required for a real launch.

After preparing a **fresh** dedicated BF16 launch spec with the established
stage workflow, configure it before its mounted-source preflight:

```sh
python defaults/configure_launch.py --spec /path/to/fresh-stage/launch.json --image sha256:IMAGE_ID
python defaults/configure_launch.py --spec /path/to/another-fresh-stage/launch.json --image sha256:IMAGE_ID --lane bf16
```

The first command defaults to speed. The helper backs up the original spec,
preserves model/mounts/preflight/server arguments, sets the real capture scope,
and refuses a stage that has already started. Run the existing mounted-source
preflight, then use the lane-aware readiness check before submitting requests:

```sh
python defaults/check_ready.py --root /path/to/stage
python defaults/check_ready.py --root /path/to/bf16-stage --lane bf16
```

The retained GPU
gates must qualify every rank, requested graph shape, native row, sampler
ownership check and prefill component. Keep 140 W, the 1152 MiB capture cap,
thermal protections and the 1 GiB startup free-memory floor.

Changing **speed ↔ BF16 projection requires a cold load**, because their target
graph banks differ. The quality-reference target is available inside either
qualified bank. No simultaneous second model or larger capture pool is needed.

## Select a qualified request lane

```sh
# Speed is the default; requires the qualified speed boot.
python defaults/lane_client.py --root /path/to/stage --request request.json --out result.json

# Requires the qualified BF16-projection boot.
python defaults/lane_client.py --lane bf16 --root /path/to/stage --request request.json --out result.json

# Original quality reference, available in either bank.
python defaults/lane_client.py --lane quality --root /path/to/stage --request request.json --out result.json
```

The JSON is one streaming API body or a list of up to eight bodies. The client
checks all four live identities against durable startup receipts, refuses an
unavailable/wrong-scope lane, selects the requested target/prefill transaction,
records wall/TTFT/SSE gaps, then restores the private paused reference state.
It never silently falls back from speed to BF16. Request temperature/seed are
preserved. Greedy proposals remain explicit `--greedy-proposals` opt-in.

This release uses the existing private admission protocol. It does not repoint
Hermes or enable public serving. The earlier BF16-only client/recipe and image
remain preserved as historical artifacts.

## Validation scope

The image build checks unchanged inference/native source pins, existing image
CPU/ABI checks and actual profile/scope compatibility, plus 25 lane/entry-point/
request guards. CUDA is not initialized during the build. A new speed-bank cold
GPU load and new speed/quality measurement are not claimed by this build.

Attribution remains in the existing `NOTICE.md`; selected GNmac components and
the Morrowmake/ExLlamaV3/vLLM foundations retain their source/license boundaries.
