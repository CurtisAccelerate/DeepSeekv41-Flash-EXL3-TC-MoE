# Download and load the speed-default Docker image

The image defaults to short-row FP8 output projection with **BF16 KV**. It keeps our specialized expert shapes, split-attention and communication tuning, full 40-layer integer prefill fusion, Marlin prefill, and integrated fast drafting/sampling. Fixed-five verification and gate-off remain selected after local policy tests. Special thanks to GNmac for the selected recipe components adapted into this stack; component attribution and upstream licenses remain in NOTICE.md.

The release contains a real Docker image archive, split into 512 MiB parts to fit GitHub's per-asset limit. Python 3.11+ and Docker are required. Download `download-ds1017.py` from the release, then run:

```sh
python download-ds1017.py --load
docker run --rm --runtime=runc --network=none ds1017:2026.11.02-speed-k5 describe
docker run --rm --runtime=runc --network=none -e DS1017_LANE=bf16 ds1017:2026.11.02-speed-k5 describe
docker run --rm --runtime=runc --network=none -e DS1017_LANE=quality ds1017:2026.11.02-speed-k5 describe
```

The downloader verifies every part and the full compressed archive SHA256 **before** loading, verifies the loaded image ID, and applies the versioned local tag. Valid existing parts are reused after checksum checks. `--join` optionally writes a single archive; `--load` streams the numbered parts without creating that extra copy. Set `--directory` to choose download storage. Keep space for the compressed parts and the approximately 22 GB Docker image.

`describe` checks settings without loading a model. Actual serving still requires the matching EXL3/official checkpoint selection, Engram backing, model/data mounts, tensor-parallel four on compatible hardware, and the retained qualified launch/preflight workflow. No weights are included. See the [settings guide](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE/blob/main/experiments/ds1017-bf16-full-fusion/release/speed-default/README.md).

| Setting | Target execution | KV cache | Intended use |
| --- | --- | --- | --- |
| `speed` (default) | Integrated short-row FP8 projection/dense, hybrid experts | BF16 | Faster recipe family |
| `bf16` | Integrated BF16 projection, retained fast dense/hybrid experts | BF16 | Precision-conservative option |
| `quality` | Original target and reference prefill | BF16 | Latest synthetic-fixture numerical likelihood leader |

Changing speed/BF16 projection capture banks requires a cold load; the original quality reference is available inside either qualified bank. An unavailable lane fails rather than silently falling back. The indexer remains quantized; these are not all-BF16 model weights. Request temperature/seed remain unchanged; single-user temperature-one greedy proposals remain opt-in.

Recorded one-user prose: **143.06 tok/s** for the earlier fast integration, **135.18** for the later BF16-projection confirmation, versus a historical GNmac **141.55**. The measurements are from separate boots/configurations, not a fresh matched victory. Latest matched likelihood: original PPL **1.166442**, BF16-projection option **1.168604**, with the difference below repeat variation. These are not success rates for agent tasks. The Docker package was not freshly GPU-booted or rebenchmarked.

The public filesystem was scanned and flattened to discard inherited private build layers/history. All 327 protected runtime/native/published-source pins were retained. Source/ABI/CPU checks and 25 settings guards passed without CUDA initialization; all three entry-point settings were inspected. A fresh installation must pass the existing all-rank startup GPU gates. Do not treat CPU packaging checks as GPU qualification on a different machine.

The archive is a GitHub Release download, not a container-registry `docker pull` reference. No existing service or Hermes endpoint is changed by downloading/loading it.
