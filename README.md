# DeepSeek V4.1 Flash — tensor-core EXL3 MoE on CMP 170HX

[CurtisAccelerate](https://github.com/CurtisAccelerate).

Serving recipe for DeepSeek V4.1 Flash, EXL3, tensor parallel 4, on 4× CMP
170HX (sm_80, 140 W, HBM locked at 1728 MHz) in a 2× Xeon E5-2689 v4
Broadwell host, PCIe Gen2 x16, no NVLink. Fastest build measured for this
mode: these weights, four of these cards, this host.

## DS1017 update — October 25, 2026

[Native mixed experts and experimental split attention / captured drafter](experiments/ds1017-native-mixed-and-split/)
now have canonical source, explicit default-off component flags, startup
integration guidance, retained dependencies, generated fixtures and CPU checks.
Native mixed adds **+4.10% controlled counting** over the already published
transpose + six-row baseline. The combined C1 path uses split8/core8 attention
and a captured vocabulary-sharded K5 drafter on top of native mixed.

Experimental, opt-in: initial numerical/runtime checks passed and the path was evaluated in
limited matched end-to-end tests; task-quality results were mixed. Split attention changes floating-point reduction and
is not bit-exact; broader numerical accuracy and task-quality verification is
still warranted. The bundle requires serving-specific startup/capture
integration and does not enable the historical launch scripts by itself.
See the [1017 performance and quality report](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/tree/main/benchmarks/2026-10-25).

## Benchmarks

Shipped stack. Coop through 42 rows, tensor-core MoE from 43 rows through
prefill, host all-reduce, peer access off. Decode is
`(completion_tokens - 1) / (end - first_token)`. Temperature 0, thinking off.
Medians use all five runs. Needles passed 9/9 through 256K. The KV pool is
616,916 tokens.

| | This build |
|---|---:|
| Structured decode | 178.3 tok/s |
| Structured, 8 users | 587 tok/s |
| Prose decode | 91.2 tok/s |
| Prose, 8 users | 260 tok/s |
| Python, 99-token median | 129.5 tok/s |
| Median time to first token, 2K through 255K | 8.37 s |
| Time to first token at 255,424 tokens | 10.9 s |

Mia cells on the same server, median of 5. The power cap was raised only for
these rows and put back to 140 W when the run finished. Decode does not move
with the cap.

| W | structured, 1 user | structured, 8 users | code, 1 user | code, 8 users | prose, 1 user | prose, 8 users |
|---:|---:|---:|---:|---:|---:|---:|
| 140 | 178.6 | 564.7 | 138.7 | 308.1 | 84.9 | 249.7 |
| 165 | 178.6 | 574.5 | 136.0 | 306.9 | 94.2 | 256.8 |
| 180 | 178.6 | 574.6 | 136.2 | 313.1 | 86.3 | 264.8 |

Cold prefill and decode on `/v1/completions`, 128 generated tokens, median of
3. Prefill is prompt tokens / time to first token.

| W | 2,048 prefill / decode | 8,192 | 32,768 | 65,536 | 131,072 |
|---:|---:|---:|---:|---:|---:|
| 140 | 1,570 / 149.2 | 1,528 / 146.8 | 1,519 / 148.5 | 1,453 / 149.9 | 1,329 / 156.5 |
| 165 | 1,595 / 149.3 | 1,560 / 147.1 | 1,542 / 148.3 | 1,485 / 150.0 | 1,358 / 154.8 |
| 180 | 1,577 / 149.5 | 1,525 / 147.3 | 1,546 / 148.0 | 1,488 / 149.9 | 1,367 / 154.3 |

Same loaded server, model not restarted. The HBM lock was dropped from NDIV 64
(1728 MHz) to NDIV 54 (1458 MHz) and put back. `nvidia-smi` printed 1458 MHz
on every sample at both locks. That print is not the lock.

| | 1728 MHz | 1458 MHz |
|---|---:|---:|
| Structured, 1 user | 178.5 | 174.2 |
| Structured, 8 users | 575.7 | 568.2 |
| Code, 1 user | 138.6 | 139.1 |
| Prose, 1 user | 90.2 | 93.1 |
| 2,048 prefill / decode | 1,556 / 149.4 | 1,561 / 145.8 |
| 32,768 prefill / decode | 1,532 / 148.5 | 1,512 / 144.6 |

A later 165 W pass left the lock at NDIV 64. The lock tool read 1728 MHz for
the whole run, and the cap went back to 140 W at the end. Structured 178.7 /
572.5, code 136.4 / 315.5, prose 88.2 / 257.1. Cold prefill / decode was
1,596 / 149.5 at 2,048, 1,562 / 146.9 at 8,192, 1,541 / 148.7 at 32,768, and
1,483 / 150.7 at 65,536.

## Comparisons

Same checkpoint, same four cards, same host. The right column is this build.
The left column is the previous stack on this server: full CUDA graphs, coop,
and NCCL.

| | Graphs + coop | This build | |
|---|---:|---:|---:|
| Structured tok/s | 172.65 | 178.27 | +3.3% |
| Structured cycle | 34.49 ms | 33.41 ms | |
| Structured acceptance | 5.970 | 5.970 | same text, hash `4e4e24a20477` |
| Python, 99-token median | 125.11 | 129.54 | +3.5%, hash `4a9988321ab7` |
| Prose tok/s | 86.13 | 91.17 | +5.9% |
| Structured c1 / c4 / c8 | 173.2 / 313.8 / 456.4 | 178.8 / 343.9 / 586.6 | c8 +28.5% |
| Prose c1 / c4 / c8 | 82.1 / 162.5 / 193.3 | 93.3 / 165.8 / 260.2 | c8 +34.6% |
| Growing c1 TTFT, median | 10.15 s | 8.37 s | 17.5% lower |
| TTFT at 2,080 tokens | 2.09 s | 1.78 s | |
| TTFT at 255,424 tokens | 12.63 s | 10.90 s | |
| KV pool | 672,372 | 616,916 | still 2.35× at 262,144 |

Every one of the 30 growing-context turns was faster. No preemptions.
Needles passed at 32,768, 131,072, and 261,568 tokens, depths 10 / 50 / 90
(9/9). The needle string was `cobalt-heron-4417`.

The intermediate arm (fullgraph + host all-reduce, coop for every row) is
what the tensor-core dispatch was compared with:

| | Host all-reduce | This build |
|---|---:|---:|
| Structured | 178.34 | 178.27 |
| Python median (mixed length) | 139.27 | 130.03 |
| Prose | 90.85 | 91.17 |
| Structured c4 / c8 | 337.6 / 459.6 | 343.9 / 586.6 |
| Prose c4 / c8 | 175.2 / 186.4 | 165.8 / 260.2 |
| TTFT median | 10.27 s | 8.37 s |
| KV pool | 687,201 | 616,916 |

Structured single-stream is the same hash on all three stacks. The 8-user gain
is the 43–64 row window. The time-to-first-token gain is the prefill kernel.

The port:

- Hidden 5120, local intermediate 640 and 512, 384 experts, top-6, mul1
  codebook, K3 on every served layer. A K2 down-projection that spilled 8
  bytes at the 255-register cap was split into two passes. The served
  kernels spill zero registers.
- Gate 0 on these weights. v2 loses to coop at 1–42 rows and beats the stock
  path from 48 rows through prefill, so coop keeps 1–42, `moe_out_v2` is
  only 43–64, and `moe_out_prefill` is 65–2048.
- One 540.4 MiB workspace per GPU, shared by every MoE layer, allocated when
  the first layer loads and before the KV cache is profiled. A buffer per
  layer does not fit beside this model on a 64 GB card.
- The hook stays on the existing EXL3 fused path. Activations stay BF16.
  The path is not turned on by casting FP16 to BF16.
- Host all-reduce, peer access left off, is the decode step from 172.7 to
  178.3 tok/s on the same structured text. The tensor-core path is what cuts
  prefill time and raises the 8-user rate.

The tree is that source. It does not contain weights and it does not publish
a container image. Build the extension against the same PyTorch the server
imports. `TORCH_LIBRARY` is not a stable ABI. Upstream licenses and the
checkpoint credit are in [NOTICE.md](NOTICE.md).

## DeepSeek V4.1 Flash 1017

The measurements above are the previous checkpoint, Mia-AiLab EXL3 3.0 bpw.
1017 is a different weight selection. Layers 0–39 routed experts are LibertAI
EXL3 2.9. Attention, shared experts, the head, the draft, and Engram stay the
official FP8 checkpoint. These kernels and this launch still run the experts.
They do not choose which file an attention weight comes from. Leave `head_bits`
unset and set `mtp_experts` to `source`, or the loader quantizes tensors that
serve left official.

The 1017 recipe, the hybrid config, the launch script, and those measurements
are [DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX).

## Kernel and resident-graph update — October 22, 2026

**Small measured decode gains, plus kernel A/B testing without reloading weights.**
On DeepSeek V4.1 Flash 1017, controlled counting confirmed **+2.86%** from the
EXL transpose change and **+1.74% further** from six-row host all-reduce tuning.
A separate eight-pair confirmation measured **184.21 → 188.13 tok/s (+2.21%)**
for the host tuning. These are counting-workload gains, not general agent speedups.

- [New six-row host all-reduce source patch](experiments/host-allreduce-six-row/):
  existing two-shot kernel, one block, 512 threads, exact SM80/TP4/BF16 shape guard.
- [Hybrid P2P on Broadwell](docs/hybrid-p2p-broadwell.md): local GPU pairs plus
  host communication across sockets; **572.60 → 586.31 C8 counting tok/s (+2.43%)**.
  Only the 48-row path won qualification; large-prefill paths remain unchanged.
- [Online CUDA graph optimization](docs/resident-cuda-graph-optimization.md):
  compatible kernel registration across four ranks in about **0.30 s**, with
  **600 node updates/rank in 2.7–3.9 ms**, no model reload or graph recapture.

The [1017 benchmark report](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-10-22/README.md)
includes all limitations: the full eight-task C8 aggregate was **271.57 tok/s**,
versus the previous **273.48** and published four-Spark **221.66**. No broad
throughput improvement is claimed. Source experiments are opt-in; existing
launch defaults are unchanged. The historical checkpoint results below retain
their original scope.

## Cooperative-decode experiment — October 21, 2026

The [six-row ExLlamaV3 cooperative-decode experiment](experiments/exl3-cooperative-decode/)
is maintained here with its measured source patch, CPU mapping check, source
identities and qualification notes. It transposes MMA operands on the tested
SM80 K2/K3 path. This is a separate experiment from the `tc_moe` kernels below;
the existing hook and launch scripts do not enable it.

The model-specific [controlled sweep tables](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/e560d05caa54969b5fc664056202bea863ca9513/benchmarks/2026-10-21/kernel-sweeps.md)
and [4× CMP 170HX versus 4× DGX Spark comparison](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/e560d05caa54969b5fc664056202bea863ca9513/benchmarks/2026-10-21/spark-comparison.md)
remain in the 1017 repository. The experiment measured modest decode gains;
its source template still needs a guard matching the tested serving scope
before general integration.

## What is on

| Rows | Kernel |
|---:|---|
| 1–42 | coop, kept from the optimized stack (`tokens * topk <= 256` at top-6) |
| 43–64 | `moe_out_v2` (`VLLM_EXL3_TC_MOE_V2=1`) |
| 65–2048 | `moe_out_prefill` (`VLLM_EXL3_TC_MOE_PREFILL=1`) |

One FP32 workspace per GPU, shared by every MoE layer, sized for the larger of
v2 at 64 rows and prefill at 2048 rows with intermediate 640. On this model
that buffer is 540.4 MiB. Allocate it when the first layer loads, before the
KV cache is profiled. A buffer per layer does not fit.

Activations on the tensor-core path are BF16. Weights passed into the op are
FP32. The codec decodes each lane into `mma.m16n8k16` FP16 fragments. The
multiply uses a hi/lo FP16 split of the activations, accumulates in FP32, and
does not let one row depend on another.

## Environment

```
VLLM_USE_V2_MODEL_RUNNER=0
DSV41_DSPARK_V1=1
VLLM_EXL3_COOP=1
VLLM_HOST_ALLREDUCE=1
NCCL_P2P_DISABLE=1
VLLM_EXL3_TC_MOE_V2=1
VLLM_EXL3_TC_MOE_PREFILL=1
VLLM_EXL3_TC_PREFILL_ROWS=2048
VLLM_EXL3_MOE_TP_ALIGN=128
VLLM_EXL3_MOE_TP_ROTATE=1
TORCH_CUDA_ARCH_LIST=8.0
```

Serve flags that match the measurement: `--tensor-parallel-size 4`,
`--max-model-len 262144`, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`,
`--gpu-memory-utilization 0.97`, `--kv-cache-dtype fp8_ds_mla`, Engram
`cpu_offload`, prefix caching, `--async-scheduling`, DSpark depth 5, and
CUDA graph mode `FULL_AND_PIECEWISE` with capture sizes
`1,2,3,4,6,8,12,18,24,30,36,42,48`.

`scripts/run-tp4-tcmoe.sh` is the launch template. Point it at your image,
your checkpoint directory, the host-allreduce cherry-pick, the hooked EXL3
plugin, and the built `.so`. There is no separate container image in this
repository.

Set `DSV41_ARENA_CACHE_DIR` to turn on the EXL3 arena cache. The script
passes `VLLM_EXL3_ARENA_CACHE=1` and mounts that directory. The first start
of a checkpoint writes the arenas. The next start reads them. The loader
skips the write if less than 4 GiB would remain free. One layer file is
about 1.1 GiB. Decode does not use the disk cache. It only shortens the
next load. Prefix caching is the separate `--enable-prefix-caching` flag.

## Plugin

Load `exl3_tc_hook.py` from the mounted path. Call `try_build(layer, inners)`
at the end of fused-expert setup. At the start of the fused MoE call:

```python
_tc_out = _tc_hook.try_apply(x2d, ids, weights, layer, expert_map, limit)
if _tc_out is not None:
    return _tc_out
```

`try_apply` returns `None` when the row count stays on coop, when the flags
are off, or when the layer fails the shape check. A skipped layer is coop,
not a silent tensor-core call. The startup line is
`EXL3 TC MoE layer=... v2=on prefill=on ws=540.4MiB`.

## Build

Measured toolchain: nvcc 13.0 (V13.0.88), torch 2.13.0+cu130. The build does
not need a GPU. The running server does.

```bash
python3 vllm_exl3/tc_moe/build_standalone.py \
  --out ./out --build-dir ./build
```

That writes `out/_dsv41_tc_moe_C.abi3.so`. Register-spill on the measured
build was zero for every `moe_lane_kernel` and GEMM. K2 down-projections that
spilled 8 bytes at the 255-register cap were split into two passes of at most
four row-groups. The served checkpoint did not use that K2 path.

`gate0.py` compares the extension with `apply_exl3_fused_moe` on one GPU.
Set `GATE0_MODEL` to the checkpoint and `GATE0_PLUGIN` to the EXL3 plugin.

## Hardware for these numbers

4× CMP 170HX, sm_80, 140 W power cap, in a 2× Xeon E5-2689 v4 (Broadwell)
host, PCIe Gen2 x16, no NVLink. Peer access left off. HBM is locked at
NDIV 64, **1728 MHz**. The memory clock `nvidia-smi` prints stays at the
card-reported 1458 MHz and does not show that lock, including while a
generation is running. Decode tok/s is `(completion_tokens - 1) / (end - first_token)`.
Medians use all five runs, including the cold first sample. Thinking was off.
Temperature 0, top_p 1, max tokens 400.

## Requirements

4× CMP 170HX, 64 GB each. This run fills about 62.5 GB of each card, so a 48 GB card does not fit. The checkpoint file is 674 GB on disk. It is read as needed and does not have to sit in RAM.

Of each card, about 56 GB is the model weights. The rest is the KV cache, which holds about 617,000 tokens, plus a small CUDA graph.

The host has 378 GB of RAM visible to the OS. That is the machine, not a minimum. While this model is serving, the host is using about 288 GB of that. Engram does not go on the GPUs. It is pinned in system RAM, about 190 GB. A 256 GB machine does not have room for that.

## Gate 0

One GPU, no server. Median of 200 CUDA events after 20 warm-ups. Decode
inside captured graphs. Prefill eager. Incumbent is coop through 42 rows and
the stock EXL3 path above that. Both served shapes are K3. K2 is not in this
checkpoint.

| Shape | 48 rows | 1024 rows | 2048 rows |
|---|---:|---:|---:|
| I=640 K3, v2 / stock ms | 1.610 / 2.871 | 6.724 / 13.200 | 10.958 / 21.889 |
| I=512 K3, v2 / stock ms | 1.528 / 2.164 | 5.677 / 8.897 | 9.319 / 14.713 |

v2 is slower than coop at 6, 12, and 36 rows on both shapes, which is why
those rows stay on coop. Correctness passed on layers 0, 10, 18, 22, 30, and
39, both intermediates: FP64 mean error 0.64× to 0.73× the incumbent, max
0.73× to 1.00×, clamp, three bitwise repeats, R=512 equal to R=2048, 36-row
CUDA-graph replay, and ids of −1 writing zeros.

## Open items

Python text has varied between a 99-token completion (`4a9988321ab7`) and a
153-token completion (`1a53abb1225f`) on both arms since host all-reduce was
enabled. The graphs + coop stack produced the 99-token completion on all five
runs. Per-output speed matches across arms. The mixed median is not a kernel
regression.

There is no scorer here for tool calls, JSON schema, or KL against the
graphs + coop stack. The quality evidence for this dispatch is Gate 0's FP64 pools and
the 9/9 needle passes. `nsys` was not available, so there is no kernel
profile in this repository.
