# DeepSeek V4.1 Flash — tensor-core EXL3 MoE on CMP 170HX

Curtis White ([CurtisAccelerate](https://github.com/CurtisAccelerate)).

Serving recipe for DeepSeek V4.1 Flash, EXL3, tensor parallel 4, on 4× CMP
170HX (sm_80, 140 W, HBM locked at 1728 MHz) in a 2× Xeon E5-2689 v4
Broadwell host, PCIe Gen2 x16, no NVLink. Morrowmake published the
tensor-core MoE kernels and the host-staged all-reduce for
Qwen3.8-Flash-Next. This repository is those kernels brought onto this
checkpoint: the shapes, the row rule, the shared workspace, the EXL3 hook,
and the measurements.

On that machine the shipped stack does **178.3 tok/s** structured decode,
**587 tok/s** across 8 users, and a median time to first token of **8.37 s**
from 2K through 255K context (10.9 s at 255,424 tokens). The numbers it is
compared with are the optimized stack already in service on this host:
full CUDA graphs, coop, and NCCL. Coop is part of that optimization. It is
not an untouched default. That stack did 172.7 tok/s, 456 tok/s at 8 users,
and 10.15 s. Needles passed 9/9 through 256K. The KV pool is 616,916 tokens.

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
plugin, and the built `.so`.

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

## Morrowmake's published CMP 170HX figures

These are their numbers, on a different model, on two cards. They are here
so the hardware class is comparable. They are not a head-to-head.

The number from this port is below. Same card, four of them, DeepSeek V4.1
Flash, the shipped kernels, HBM lock 1728 MHz. The cap was raised only for
this measurement and put back to 140 W when it finished. Decode does not
move with the cap.
Prefill gains a few tens of tokens per second from 140 W to 180 W and stays
near 1,500 tok/s, not their 2,651.

Mia decode cells, median of 5, temperature 0, thinking off. `agg` is the
8-user total.

| W | structured c1 | structured c8 | code c1 | code c8 | prose c1 | prose c8 |
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

Same loaded server, model not restarted. The lock was dropped from NDIV 64
(1728 MHz) to NDIV 54 (1458 MHz) and put back. Median of 5 for decode, median
of 3 for the cold curve. `nvidia-smi` printed 1458 MHz on every sample at
both locks.

| | 1728 MHz | 1458 MHz |
|---|---:|---:|
| Structured c1 | 178.5 | 174.2 |
| Structured c8 | 575.7 | 568.2 |
| Code c1 | 138.6 | 139.1 |
| Prose c1 | 90.2 | 93.1 |
| 2,048 prefill / decode | 1,556 / 149.4 | 1,561 / 145.8 |
| 32,768 prefill / decode | 1,532 / 148.5 | 1,512 / 144.6 |

[Qwen3.8-Flash-Next EXL3 5.05 bpw, 2× CMP 170HX](https://github.com/Morrowmake/qwen38-flash-next-cmp170hx-recipe)
(recipe 1.0.0):

| | |
|---|---:|
| 1 user | 152.8 tok/s |
| 8 users | 674.7 tok/s |
| cold prefill | 2,651 tok/s |

Their host-allreduce microbench, same engine commit, message size versus NCCL:
8 KiB 2.04×, 32 KiB 1.06×, 64 KiB 1.59×, 128 KiB 1.43×, 256 KiB 1.41×.
Messages above 512 KiB stay on NCCL. A DeepSeek verify step is about 10 KiB
per token (hidden 5120, BF16), so one user is about 60 KiB and eight users
are about 480 KiB. That microbench is theirs. The serving tables below are
this port, on 4× CMP 170HX.

## Serving results

Checkpoint: Mia-AiLab DeepSeek-V4.1-Flash-EXL3-3.0bpw. The left column is
the optimized stack already in service: full CUDA graphs, coop, and NCCL.
Coop is that optimization, not a stock default. Shipped keeps coop for 1–42
rows and adds host all-reduce plus the tensor-core dispatch above.

| | Graphs + coop | Shipped | |
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

| | Host all-reduce | Shipped |
|---|---:|---:|
| Structured | 178.34 | 178.27 |
| Python median (mixed length) | 139.27 | 130.03 |
| Prose | 90.85 | 91.17 |
| Structured c4 / c8 | 337.6 / 459.6 | 343.9 / 586.6 |
| Prose c4 / c8 | 175.2 / 186.4 | 165.8 / 260.2 |
| TTFT median | 10.27 s | 8.37 s |
| KV pool | 687,201 | 616,916 |

Structured single-stream is the same hash on all three stacks. The c8 gain
is the 43–64 row window. The TTFT gain is the prefill kernel.

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
