# Hybrid P2P on dual-socket Broadwell

**A narrow C8 counting gain: 572.60 → 586.31 aggregate tok/s (+2.43% paired).**
Four matched, quiet pairs support this result. It is not evidence that enabling
P2P generally improves Broadwell inference, or that prefill became faster.

## What we changed

The topology is four CMP 170HX cards on PCIe Gen2 ×16, two per CPU socket,
without NVLink. Peer access was qualified within each socket; cross-socket
peer access was unsupported. The selected transport uses local GPU peer
communication inside each pair and host-staged communication across sockets.
Host staging means GPU access to pinned host memory; it does not mean CPU
cores perform the tensor sum. It still consumes host-memory/inter-socket bandwidth.

The measured policy is deliberately small:

| Message | Selected path |
|---|---|
| 6 × 5120 BF16 = 61,440 bytes | Tuned six-row host all-reduce |
| 48 × 5120 BF16 = 491,520 bytes | Socket-local P2P plus host cross-socket path |
| Large prefill | Existing NCCL path |
| Other sizes | Existing dispatch |

The 48-row path uses a dedicated host/IPC arena and sequence counters, eight
blocks of 256 threads, and a 1,024-element tile schedule. Initialization and
peer/arena qualification happen before model capture. Every rank must agree
on topology and readiness. Switching profiles edits retained graph nodes while
requests are drained; exact matching DMA copies are disabled only when the
replacement kernel supplies that work. Buffer/argument ownership is retained.

We tested broader hierarchical/prefill paths and did not select them because
they lost to the incumbent. We also rejected the four-block small host path
after a mixed-size sequencing failure. A favorable isolated microbenchmark
was not sufficient for adoption. Driver/BIOS changes are not packaged here;
enabling peer capability alone is not this transport implementation.

## Serving result and limits

| C8 task, four rounds | Host-only aggregate tok/s | Hybrid aggregate tok/s | Paired change | Full work parity |
|---|---:|---:|---:|---|
| Counting / structured | 572.60 | 586.31 | +2.43% | 4/4 |
| Code | 325.32 | 332.62 | +2.25% observed | 0/4 |
| Prose | 266.31 | 266.63 | +0.33% observed | 0/4 |

Code/prose differences are descriptive because speculative work differed.
C1 executed zero pair-P2P calls; C1 on/off differences cannot be attributed
to this kernel. The 2K–128K context sweep also executed zero pair calls and
was essentially flat. The broader eight-task Spark protocol produced
**271.57 C8 aggregate tok/s versus the previous 273.48**, with higher TTFT;
the comparison changed both six-row host tuning and P2P and does not isolate
either. A later resident on/off subset found no consistent P2P regression,
but had no fully matched C8 work pair and cannot prove a broad benefit.

Recommendation: retain this qualified 48-row route on the measured system;
keep adoption elsewhere opt-in and topology-tested. Negative Broadwell P2P
results remain relevant to other message sizes and transport designs.

This page documents the tested design and evidence, not an installable P2P
runtime. The private host-control/deployment bundle is not distributed.
[Complete tables and sanitized numerical records](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-10-22/README.md).
