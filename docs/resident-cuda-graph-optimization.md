# Online CUDA graph optimization with resident weights

**Compatible kernel experiments in seconds, with no weight reload or graph
recapture.** One full-model trial registered a new EXL candidate across four
ranks in about **0.30 s**; changing **600 executable nodes per rank** took
**2.7–3.9 ms per rank**. A collective-profile transaction took about **0.50 s**.
These are measured control operations, not total experiment time: draining,
validation, resume and inference add overhead. No inference speedup is implied.

## How it works

At initial startup we retain the source CUDA graphs, executable handles and
node inventory, including every segment of the engine's breakable graphs.
Model weights, KV allocations and Engram mappings stay owned by the same
workers. Boot-time baseline/candidate graph selection was the first A/B path;
the later implementation can also register a compatible cubin after capture.

The central CUDA API is `cuGraphExecKernelNodeSetParams`: update the function
and qualified launch parameters on an existing executable node. The original
node must remain available and the function must belong to the same CUDA
context. NVIDIA documents further restrictions, including dynamic parallelism
and device-side launches; graph objects are not thread-safe.
[CUDA 12.8.1 Driver API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__GRAPH.html).

Our integration adds the distributed controls that a single API call lacks:

1. Close external inference admission, drain requests, synchronize all ranks.
2. Verify candidate hash, ABI, symbols, device, tensor shapes and scratch needs.
   Load the module into each existing worker context with a memory allowance.
3. Inventory exact graph/node identities and owned argument blocks. For the
   qualified EXL path: six rows, hidden 5120, top-6, intermediate 512/640,
   K2/K3, 344-byte parameter ABI, and complete rotation/gate/down node triples.
4. Prepare and apply on every rank. Patch executable nodes and synchronize
   eager dispatch with the same selection. Commit only after every rank agrees.
5. On expected rejection, restore every modified node while admission stays
   closed. On an unsafe CUDA error or failed rollback, prevent further serving
   until recovery. Never resume a partially switched distributed model.
6. Verify outputs, actual route execution, cache counts and speculative work;
   run paired A/B or AB/BA measurements; restore or keep the qualified winner.

For collective profiles we also preserve sequence state, hold deep-owned
argument storage, and verify exact copy-node source/destination ranges and
dependencies before disabling a redundant copy. Modules and buffers stay alive
while referenced by any executable. Removing a Python function or replacing a
shared library on disk is not equivalent to changing a captured kernel node.

## What was demonstrated

| Test | Evidence |
|---|---|
| Component function replacement | 32 cases; 1,152 bitwise dynamic-input checks; 32 partial rollback checks |
| Tiny TP4/native HTTP integration | Late registration, repeated switches, rejection and partial-rank rollback |
| Mixed collective replay | 4,680 outputs across four GPUs, changing inputs and rank skew |
| Full-model late registration | Same workers, model/Engram mappings and capture generation; 600 nodes/rank; zero recaptures |
| Full-model prefix-cache preservation | 8K and 32K cases retained cached tokens through switching and restoration |
| Later paired kernel/P2P campaign | 732 short-sweep requests; same resident workers/capture generation |
| Context + standardized Spark campaign | 54 completed context requests and 804 Spark requests; no reload/recapture |
| Resident Spark on/off subset | 171 requests; no reload/recapture |

The newly registered EXL candidate did not establish a worthwhile speed gain
and was restored. The retained improvements were the earlier transpose kernel,
six-row collective tuning, and later the qualified 48-row hybrid P2P route.
Successful iteration infrastructure does not make every candidate a winner.

## Boundaries

This is **online residency with a drained maintenance window**, not concurrent
mutation while requests execute. It requires preparation at initial capture;
an arbitrary already-running server may not have the retained graph handles.
The demonstrated path preserves the kernel ABI, tensor layout, graph topology
and memory ownership. Changes to the quant/model, scratch layout, topology or
driver need separate integration and may require recapture or restart. An
unsafe CUDA error can still require worker recovery.

CurtisAccelerate's persistent Engram/n-gram backing and reattachment integration
is a separate loading optimization: it can survive worker replacement while
its backing remains alive. The full-payload four-rank attachment proof checked
bit-exact native gathers, unchanged payload hashes and retention after child
exit. Same-worker graph updates
avoid even reattachment. This publication does not claim weight hot-swapping,
reboot persistence, or a general GPU IPC keeper in the measured server.

This is an implementation report, not a standalone live-patching SDK. The
deployment/admission controller is not included. [Measurements and source
patches](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-10-22/README.md) describe the
qualified scope without exposing infrastructure identifiers.
