# Scope and attribution

The [attention prior-work and contribution record](PRIOR-WORK.md) identifies
the upstream sparse-attention, split-K and SM80 work by Hexiang Wang, Fangzhou
Ai, Tuukka Sarvi, haosdent, thomaslwang and the vLLM/PixelML contributors, with
source commits and retained-file hashes. CurtisAccelerate's DS1017 adaptation,
launch tuning, capture integration, drafter implementation and measurements
are described separately there.

The native sources derive from Turboderp's ExLlamaV3, upstream commit
`da68d57de61781e5b74ad68698ac4caf36408d7a`, and the previously published
six-row transpose experiment. The included MIT license applies to that code.
The native mixed specialization retains guarded K2 behavior and specializes
K3 for the documented six-row contract. It must not replace unrestricted
upstream dispatch.

The retained vLLM attention files carry their original Apache-2.0 SPDX notices;
their license is included in `retained/LICENSE`. They are exact dependencies,
not newly invented attention kernels. The split component changes scheduling,
split/reduction geometry and capture integration around those kernels.

The host-allreduce foundation is credited to Morrowmake. CurtisAccelerate's
DeepSeek port of the earlier tensor-core/prefill operators has Morrowmake source
lineage; the six-row transpose and native-mixed changes in this component are
our implementations on Turboderp's cooperative kernels. Our DeepSeek serving
build and Engram retention integration are also documented in the
[contribution breakdown](PRIOR-WORK.md#deepseek-build-expert-kernels-and-loading-separate-contributions).
SM80 attention work comes from PixelML/sm80vllm and vLLM contributors. DeepSeek
provides the checkpoint and DSpark architecture; LibertAI provides the EXL3
2.9 bpw routed-expert pack. No weights or model-generated task archives are
included here.

The Python components are public extractions of the tested implementation:
private coordinator selection is replaced by explicit state arguments and
startup calls. Numerical algorithms, metadata checks, loaded-weight sampler
checks and attention launch geometry are retained. This publication adds CPU
checks for those adapters and verifies the native build's five machine-code
sections against the measured binary. It does not claim new end-to-end GPU
qualification of an arbitrary consumer's integration. Use Python without `-O`;
the retained preparation code uses assertions for several startup invariants.

Performance/quality measurements refer to the original integrated stack, not
to an untested combination assembled from these modules. No universal task
quality equivalence or general concurrency/prefill improvement is established.
