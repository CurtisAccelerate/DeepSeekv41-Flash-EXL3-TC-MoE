# Attribution and source lineage

This integration and the November 1 measurements are maintained by
[CurtisAccelerate](https://github.com/CurtisAccelerate). It builds on the retained
optimized serving stack; it does not claim authorship of its upstream kernels,
model, quantization or pre-existing optimizations.

| Component | Source / contribution boundary |
|---|---|
| Short-row FP8 dense and `wo_a`, FP8 draft head, fused Markov selection | GNmac's recipe and donor code; ported into our ABI, graph bank, sampling and qualification guards |
| Adaptive verification and masked speculative expert work | GNmac/Morrowmake designs retained as optional source; tested in the November2 finite locally calibrated program; fixedfive/gateoff selected, adaptive/gated policies not promoted |
| Tensor-core MoE prefill schedule | Morrowmake operator foundations, integrated here for hidden 5120, intermediate 512/640, K2/K3, exact API/workspace/dispatch qualification |
| Full direct integer combine/ragged pack | This V6 integration: actual ratio 1/2/0 coverage, closed-form row offsets, preserved invalid slots/order/bounds, existing attention API |
| Floating sparse attention and index semantics | Retained vLLM ROCm primitives and SM80 adaptation; floating attention is not replaced by the integer fusion |
| Existing optimized six-row/cooperative/attention/Engram/host route work | Retained CurtisAccelerate work and its separately credited Morrowmake/ExLlamaV3/vLLM foundations; see the root NOTICE and prior-work record |

## GNmac

Related published recipe examined at
[`GNmac-lab/deepseek-v41-flash-1017-cmp170hx@2ad5741`](https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx/tree/2ad5741ab1a278d64f33b842706a41717dfbf181).
The exact retained donor artifact was image
`cdaad5bb40b669f1cee7b44fa5f79f61332fbc021bf56c0a3e88df83d266fddd`;
an image pin is not asserted to be a Git commit identity. The GNmac repository's
[MIT grant](licenses/GNmac-MIT.txt), copyright 2026 GNmac-lab, is included.
Existing embedded upstream notices remain in the copied files.

The FP8/native donor sources are under `source/package/donor-cuda/`. Our
integration does not make GNmac's measured speed or any universal quality claim
our own. Historical GNmac numbers in the companion benchmark report are our
measurements of its code on our 140 W machine, not its published 180 W results.

## Morrowmake, ExLlamaV3 and vLLM

The new MoE prefill source identifies
`Morrowmake/vllm-cmp170hx@a0c57b52376efd67c57735bc949d4bafa22ae29e`,
`csrc/exl3/moe_prefill_try.cu` (Apache-2.0). It is separate from the older
Morrowmake tensor-core/host-collective pin documented in the root NOTICE.
Its preserved comments describe the fixed-order arithmetic and our TP shape
instantiations; the full-model quality claim is narrower than operator parity.

EXL3 codec material retains the Turboderp MIT grant embedded in `codec.cuh`.
vLLM-derived files retain their Apache-2.0 headers; the Apache license is also
[included](licenses/Apache-2.0.txt). The imported EXL3 reader retains the Mia AI
Lab/Turboderp notices. The model and LibertAI packs are not redistributed here.

The attention lineage and named upstream contributors remain credited in the
root [NOTICE](../../NOTICE.md) and
[prior-work record](../ds1017-native-mixed-and-split/PRIOR-WORK.md).
The new integer kernel is not a claim to have invented sparse attention,
speculative decoding, FP8 arithmetic or the existing ragged attention interface.
