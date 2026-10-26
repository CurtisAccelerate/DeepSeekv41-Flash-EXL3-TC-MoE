# Explicit startup integration contract

This is a scoped source bundle, not a drop-in image. Its Python package can be
imported with this directory on `PYTHONPATH`. Integrate at source callsites in
your loader, proposer and attention backend before starting a new process.
Do not replace functions or mutate captured graphs in a running server.

| Flag | Default | Implemented effect |
|---|---|---|
| `DS1017_NATIVE_MIXED` | `0` | `prepare_native` loads the reviewed cubin; `dispatch` uses it only for eligible calls |
| `W26_ATTENTION_SPLIT` | `0` | `prepare_attention` constructs/probes split8/core8/reducer4/copy state for explicit target capture |
| `W26_SAMPLER_BANK` | `0` | `sampler.prepare` collectively builds and checks the captured drafter; explicit selected state enables sampling |

They are independent. These flags have no effect unless the callsites below
are integrated. Selection is an explicit state argument: pass `None` to use the
reference implementation. Public launch scripts do not call these APIs.

## Native expert callsite

The integration accepts the exact existing `exl3_moe_coop` argument tuple used by
the pinned ExLlamaV3 interface. At initialization, in the worker's existing
current CUDA context, before warmup/capture:

```python
import json
from pathlib import Path
from ds1017_components.native import prepare_native, dispatch

build = Path(configured_build_directory)
identity = json.loads((build / 'build.json').read_text())
native_state = prepare_native(build, device, identity['sha256'])
```

At the normal extension callsite, use
`dispatch(native_state, original_exl3_moe_coop, *args)`. The original function
must accept the same 36 positional arguments and return void. The existing
caller owns scratch allocation and output tensors, as before. Keep the returned
backend alive for the entire lifetime of all graphs containing its kernels.
Its modules are intentionally retained until CUDA-context destruction. The
native adapter does not alter collective tuning or the original fallback.

## Attention target-bank callsite

`prepare_attention(device)` performs the generated empty-segment compile probe
before capture. This module uses exact source hashes for:

```
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
vllm/v1/attention/ops/fp8_sm80.py
```

The complete matching sources are in `retained/`. Run `check_sources.py` against
the integration's source tree/site-packages; it fails on mismatches instead of
assuming an arbitrary current vLLM version is compatible. These files also need
the compatible vLLM SM80/DeepSeek V4.1 APIs they import, PyTorch, and Triton.
The historical recipe's vLLM base is Morrowmake/vllm-cmp170hx at
`18305948acba3f4c4d912cdd56f2202d2ebd94e8`; that pin alone is **not** an assertion
that the two retained overrides or the DS1017 model integration are installed.

In a dedicated startup target capture, call `attention_state.begin_capture()`
outside `torch.cuda.graph(...)`, then at each compressed attention callsite:

```python
from ds1017_components.attention import decode_during_capture

decode_during_capture(attention_state, layer, original_decode, **attention_kwargs)
```

After leaving capture, require `attention_state.finish_capture()` to succeed.
It requires all 38 compressed main layers in order and rejects incomplete
coverage. The state owns one fixed-stream workspace. Pass `None` outside this
dedicated capture; replay the captured graph normally afterward. Never call
`run` as a general eager attention replacement. The component validates T6,
16 local heads, 512 dimensions, BF16 query/output, compression ratio, FP8 cache
byte strides, sink and ragged indices. A rejected optional bank must be discarded
as a whole; never advertise a partially captured bank as qualified.

## Drafter callsite

The proposer must expose the existing DSpark model, input/anchor buffers,
`num_speculative_tokens`, `_draft_topk`, `_conf_threshold`, and the unchanged
reference method `_sample_sequential_incumbent(num_reqs, hidden)`. Retain that
method under this name in the proposer class source; do not dynamically patch
a live instance. The model/logits API must support the `skip_gather=True` local
head path and the exact metadata checked by `eligibility_report`.

After weights, the CPU process group, and the existing PyNccl communicator are
ready, on all four ranks in the same startup order, before serving:

```python
from ds1017_components import sampler

sampler_state, preparation = sampler.prepare(proposer, cpu_group, pynccl)
```

Preparation checks the loaded full/local Markov projections and whole draft
call with fixed generated inputs, journals/restores the anchor, retains graph
roots, and verifies output ownership. It may return `None` for an unsupported
optional component. Exceptions classified `UnsafeSamplerPreparation` are fatal
to that startup attempt; do not continue claiming a healthy fallback after an
uncertain collective/capture failure. No communicator is replaced or destroyed.

At the sequential sampling callsite:

```python
tokens = sampler.sample_or_fallback(
    selected_sampler_state, proposer, num_reqs, hidden,
    proposer._sample_sequential_incumbent,
)
```

Pass the prepared state only when the same selection is committed on every
rank; pass `None` for the reference. The guard falls back for unsupported batch,
depth, confidence/top-k, hidden shape/dtype, margin observation or outer capture.
Target sampling and rejection verification remain the caller's unchanged code.

## Combined selection and remaining integration work

The measured combined bank was eligible only for the FULL descriptor
`(num_tokens=6, num_reqs=1, uniform=True, has_lora=False, num_active_loras=0)`.
A serving integration must reserve graph/workspace memory before KV budgeting,
journal all model-written state during startup qualification, restore it,
retain graph/module/workspace lifetimes, and commit compatible attention and
sampler selection across all ranks with requests drained. The components do
not provide that server-specific coordinator. A selection example is
`selected_sampler_state = sampler_state` only after the matching target bank
has completed capture and qualification on every rank; otherwise select the
reference graph and pass `None`.

For independent experiments, prepare and select only one component. For the
measured combined configuration, the dedicated attention target graph also
captured the guarded native-mixed dispatch and used the existing six-row host
collective. Keeping native mixed globally on fallback shapes would be a
different whole configuration and needs its own verification. No new
production integration, fallback composition, or quality equivalence is
qualified by this source publication.
