# DS1017 native mixed experts and split attention / captured drafter

**Experimental, opt-in components for 4× CMP 170HX.** This source bundle is for
integration into a compatible startup/capture path. It is not a complete server
image or an installer for the repository's historical launch scripts. The
independent flags below are implemented here; exporting them alone does not
modify an existing server. No private operator coordinator is required to import
the components, but a serving integrator must supply the graph lifecycle and
all-rank selection described in [INTEGRATION.md](INTEGRATION.md).

The measured implementation passed initial numerical/runtime checks and was evaluated in limited
matched end-to-end tests; task-quality results were mixed. Split attention changes floating-point
reduction/rounding and is not bit-exact with the reference; broader verification
of numerical accuracy and task quality is still warranted. Results do not
establish universal quality equivalence. The public explicit-state adapters
were extracted from that implementation and checked on CPU; their integration
into a new server has not been GPU-qualified by this publication.

## What is new

1. **Native mixed experts:** K2 retains guarded parameters and writer-fence
   handling; K3 specializes the fixed eligible six-row native MoE contract.
   The K3 path folds fixed shape/activation/nullable-field branches. It is not
   safe as an unconditional replacement for generic EXL3 dispatch. The included
   adapter checks the original 36-argument call, SM80, six rows, top-6, hidden
   5120, 384 local experts, intermediate 512/640, K2/K3, FP16 input/scratch,
   FP32 output, no bias/shared additions and required capacities. Everything
   else falls back to the caller's existing implementation. Counting measured
   **187.723 → 195.423 tok/s (+4.10%)** over transpose + six-row host tuning;
   this is a controlled counting result, not a universal code/prose gain.
2. **Split attention plus captured vocabulary-sharded drafter**, layered on
   native mixed for the measured single-request T6/B1 target graph:
   - Attention uses **8 splits, 8 core warps, 4 reducer warps, then output copy**.
     It partitions the selected ragged attention work and reduces partial
     results. Selected IDs, cache layout and weights stay the same. Main
     layers 0/1 and draft layers remain on the reference attention path.
   - The greedy drafter retains norm/local-head work, computes each rank's
     32,320 vocabulary rows of the 129,280-row Markov projection, exchanges
     local maximum value/token-ID pairs, and captures the five-position chain.
     This reduces projection and exchange work. The target rejection sampler,
     target RNG, target verification, and speculative depth **K5** are unchanged.
     Limited loaded-weight full/local projection and whole-call checks passed;
     that does not prove identical argmax for every possible hidden vector.

These build on the previously published transpose and six-row collective
improvements. They do not introduce a new quantization, SM unlock, or P2P/prefill
optimization. The weights remain the same LibertAI 1017 routed-expert selection,
approximately 2.9 bpw, with the other components from the official checkpoint.

## Files and checks

- `native/`: complete measured specialization and headers, MIT ExLlamaV3
  attribution; `instantiate.cu` emits the four K2/K3 kernels plus rotation.
- `ds1017_components/`: metadata-guarded native adapter, attention state/profile,
  captured sampler, and 344-byte parameter ABI. Explicit startup preparation;
  no graph mutation or service-control API.
- `retained/`: exact LF source snapshots of the two vLLM attention dependencies,
  under Apache-2.0. Their existing kernels are reused, not claimed as new here.
- `fixtures/`: generated FP8/BF16 cache layout, independent FP64 attention
  reference, and error metrics. These contain no model data.
- `provenance.json`: measured source identities and public extraction changes.

Run from this directory, with Python 3.12:

```bash
python check_sources.py
python test_cpu.py
# Optional: validate an existing compatible installation, without importing it:
python check_sources.py --site-packages /path/to/site-packages
```

These check Python syntax, both retained attention kernel signatures, exact
dependency hashes, parameter ABI, default-off behavior, and supported/rejected
native metadata. They do not perform GPU numerical or end-to-end testing.
The publication also compiled the supplied native sources with CUDA 12.8:
all five kernel machine-code sections matched the measured native-mixed
artifact. Full cubin hashes can differ because of line information/build paths.

Build on a machine with CUDA 12.8 and the PyTorch headers from the target image:

```bash
python build_native.py --torch-include /path/to/torch/include --out ./build
```

No GPU is needed to compile. On Windows, use a Visual Studio x64 developer
shell and pass `--nvcc` if needed. The builder emits `native_mixed.cubin` and
`build.json`; verify its recorded hash before passing it to `prepare_native`.
Do not publish or reuse a new binary as qualified merely because it compiled.
The measured environment used PyTorch 2.13.0+cu130; the native specialization
was built separately with CUDA 12.8. Native launch is Linux CUDA-driver based.

The 1017 repository owns the [performance and quality report](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/tree/main/benchmarks/2026-10-25).
Unsupported combined shapes in the measured configuration used the older
transpose/boot + six-row fallback, while its reference used global native mixed
+ six. Concurrency measurements therefore compare whole configurations and do
not isolate attention/drafter effects. No general C4/C8 or cold-prefill gain is
established.
