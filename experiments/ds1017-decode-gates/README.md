# 2–6 user gate kernel — October 28, 2026

This is the source of the gate, up, and down libraries measured in the
[October 28 1017 report](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-10-28/README.md).

It is not turned on by cloning this repository. The historical launch scripts
do not load it. The shared libraries and the resident dispatch are not in
this tree.

## What is new

The cooperative kernel takes a row count as a template argument, `SPEC`.
`launch.cu` compiles a separate gate/up and down kernel for 8, 12, 18, 24,
30, and 36 rows. Those are 2 through 6 users, because each decode user is 6
rows. `fold_serve.py` launches that library only for those shapes, and only
when the call matches the live expert contract: hidden 5120, top-6, 384
experts, K2 or K3, intermediate 512 or 640, gated SiLU with limit 10.

The serving build keeps `GU_DIRECT` false, so gate/up stays on the FP16 fold.
K3 is `libgu_rows.so`. K2 is `libgu_k2.so`. The pins in `fold_serve.py` are
the libraries that were measured.

Six rows stay on the October 26 mixed-attention kernel. A fused copy of this
body under that kernel's symbol names was built and was not faster, so it is
not the serving path and it is not in this directory. Forty-two rows are
compiled and are not a captured graph. Seven and eight users are 48 rows and
take the tensor-core path.

The kernel matches the expert kernel it replaces bit for bit on the shapes
it serves. The decode gain is in the 1017 tables: about 6–15% at 2–6 users.

## Lineage

The kernel body is the ExLlamaV3 cooperative decode kernel, including the
transpose schedule from
[experiments/exl3-cooperative-decode](../exl3-cooperative-decode/README.md).
The October 28 change is the per-row specialization and the serving launcher.
Upstream kernel authors remain credited in
[PRIOR-WORK.md](../ds1017-native-mixed-and-split/PRIOR-WORK.md).

## Build

From this directory, for sm_80:

```bash
nvcc -shared -Xcompiler -fPIC -arch=sm_80 -std=c++17 -O3 \
  --expt-relaxed-constexpr -DGU_DIRECT=false -DKBITS=3 \
  -I src launch.cu -lcuda -o libgu_rows.so
nvcc -shared -Xcompiler -fPIC -arch=sm_80 -std=c++17 -O3 \
  --expt-relaxed-constexpr -DGU_DIRECT=false -DKBITS=2 \
  -I src launch.cu -lcuda -o libgu_k2.so
```

Building these libraries does not install them into a running server.
