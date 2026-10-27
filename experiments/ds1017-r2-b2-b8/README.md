# 2–8 user decode lane

This is the batch-2 through batch-8 path measured on October 26. It is not the
one-user bundle in `experiments/ds1017-native-mixed-and-split/`. That bundle
stays the C1 source. This directory is the multi-user contract and the modules
the server imports only when `DSV41_R2=1`.

`DSV41_R2` defaults off. Unset or `0` builds no plan and leaves every hook on
the existing path. `1` is the only armed value. A malformed flag raises at boot.

These files belong in the server `hybrid` package. Importing this directory by
itself does not capture graphs and does not change a running server. There is
still no pullable image. The measurements are in the
[October 26 1017 report](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/blob/main/benchmarks/2026-10-26/README.md).

## Flags

| Flag | Default | Effect |
|---|---|---|
| `DSV41_R2` | `0` | Master switch. `1` arms batches 2–8. |
| `DSV41_R2_BATCHES` | `2-8` | Which batch sizes get a plan. |
| `DSV41_R2_COMPONENTS` | attention, experts, collective, sampler, sampled | Which pieces the plan records. |
| `DSV41_R2_PAD7` | `1` | Batch 7 uses the 48-row batch-8 graph. |
| `DSV41_R2_SPLITS` | per batch, nearest the one-user 96-CTA shape | Attention split. Empty uses the default. |
| `DSV41_R2_EXPERTS` | coop, or `tc` when tensor-core v2 covers the row count | Expert kind. |
| `DSV41_R2_COLLECTIVE` | pair at 48 rows, otherwise the host all-reduce size rule | Collective. |
| `DSV41_R2_CAP_MIB` | `768` | Retained-graph cap checked at startup. |
| `VLLM_EXL3_TC_MOE_V2` | `0` | Required for the tensor-core expert path on the 48-row graph. |

With `DSV41_R2=1`, `DSV41_R2_PAD7=1`, and `VLLM_EXL3_TC_MOE_V2=1`, and no
per-batch overrides, the defaults are the measured plan: cooperative experts
for batches 2–6, tensor-core experts and the pair collective for batch 8, and
batch 7 served by that batch-8 graph. Draft length stays 5, so each request is
6 rows.

The drafter at 2–8 stays greedy unless the request also sends `dsv41_draft_tq`.
That flag is the sampled drafter. It is not turned on by `DSV41_R2`. The
October 26 serving choice leaves it off for batches of 2–8 and uses it for a
single temperature-1 request, which is outside this lane.

## Files

| File | Role |
|---|---|
| `hybrid/r2_contract.py` | Flag parser, batch plans, hit ledger. No torch. |
| `hybrid/r2_attention.py` | Multi-user attention replay. |
| `hybrid/r2_experts.py` | Cooperative, native, and tensor-core expert dispatch. |
| `hybrid/r2_sampler.py` | Captured greedy rows and captured sampled rows. |
| `hybrid/r2_bank.py` | Replay gate. Runs only when this lane is selected. |
| `hybrid/r2_startup.py` | Capture before serving. Called only when `DSV41_R2=1`. |
