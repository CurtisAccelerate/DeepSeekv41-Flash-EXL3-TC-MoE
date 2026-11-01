# Next experiment: GNmac verification policy on the retained build

**Prepared design, not a measured or deployment-qualified upgrade.** The current
V6 snapshot is preserved. Its loaded bank exposes A/B/C and oracle controls,
all at fixed width five; it cannot enable the complete adaptive/gated policy
by changing an environment variable after capture.

## Why this is worth testing

GNmac's `dsv41_dynk.py` still drafts five tokens, then chooses whether the target
verifies three or five. A shorter verification can reduce target attention,
expert and collective rows when late drafts rarely survive. The donor's static
cost table was measured on another checkpoint/run and interpolates several
concurrencies. Copying those prices would not establish the right choice here.

We already tested the FP8 draft head and fused Markov selection as C. That
does not test all of GNmac's drafting policy. The remaining historical SPARK
gap could involve verification width, proposal sampling, gate behavior, target
math, templates or other run differences; no causal share is established.

## Isolated arms for one future qualified load

Hold BF16 KV, full integer prefill fusion, Marlin dense prefill, weights,
target C math, communication route and power fixed:

| Arm | Verification width | Confidence gate | Purpose |
|---|---|---|---|
| D0 | Fixed 5 | Off | Frozen current C baseline |
| D1 | Fixed 3 | Off | Price width reduction separately |
| D2 | Fixed 5 | 0.10, greedy-only initially | Isolate masked speculative expert work |
| D3 | Adaptive 3/5 | Off | Local acceptance/cost policy |
| D4, conditional | Adaptive 3/5 | Qualified 0.10 policy | Test interaction only after D1–D3 pass |

Existing `C3`, `C-AUTO` and `C-ALL` source is a starting point, not a claim
that all these exact combinations were captured or tested. A targeted drafter
scope should avoid capturing unrelated old ablation variants. Preserve the
startup memory floor and require all-rank readiness before publishing an arm.

Measure all sixteen width costs for users 1–8 and k=3/5 on this machine; do not
interpolate. Bind prices to the exact source/image/precision/gate/route identity.
Record full iteration cost including proposal, verification, sampling, policy
and communication, plus target-only timings. A gate-on policy needs compatible
prices rather than a table measured with the gate off.

Report per-position conditional reach/acceptance, proposed and verified tokens,
accepted tokens per iteration, width histogram, rejected work and fallback
counts. Record warm and first-use cases, full wall, TTFT and content gaps. Warm
representative full chunks, padded multiuser and seeded rejection/recovery
paths before the warm headline; retain the cold results separately.

## Correctness checks that matter

- Same target C, fixed frozen prompt/token inputs; greedy token IDs must match
  D0 and a forced-rejection oracle. Gate-masked positions must force rejection,
  with no stale masks on short/mixed/prefill/nonuniform/narrow/fallback steps.
- Gate changes speculative target expert work, not merely a draft-head cast.
  Prove that masked rows cannot influence accepted output; do not infer this
  from teacher-forced PPL alone.
- Keep the gated arm greedy-only until probability correction, recovery
  distribution and per-request RNG handling pass sampled/seeded tests.
- Explicitly separate target temperature, proposal temperature, seed and
  verification width. Current unseeded one-user T1 samples proposals, while
  its seed1017 control uses greedy proposals. Changing the seed also changes
  target draws, so those controls are not a clean proposal-only ablation.
- Match exact bodies, actual token counts, cache state and all-rank native
  hits. Use original Morrow and SPARK protocols, paired order and repeats.
  Teacher NLL/PPL is useful for target math but bypasses proposal generation.

Benchmark on the current B/C arms can first isolate already-integrated draft
head/Markov changes without a reload. Full D0–D4 needs another cold qualified
load; no such reload or live policy mutation is performed by this repository
update. Do not label the remaining speed gap explained until that comparison.
