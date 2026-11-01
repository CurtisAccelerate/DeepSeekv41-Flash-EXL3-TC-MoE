# Qualification and immutable snapshot

The measured V6 image was loaded with BF16 KV and A/B/C users 1–8 on four
CMP 170HX cards at 140 W. The final state was healthy, private and paused with
A/REFERENCE restored. No production/Hermes promotion is implied.

| Pin | SHA-256 / identity |
|---|---|
| V6 image | `sha256:db41e3e783756467713e317b9d5aee224a4ca64fad9553666f2435bdf001d30d` |
| Required retained base image | `sha256:dc87ecb3dfdca94ce12a958c09a26314affd5f1fc2df7eb00dd2b82d9203675b` |
| Full tested source ZIP | `86d227270485e17bb75c515af6d87137c85fadb235d7c1fb58969542144eb96b` |
| Package tree | `1a342d522f0aa2d7866291b47e53eec22c02fe3cd5dea6b0030b9c3cbedf064e` |
| Effective live code/native tree, 180 files | `ad659febdc60dfd82a50ac718a318d934348745be07cc59ca219d9ff4b7ff80d` |
| Effective-code snapshot archive | `010378117f7d7d19745f1936f87ef8ac655ddbf64f3e9f6c5a4f8170333de999` |
| Full measured evidence ZIP | `5adf723c1debbe48882d16a177ef9540a747b54df198d509db1b9eb0521ee1b2` |
| New prefill native object | `912f563b8687ca9be38ab4733ce2f5551eb8851a30b38fff0bd7b33a747c20fc` |

The snapshot digests identify retained operator artifacts, not public download
URLs. Model weights, native binaries, private launch/state and complete server
logs are not committed. `SOURCE-MANIFEST.json` hashes each public export against
its original tested package path. This cleaned export is not byte-identical to
the original full source ZIP: omitted site/history documents are listed in its
scope, while exported source bytes are identical.

## Completed checks

- 96 exact-image CPU tests, no skips; 12 native host workspace/ABI queries;
  no CUDA initialization. Repeated on the cleaned public package with the
  NVIDIA runtime and networking disabled; see `PUBLIC-SOURCE-CPU-RECEIPT.json`.
- Mounted source/config/API audit and seven independent CPU integration checks.
- Per GPU: 85 integer fixtures and 9 calls through the same floating attention
  kernel, bitwise/finite; 144 actual-weight MoE fixtures; 62 retained-combiner
  cases; 70 dense cases and 12 interoperation cases.
- All-rank A/B/C users 1–8; live full-fusion counts exactly 20 ratio-1,
  18 ratio-2 and 2 sliding-window target layers. Draft work is counted separately.
- 60 independently verified unprofiled prefill requests across smoke, failed
  initial screen, warm confirmation and final 30-request sweep.
- 60 Morrow control requests, 28 fidelity responses with 16 exact oracle/repeat
  pairs, 8 delivery probes, 1,680 full SPARK requests and 72+96 teacher requests.
- 15 bounded traces: four CUDA workers and one frontend CPU trace per profile;
  profiling was off for all headline timings. Trace and raw-evidence hashes checked.

CPU checks do not qualify another GPU installation. Operator bitwise parity
does not imply deterministic full-model logits, general task quality, 128K
accuracy, sampled distribution equivalence for untested gates, or Hermes latency.

The initial 2K screen failure and count/seeded latency exceptions are retained
in the benchmark report. A warm follow-up was explicitly separate; no fallback
was counted as candidate execution and no source/health/parity guard was relaxed.
