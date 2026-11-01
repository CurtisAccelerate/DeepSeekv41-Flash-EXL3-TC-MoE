# Review map

This is a complete source export for the retained A/B/C runtime plus the new
prefill work, not a claim that every imported vLLM or earlier optimization file
was newly written for V6. The source/native/build/test export is about 2.7 MB;
compiled modules, weights, caches and complete private logs are excluded.

Start with the five integration edits recorded by the exact-image audit:

- `hybrid/abc_kernels.py` and `hybrid/abc_startup.py`: component preparation,
  readiness, graph/session integration and all-rank qualification.
- `vllm/models/deepseek_v4_1/amd/rocm.py`: direct ragged candidate, original
  fallback retained, same floating attention call.
- `host-overlay/integration/w9_exl_ab/worker_extension.py` and
  `host-overlay/resident-api/w9_resident_api/control.py`: guarded hot prefill
  selection/status integrated with the existing private A/B/C session.

Then read `hybrid/prefill_runtime.py`, `hybrid/prefill_ragged/`, their source/API
contracts and the matching tests. `prefill-native/` contains the new expert
prefill schedule adapted from Morrowmake. Optional dense variants and their
tests are retained but not promoted as accuracy-preserving speed winners.

`CHANGE-MAP.json` classifies overlay files against the retained base's source
manifest. `SOURCE-MANIFEST.json` records exact tested bytes for every exported
file, including native/tests/tools. Existing A/B/C/GNmac/vLLM dependencies are
included so the integration can be inspected without an unavailable running
machine; inspect their lineage in `UPSTREAM-LINEAGE.json` and `NOTICE.md`.

No native/model source was reformatted for publication. Source attributes
preserve its original line endings and historical EOF layout so the hashes
continue to match; prose and metadata are newly written for this update.
