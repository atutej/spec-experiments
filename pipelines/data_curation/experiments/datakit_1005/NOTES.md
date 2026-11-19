# datakit_1005

Mix `datakit_sft_1005` (see ../../README.md). Status as of 2026-11-19: planned and smoke-tested on TACC Horizon; the full build has not run.

- `plan`: 216 sources resolved, 0 errors, 174 unique HF downloads, ~801.5 GB (nemotron_sft_v3 690 GB, open_swe_traces 43, penfever-traces 43, agenttrove 20).
- Smoke test (6 small sources, ROWS=2000, run on a Horizon login node): fetch/stats/select/write/finalize pass; tokenizer + training-template check passes
  against the 10.05 release; loss-mask check passes on all sampled rows (508/508).
- Whole-source selection on those six sources reproduces Marin's recorded store sizes exactly for `agenttrove-glm53-compactions` (188,537,557 tokens) and
  `glm-5.2-kernelgym-rollouts` (16,082,348) (`experiments/grug_sft/reusable_stores.json` at ce7c77a).
- Not verified: total token count of the full build vs ~234.4B, and the row set of sources the original rebuilt after its 09-29 fixes (e.g. Terminus).
