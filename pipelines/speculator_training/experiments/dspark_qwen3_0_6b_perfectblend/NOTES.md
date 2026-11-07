# DSpark drafter for Qwen3-0.6B on Open-PerfectBlend

Script: `genai/run.sh`. The same pipeline as `../dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen/` (read its
`NOTES.md` for the step list and the learnings that apply to both), with `mlabonne/open-perfectblend`
as the source dataset (`REGEN_LIMIT=100000` of ~1.4M conversations, `MAX_GEN_TOKENS=6144`,
`SEQ_LENGTH=8192`). Settings follow the official `dspark_qwen3_0_6b_sharegpt_online` example,
plus a targetgen regeneration step. On genai: 4 GPUs for regeneration, then 2 (hidden-state server)
+ 2 (training).

Reference result (earlier genai run, from the nemotron notes): trained in about 35 minutes on two
GPUs and reached a validation accept length of 3.99.

On Vista (set up 2026-10-29, nothing run yet): `settings.sh` (shared settings, genai/run.sh's values: `MAX_GEN_TOKENS=6144`, `REGEN_MAX_MODEL_LEN=12288`, `SEQ_LENGTH=8192`,
`MAX_ANCHORS=3072`) and `vista/` (`run.sh`, `run.sbatch`, `submit_chain.sh`), copied from the nemotron-terminal-corpus experiment's Vista scripts with two differences:
no step-0 export (the `open-perfectblend` preset is read directly, as genai does), and no raise of the training length to 16384 (perfectblend rows are short; the nemotron experiment's
rows are long). Stages: `regen`, `prepare`, `train`; `bash vista/submit_chain.sh` submits `regen` then `prepare,train` (2 gb jobs, 12 h each, afterok).
Work dir `runs/perfectblend_qwen3_0_6b_regen_online_think_mooncake_100k`, W&B run name = the folder name, HF upload on by default for full runs (public repo, name = folder + date).
Smoke test on a gb idev: `REGEN_LIMIT=300 MAX_STEPS=10 WORK_DIR=$PROJECT_ROOT/runs/vista_test_perfectblend bash vista/run.sh`.
Untested on Vista: whether regeneration with `--limit` takes the first rows or a random sample is a property of the speculators preset (genai behaviour kept).
