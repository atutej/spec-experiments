# DSpark drafter for Qwen3-0.6B on Open-PerfectBlend

Script: `genai/run.sh`. The same pipeline as `../dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy/` (read its
`NOTES.md` for the step list and the learnings that apply to both), with `mlabonne/open-perfectblend`
as the source dataset (`REGEN_LIMIT=100000` of ~1.4M conversations, `MAX_GEN_TOKENS=6144`,
`SEQ_LENGTH=8192`). Settings follow the official `dspark_qwen3_0_6b_sharegpt_online` example,
plus an on-policy regeneration step. On genai: 4 GPUs for regeneration, then 2 (hidden-state server)
+ 2 (training).

Reference result (earlier genai run, from the nemotron notes): trained in about 35 minutes on two
GPUs and reached a validation accept length of 3.99.

Not set up on Vista.
