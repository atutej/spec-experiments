# Rollout acceptance length

The trainer's `eal` is teacher-forced on dataset text, which is biased for a drafter trained on the dataset's own
completions. This measures acceptance on what the target model actually generates: serve the target with the draft in
vLLM, send held-out prompts, read vLLM's speculative-decoding counters (definitions from speculators'
`scripts/evaluate/perf_utils.py`; `acceptance_length = 1 + accepted/drafts`).

1. `build_prompts.py` (once, CPU): 500 conversations not in the 100k training sample -> `first_turn.jsonl` (first assistant
   turn) and `later_turn.jsonl` (one random later turn, context teacher-forced), rendered with the Qwen3 chat template.
   Output `runs/eval_prompts_500/` with a manifest (seed, sha256, overlap check = 0).
2. `run_eval.sh CHECKPOINT [OUT]` (one GPU): serves Qwen3-0.6B + the draft, runs `run_eval.py` (temp 0.6, top_p 0.95,
   top_k 20, max_tokens 8192, per-request seeds), writes `acceptance.csv` (first_turn, later_turn, pooled `all`),
   `results.json`, `completions_*.jsonl`; `WANDB_NAME` logs it to W&B (entity atutej, project marin_speculator).
3. Per experiment: `experiments/<name>/vista/{eval.sh,eval.sbatch,submit_eval.sh}` run it on that experiment's checkpoint.

Both drafters must be evaluated on the same prompt files and settings. Checkpoint `0/` is the end-of-training checkpoint of the
one-epoch runs (`epoch0_end` and, if validation ran, `checkpoint_best` link to it).
