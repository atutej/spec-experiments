## Results (target: open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom)

Only the Dr. Doom target was evaluated. [`Dr. Doom`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) is a de-looping fine-tune of the 09.21 target this drafter was trained against, so this is transfer to a fine-tuned target. Serving in the pinned Marin vLLM fork on one 4-GPU GB200 node (TP=1, DP=4, expert parallel), concurrency 16, T=1, 8 drafted tokens, probabilistic drafting with standard verification, one run per arm, all three arms in the same session (the two nodes of the reservation ran at the same time). Prompts: the first 256 of 1,024 random train rows of the RL prompt set, thinking on, up to 49,152 new tokens (102,400-token window).

| Arm | Acceptance length | Output tokens/s | Gain vs plain | Acceptance at positions 0, 1, 2, 3 |
|---|---|---|---|---|
| Plain decoding | n/a | 2,006 | 1.00x | |
| itw drafter, step 80,813 (09.21 hidden states, [`atutej/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw-20261016`](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw-20261016)) | 2.657 | 3,601 | 1.79x | 0.671, 0.416, 0.247, 0.143 |
| **This drafter (rlsimilar, no dedup, 09.21 hidden states)** | **2.780** | **3,772** | **1.88x** | 0.695, 0.445, 0.271, 0.161 |

This drafter is ahead of the itw drafter by 0.123 in acceptance length (+4.6%) and 4.7% in throughput; a repeat of one configuration differs by about 0.02. It is ahead at every drafted position. What the comparison does not isolate:

- **Several things differ from the itw drafter, not only the data.** The itw checkpoint is a mid-run one (step 80,813 of a 382,747-step epoch, stopped by a time limit before the learning rate decayed), while this one is the annealed end of its own epoch. Its draft vocabulary is built from its own filtered data. And its data repeats rows (70%).
- **The evaluation prompts are the kind of text the data was selected for.** The 256 prompts are rows of the RL train split that was the query set of the selection (all 84,403 rows), so the evaluation is in-distribution by construction. No other prompt set and no 09.21 target run was evaluated.
- One run on 256 prompts per arm.

The trainer's own teacher-forced validation of this checkpoint (on the held-out 0.1% of its dataset, whose rows can repeat training rows) is listed under Validation above; serving acceptance is lower than that metric, as for the other drafters.

**Prompts and evaluation.** Built as for the other drafters by [`build_nemotron_rlvr_prompts.py`](https://github.com/atutej/spec-experiments/blob/452b6f5e96ede242941a362a14de355c095626a4/pipelines/speculator_training/tools/eval_acceptance/build_nemotron_rlvr_prompts.py) (`--num 1024 --seed 0`), served by [`serve_eval.sh`](https://github.com/atutej/spec-experiments/blob/452b6f5e96ede242941a362a14de355c095626a4/pipelines/speculator_training/tools/eval_acceptance/serve_eval.sh):

```
MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom PROMPTS_DIR=<prompts> MAX_PROMPTS=256 MAX_TOKENS=49152 MAX_MODEL_LEN=102400 OUT=<out> PLAIN=1 \
  ARMS="drafter=<drafter>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
```

Raw results of the three arms: [`rl_prompts_256_target-dr-doom/`](https://github.com/atutej/spec-experiments/tree/452b6f5e96ede242941a362a14de355c095626a4/docs/results/dspark_eval/rl_prompts_256_target-dr-doom) (`rlsimilar_nodedup_trained09.21/`, `itw_80k_rerun/`, `plain_rerun/`).
