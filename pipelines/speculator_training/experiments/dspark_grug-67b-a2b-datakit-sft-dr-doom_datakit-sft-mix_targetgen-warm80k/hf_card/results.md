## Results (target: [open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom))

Three drafters for the Dr. Doom target, side by side. [`Dr. Doom`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) is a de-looping fine-tune of the 09.21 Snowball SFT model. Serving in the pinned [Marin vLLM fork](https://github.com/marin-community/vllm/releases/tag/marin-vllm-gpu-20260929-39e62869693c) on one 4-GPU GB200 node (TP=1, DP=4, expert parallel), concurrency 16, T=1, 8 drafted tokens, probabilistic drafting with standard verification; one run per arm. Acceptance length is `1 + accepted/drafts` from vLLM's counters over whole chained rollouts.

| | [itw, step 80,813](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-80k-20261020) | [itw, step 160,860](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-160k-20261020) | [**targetgen, warm start from the 80k checkpoint**](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k-20261020) |
|---|---|---|---|
| **RLVR prompts**: acceptance length | 2.689 | 2.785 | **2.851** |
| RLVR: output tokens/s (speedup vs plain 2,006) | 3,557 (1.77x) | 3,746 (1.87x) | 3,811 (1.90x) |
| **Datakit held-out prompts**, pooled: acceptance length | 2.932 | 2.995 | **3.038** |
| Datakit pooled: output tokens/s (speedup vs plain 2,081) | 3,934 (1.89x) | 4,049 (1.95x) | 4,069 (1.96x) |
| Datakit first turn / later turn: acceptance length | 2.983 / 2.522 | 3.047 / 2.587 | 3.091 / 2.595 |
| Training tokens (approx.) | 9.7B itw text | 19.3B itw text | 9.7B itw text + 2.4B regenerated |

**Prompts.** RLVR: the first 256 of 1,024 prompts sampled at random (seed 0) from the train split of [`open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30`](https://huggingface.co/datasets/open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30) (the Snowball broad-domain RL set), thinking on, replies up to 49,152 tokens (102,400-token window). Datakit: 1,024 prompts from the held-out validation rows of the Datakit SFT dataset (the last 0.1% of its rows, which none of the three drafters trained on): a uniformly random assistant turn per row with the earlier turns as context, 833 first-turn and 191 later-turn prompts, replies up to 4,096 tokens. None of these prompts are in the fine-tuning data of the targetgen drafter.

For reference, the 09.21-trained drafters ([itw](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw-20261016), [targetgen](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen-20261016)) measured 2.667 and 2.465 acceptance length (1.79x and 1.62x) on the same RLVR prompts with the Dr. Doom target in an earlier session; they were trained against the 09.21 model's hidden states and are not part of this comparison.

**Evaluation.** [`serve_eval.sh`](https://github.com/atutej/spec-experiments/blob/GH_REF_PLACEHOLDER/pipelines/speculator_training/tools/eval_acceptance/serve_eval.sh) starts one vLLM server per arm and [`run_eval.py`](https://github.com/atutej/spec-experiments/blob/GH_REF_PLACEHOLDER/pipelines/speculator_training/tools/eval_acceptance/run_eval.py) sends the prompts; plain decoding is the no-draft arm (`PLAIN=1`).

```
MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom PROMPTS_DIR=<RLVR prompts> MAX_PROMPTS=256 MAX_TOKENS=49152 MAX_MODEL_LEN=102400 OUT=<out> PLAIN=1 \
  ARMS="a=<drafter> b=<drafter>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom PROMPTS_DIR=<Datakit prompts> SUBSETS='first_turn later_turn' MAX_TOKENS=4096 MAX_MODEL_LEN=32768 OUT=<out> PLAIN=1 \
  ARMS="a=<drafter> b=<drafter>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
```

Raw results of all arms and sessions: [`drdoom_itw_vs_warm80k/`](https://github.com/atutej/spec-experiments/tree/GH_REF_PLACEHOLDER/docs/results/dspark_eval/drdoom_itw_vs_warm80k).
