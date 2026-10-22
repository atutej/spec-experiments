## Results (target: [open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom))

Four drafters for the Dr. Doom target. [`Dr. Doom`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) is a de-looping fine-tune of the 09.21 Snowball SFT model. Serving in the pinned [Marin vLLM fork](https://github.com/marin-community/vllm/releases/tag/marin-vllm-gpu-20260929-39e62869693c) on one 4-GPU GB200 node (TP=1, DP=4, expert parallel), concurrency 16, T=1, 8 drafted tokens, probabilistic drafting with standard verification. Acceptance length is `1 + accepted/drafts` from vLLM's counters over whole chained rollouts.

| Prompts | Drafter | Acceptance length | Speedup |
|---|---|---|---|
| **RLVR** | **targetgen from scratch (this model)** | **2.758** | **1.87x** |
| | [itw, step 80,813](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-80k-20261020) | 2.689 | 1.77x |
| | [itw, step 160,860](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-160k-20261020) | 2.785 | 1.87x |
| | [targetgen, warm start from the 80k checkpoint](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k-20261020) | 2.851 | 1.90x |
| **Datakit first turn** | **targetgen from scratch (this model)** | **2.909** | **1.95x** |
| | [itw, step 80,813](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-80k-20261020) | 2.983 | 1.96x |
| | [itw, step 160,860](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-160k-20261020) | 3.047 | 1.99x |
| | [targetgen, warm start from the 80k checkpoint](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k-20261020) | 3.091 | 2.02x |
| **Datakit later turn** | **targetgen from scratch (this model)** | **2.398** | **1.44x** |
| | [itw, step 80,813](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-80k-20261020) | 2.522 | 1.52x |
| | [itw, step 160,860](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-160k-20261020) | 2.587 | 1.72x |
| | [targetgen, warm start from the 80k checkpoint](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k-20261020) | 2.595 | 1.57x |

Speedup is output tokens/s divided by plain decoding on the same prompts: 2,006 tokens/s on the RLVR prompts, 2,142 on the 833 Datakit first-turn prompts and 1,624 on the 191 later-turn prompts. These references come from plain-decoding runs of earlier sessions, and this model's runs had no plain arm of their own. Each number is one run, and the arms were not all measured in one session: the three other drafters come from the earlier comparison (the step-80,813 arm in an earlier session than the other two), this model from a separate later session. Plain throughput varies between sessions, so differences of a few hundredths in acceptance length and a few percent in speedup are within noise. The first RLVR run of this model was slow (acceptance length 2.734, 2,598 tokens/s, 1.29x) while a repeat on the same node gave the 2.758 and 3,744 tokens/s (1.87x) shown above; both runs are in the raw results.

**Prompts.** RLVR: the first 256 of 1,024 prompts sampled at random (seed 0) from the train split of [`open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30`](https://huggingface.co/datasets/open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30) (the Snowball broad-domain RL set), thinking on, replies up to 49,152 tokens (102,400-token window). Datakit: 1,024 prompts from the held-out validation rows of the Datakit SFT dataset (the last 0.1% of its rows, which none of the four drafters trained on): a uniformly random assistant turn per row with the earlier turns as context, 833 first-turn and 191 later-turn prompts, replies up to 4,096 tokens.

**Evaluation.** [`serve_eval.sh`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance/serve_eval.sh) starts one vLLM server per arm and [`run_eval.py`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance/run_eval.py) sends the prompts.

```
MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom PROMPTS_DIR=<RLVR prompts> MAX_PROMPTS=256 MAX_TOKENS=49152 MAX_MODEL_LEN=102400 OUT=<out> \
  ARMS="a=<drafter>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom PROMPTS_DIR=<Datakit prompts> SUBSETS='first_turn later_turn' MAX_TOKENS=4096 MAX_MODEL_LEN=32768 OUT=<out> \
  ARMS="a=<drafter>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
```

Raw results of this model: [`drdoom_targetgen_scratch/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/docs/results/dspark_eval/drdoom_targetgen_scratch); of the other three drafters: [`drdoom_itw_vs_warm80k/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/docs/results/dspark_eval/drdoom_itw_vs_warm80k).
