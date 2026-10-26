## Results (target: [open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom))

Serving in the pinned [Marin vLLM fork](https://github.com/marin-community/vllm/releases/tag/marin-vllm-gpu-20260929-39e62869693c) on one 4-GPU GB200 node (TP=1, DP=4, expert parallel), concurrency 16, T=1, 8 drafted tokens, probabilistic drafting with standard verification. Acceptance length is `1 + accepted/drafts` from vLLM's counters over whole chained rollouts. Throughput is output tokens/s of the whole eval.

| Prompts | Drafter | Acceptance length | Tokens/s | Speedup |
|---|---|---|---|---|
| **RLVR** | **3x FFN, 3 layers, step 69,323 (this model)** | **2.746** | **3,681** | **1.83x** |
| | [itw, step 80,813](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-80k-20261020) | 2.689 | 3,557 | 1.77x |
| | [itw, step 160,860](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-160k-20261020) | 2.785 | 3,746 | 1.87x |
| | plain decoding | 1.000 | 2,006 | 1.00x |
| **Datakit first turn** | **3x FFN, 3 layers, step 69,323 (this model)** | **3.043** | **4,220** | **1.97x** |
| | [itw, step 80,813](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-80k-20261020) | 2.983 | 4,193 | 1.96x |
| | [itw, step 160,860](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-160k-20261020) | 3.047 | 4,255 | 1.99x |
| | plain decoding | 1.000 | 2,142 | 1.00x |
| **Datakit later turn** | **3x FFN, 3 layers, step 69,323 (this model)** | **2.580** | **2,555** | **1.57x** |
| | [itw, step 80,813](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-80k-20261020) | 2.522 | 2,469 | 1.52x |
| | [itw, step 160,860](https://huggingface.co/atutej/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-160k-20261020) | 2.587 | 2,791 | 1.72x |
| | plain decoding | 1.000 | 1,624 | 1.00x |

At 69k steps (18% of the way to 382,747) it is above the 80k itw drafter in acceptance length (+0.06 RLVR, +0.06 Datakit first turn, +0.06 later turn) and about level with the 160k one, at less than half its steps; the wider FFN costs +19% per training step (622 ms against 523 ms) and the speed gain over the 80k drafter is within the cross-session noise.

Speedup is output tokens/s divided by plain decoding on the same prompts: 2,006 tokens/s on the RLVR prompts, 2,142 on the 833 Datakit first-turn prompts and 1,624 on the 191 later-turn prompts. These references come from plain-decoding runs of earlier sessions; this model's runs had no plain arm of their own, and the arms were not all measured in one session (the step-80,813 itw arm in the earliest, the step-160,860 arm in a second, this model in later ones). Plain throughput varies between sessions by about 15%, so speedups are estimates, and acceptance-length differences of a few hundredths are within noise. Each number is one run, 0 failed requests.

**Prompts.** RLVR: the first 256 of 1,024 prompts sampled at random (seed 0) from the train split of [`open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30`](https://huggingface.co/datasets/open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30), thinking on, replies up to 49,152 tokens (102,400-token window). Datakit: 1,024 prompts from the held-out validation rows of the Datakit SFT dataset (the last 0.1% of its rows, none of these drafters trained on them): a uniformly random assistant turn per row with the earlier turns as context, 833 first-turn and 191 later-turn prompts, replies up to 4,096 tokens.

**Evaluation.** [`serve_eval.sh`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance/serve_eval.sh) starts one vLLM server per arm and [`run_eval.py`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance/run_eval.py) sends the prompts; [`eval_after_train.sbatch`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance/eval_after_train.sbatch) ran both evals on the latest checkpoint.

```
MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom PROMPTS_DIR=<RLVR prompts> MAX_PROMPTS=256 MAX_TOKENS=49152 MAX_MODEL_LEN=102400 OUT=<out> \
  ARMS="a=<drafter>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom PROMPTS_DIR=<Datakit prompts> SUBSETS='first_turn later_turn' MAX_TOKENS=4096 MAX_MODEL_LEN=32768 OUT=<out> \
  ARMS="a=<drafter>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
```

Raw results of this model: [`drdoom_ffn3x/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/docs/results/dspark_eval/drdoom_ffn3x); of the itw drafters: [`drdoom_itw_vs_warm80k/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/docs/results/dspark_eval/drdoom_itw_vs_warm80k).
