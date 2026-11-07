> Archived 2026-10-31 from the experiment folder `dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend_itw` (deleted; its builders now live in `../data/`). Relative links below point at the old location: parent folders are siblings of `dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw` in `experiments/`; run outputs were in `runs/grug_dspark_itw_drdoom_perfectblend` (deleted 2026-10-31, eval results kept in `runs/eval_pblend_step28284_2026-10-30/`).

# itw DSpark drafter for Dr. Doom on Open-PerfectBlend (first run: rows WITHOUT a system header)

Parent: [`../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/) (3 Qwen3 layers, FFN 3840, 280M parameters, 32k draft vocabulary). Same drafter, recipe, 4-rank layout and seed; the data is
`mlabonne/open-perfectblend` (1,420,909 conversations, human/gpt turns from MetaMathQA, UltraInteract, UltraChat, Orca-math, UltraFeedback, Evol-codealpaca, AutoIF, LMSYS) instead of the Datakit mix, 10 epochs planned.
**The redo with the correct header is [`../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nothink_itw`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nothink_itw/); this run's checkpoints are kept as they are and are not continued.**

## Data (`data/build_perfectblend.py`, `dataset_seq32768`)
Rendered with `MARIN_CHAT_TEMPLATE` (human -> user, gpt -> assistant), tokenized and masked with the Datakit code (`_write_job`, `_write_shard`): 1,358,785 rows kept of 1,420,909 (62,124 dropped because the turns do not start with a user turn, alternate, and end with an assistant turn),
0.687B tokens, 0.491B supervised, nothing clipped. The mask check against the template's generation spans passed on 278 of 278 rows.
**Mistake: no system header.** The template writes `<|start_header_id|>system<|end_header_id|>Reasoning: /think|/nothink<|eot_id|>` only when a reasoning mode is passed, and this build passed none. The Datakit rows carry `/nothink` in 83%, `/think` in 16%, nothing in 1% (sample of 4,146 rows); every eval prompt uses `/think`.
PerfectBlend's replies are direct answers (no `<|start_think|>` block in 67,941 sampled rows), i.e. what `/nothink` rows look like in Datakit (0% of those have a reasoning block; 100% of the `/think` rows do).

## Run (job 1109877, 2 gb nodes, 12 h limit, work dir `runs/grug_dspark_itw_drdoom_perfectblend`)
Stopped by the time limit at step 28,744 (5.3 epochs of about 5.3k steps; the end-of-epoch checkpoints `0/` .. `4/` and `checkpoint-28284` exist). Held-out (0.5%) trainer validation, expected acceptance length: epoch 1 3.93, 2 4.22, 3 4.35, 4 4.43, 5 4.49 (still rising; PerfectBlend text, not comparable with serving numbers).
**Step time: mean 1.48 s against a median of 0.29 s.** 13% of the steps stalled 12-16 s each and held 83% of the time; the gap between stalls was 12 steps most often, the dataloader's `num_workers`. A step is about 65 short rows, each fetched by its own hidden-state request in one worker (about 0.27 s per request, about 15 s per batch),
so 12 workers deliver bursts of 12 batches and then wait. The Datakit runs are compute-bound (fetch 2% of the step, 0.3% of the steps stall) and do not need more workers. The fix used in later PerfectBlend settings: `--num-workers 48 --prefetch-factor 2` (not tested at the time of writing).

## Evaluation (checkpoint-28284 on the Dr. Doom target, T=1, probabilistic drafting, concurrency 16, one run, no plain arm)
| | PerfectBlend itw (no header) | Datakit itw-80k (same day) |
|---|---|---|
| RLVR (256 prompts) | 1.655 | 2.723 |
| Datakit first turn (833) | 2.227 | 2.991 |
| Datakit later turn (191) | 1.923 | 2.525 |
| Datakit pooled (1024) | 2.196 | 2.939 |

Two causes are mixed in the gap: the PerfectBlend prompts and replies are both off-policy (written by other models, short chat and math), and the rows lack the system header every eval prompt has. They cannot be separated from this run.
Raw results: `runs/eval_pblend_step28284_2026-10-30/`.
