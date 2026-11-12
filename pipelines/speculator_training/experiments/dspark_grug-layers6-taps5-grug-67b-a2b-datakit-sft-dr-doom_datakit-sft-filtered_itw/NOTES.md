# itw DSpark drafter for Dr. Doom: 6 Grug layers, five taps, block 8, ONE EPOCH of the Datakit SFT mixture WITHOUT the finance_v1 rows

Written 2026-11-10. Counterpart: [`../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/) (the same recipe on the full dataset). Question: does removing the long finance rows (nemotron_sft_v3/finance_v1/train, 326,695 rows, 9.41B tokens, 20.5% of Datakit's tokens, about 30k tokens per row with short replies) change the drafter's acceptance length on the neutral benchmarks and on a Datakit eval that represents the long rows?
Motive: the 24 h BigMix targetgen drafter beats the Datakit-trained drafters on short-prompt benchmarks, and Datakit's context is dominated by long finance filings. See the eval set `runs/val_prompts_uncapped_512_20261110` (`datakit_full` benchmark) and `runs/finance_val_prompts_20261111` (`finance`).

## The filtered dataset
`/scratch/09749/atutej/marin_datakit/dataset_seq32768_nofinance` (585 GB), built by `data/build_filtered_dataset.py` from the scan `data/scan_finance.py` (the finance rows are recognised by the task's opening instruction in the user turn: 326,695 matched = the manifest's finance_v1 rows_kept, exactly):
- 6,144,464 rows (was 6,471,159), 36.58B tokens (was 45.99B), 23.91B supervised (was 24.33B); same shard names, same row order, same schema; the source dataset is only read.
- Build checks (the script stops if one fails): rows removed == manifest finance rows; tokens removed == the leaf's tokens_kept (9.41B); supervised tokens kept == total supervised - the leaf's. `finalize_dataset.py` wrote token_freq.pt (126,800 distinct supervised tokens) and loaded the directory as the trainer does.
- The trainer's validation split is the LAST 0.1% of the rows by position. The old validation rows without finance stay at the end (6,123 of the new 6,145), so the held-out eval prompts, all taken from the old validation rows, remain held out.
- The 32,000-token draft vocabulary is built from THIS dataset's token_freq.pt, so it differs a little from the full-dataset run's.

## Schedule
One epoch of the filtered dataset: **300,412 steps** (derived from the scheduler's learning rate in the checkpoints; the same derivation gives 382,745 for the full-dataset run, whose true length is 382,747), against 382,747 for the full dataset. About 79.5k supervised tokens per step (the full run: 63.5k, because the finance rows have short replies). The measured speed is 1.0 to 1.15 s per step, so one epoch is about 90 to 96 h. The learning rate (peak 3e-4, 1% warmup) decays linearly over THIS epoch, so it falls faster per step than in the full-dataset run. For the full run's rate per step (and an unannealed end), add `--scheduler-total-steps 382747` to TRAIN_EXTRA. Checkpoints every 0.13% of the epoch (about 400 steps).

## Run it (from a LOGIN node)
```
bash vista/submit_chain.sh --dry-run
JOBS=2 bash vista/submit_chain.sh        # two 12 h jobs (comparison points ~12 h and ~24 h); AFTER=<id> JOBS=n continues
```
Work dir `runs/grug_dspark_itw_drdoom_grug-layers6-taps5-nofinance` (override `GRUG_L6T5_NOFIN_WORK_DIR`). Evaluate with `tools/eval_acceptance/eval_matrix.sh` (`label=<checkpoint dir>@dspark@8+overlay`, benchmarks `speedbench datakit datakit_full finance bigmix paper`).

## Run history (2026-11-10 to 2026-11-12)
- Job 1136569 (6 h): to step 18,169; newest complete checkpoint 17,986 (the 6 h point), kept in `milestones/`. A Vista maintenance window (2026-11-11 06:15 to 22:00) held the next jobs, so the chain was split into 6 h chunks.
- Job 1137652 (12 h): from 17,986 to about step 46k at 1.15 s per step; paused by hand at the checkpoint of step 45,747 (about 15 h of training in all, 0.15 epoch), kept in `milestones/` (the checkpoint of step 45,356 too).
- Job 1138548 (9 h): resumed from 45,747 on the same schedule (no override), after the evals of that checkpoint.
- No `epoch0_end` exists yet: the learning rate has not been annealed (about 0.85 of the peak at step 45.7k).
- Evaluations (AL, tokens/s) of the 6 h and 45,747 checkpoints on SPEED-Bench, the Datakit held-out set, a Datakit set that represents the long rows (`datakit_full`), `finance_v1`, the BigMix held-out set and GSM8K / MATH-500 / HumanEval / MBPP / MT-Bench are summarised in `docs/results/dspark_eval/drafter_row_stats_vs_al_20261112/` (`al_by_benchmark.csv`): on the six neutral benchmarks the 45.7k checkpoint is level with the full-dataset run at 75.8k steps (3.379 against 3.380 mean AL) with 60% of the steps and 25% fewer supervised tokens seen; `finance_v1` AL falls to about 3.36 (the full run at 24 h: 4.19), as expected without those rows.
