# itw DSpark drafter for Dr. Doom on the big dataset, about 11.7B tokens (92,000 steps, about 11 h), learning rate annealed over 46B tokens

Sibling of [`../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw/) (the 2-epoch run, job 1111721; its dataset builders and data notes live there).
Same data (`/scratch/09749/atutej/marin_big_pool/dataset_itw_seq32768`, read only), drafter (3 Qwen3 layers, FFN 3840, 280M), recipe, seed, 4-rank layout, `--num-workers 48 --prefetch-factor 2` and validation split (0.5% of the rows).
(Renamed from `..._9p7b-lr46b` before any cluster run: the budget was raised from 76,200 to 92,000 steps.)

## Why
The 2-epoch run saw 8.7B total tokens (about 5.3B supervised) in 7.9 h. The Datakit itw-80k run saw 9.7B total tokens (about 5.1B supervised; 80,813 of the 382,747 steps of one epoch) with a full-epoch schedule (about 46B tokens) that its 12 h limit stopped, so its rate was still about 0.8 of the peak.
This run uses the same schedule shape (**horizon about 46B tokens**) and stops after **92,000 steps = about 11.7B total tokens (about 7.1B supervised)**, which fills about 11 of the 12 h of a job. It therefore sees about 20% more total tokens than the Datakit original, not the same number.

## Settings that differ from the 2-epoch run
| | 2-epoch run | this run |
|---|---|---|
| steps | 68,334 (EPOCHS=2) | `--max-steps 92000` (about 11.7B tokens at 127.3k tokens per step) |
| schedule | linear, 1% warmup, zero at step 68,334 | `--scheduler-total-steps 361000` (about 46B tokens): 3,610 warmup steps, then linear to zero at step 361,000; at step 92,000 the rate is 0.753 of the peak (2.26e-4) |
| EPOCHS | 2 | 3 (34,167 steps per epoch, so the loop ends inside epoch 3; `--max-steps` stops it) |
| work dir | `runs/grug_dspark_itw_drdoom_big_pool` | `runs/grug_dspark_itw_drdoom_big_pool_11p7b` (new) |
| final checkpoint | `epoch1_end` | `epoch2_end` (the trainer saves it when `--max-steps` ends the loop; the chain's guard looks for it) |
End-of-epoch checkpoints `0/` (= `epoch0_end`) and `1/` are never pruned; a step checkpoint every 0.03 epoch (about 1,025 steps), newest 3 kept.

## Time (estimates)
0.41 s per step in the 2-epoch run (3 h 55 min per epoch): 92,000 steps are about 10.5 h, plus about 0.4 h for startup (server, the first dataloader step with 48 workers) and three validation passes: about 11 h in one 12 h job, so about 1 h of margin. A second chained job exits at once if `epoch2_end` exists. Not measured at this length: step time varied between 0.39 and 0.41 s in earlier runs.
Smoke test (2026-11-02, one node, 40 steps, horizon 400, EPOCHS=3): learning rate at step 40 was 2.73e-4 as expected, `epoch0_end`, `epoch1_end` and `epoch2_end` were saved, clean exit (`SMOKE_MAX_STEPS` / `SMOKE_SCHED_STEPS` override the two numbers for such a test).

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw-11p7b-lr46b
bash vista/submit_chain.sh --dry-run
JOBS=1 bash vista/submit_chain.sh
```
The tokens-per-step figure is from the 2-epoch run (4.35B training tokens over 34,167 steps); the exact horizon only moves the final learning rate by a fraction of a percent.
