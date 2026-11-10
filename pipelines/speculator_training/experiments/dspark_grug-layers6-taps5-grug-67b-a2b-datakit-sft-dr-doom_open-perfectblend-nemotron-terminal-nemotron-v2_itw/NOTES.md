# itw DSpark drafter for Dr. Doom: 6 Grug layers, taps 2/7/12/17/22, block 8, Markov rank 1024, trained on the BIG POOL (Datakit's full 382,747-step schedule; 12 h + 12 h first)

Question: with the same drafter, does the big pool (Open-PerfectBlend + Nemotron-Terminal-Corpus + Nemotron-Post-Training-Dataset-v2, existing text, no regeneration) beat the Datakit SFT mixture (existing text) after 12 h and 24 h? Written 2026-11-08. Counterpart: [`../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/) (evaluated at steps 37,917 and 75,834).

## What is identical to the Datakit run
Architecture (`draft_config.json` is a copy: 6 layers, FFN 7,936), taps, block size 8, Markov rank 1024 (vanilla), 32,000-token draft vocabulary, `--draft-layer-type grug --grug-qk-mult 1.75`, 12,288 anchors, 32,768-token rows, loss, seed, peak learning rate 3e-4.

## Learning-rate schedule (the point of this folder)
The Datakit run is one epoch of 382,747 steps (1% warmup = 3,827 steps, linear decay to zero at step 382,747) and will be run to completion. This run follows the SAME schedule to the end:
- `--scheduler-total-steps 382747`: same horizon and warmup, so at any step both runs have the same learning rate (at step 75,834: 0.81 of the peak).
- `--max-steps 382747` and `EPOCHS=12`: the big pool has 34,167 steps per epoch, so the same 382,747 steps are 11.2 epochs; the loop ends inside epoch 12, and the chain's final-checkpoint guard is `epoch11_end`.
- **Pausing is free.** The 12 h and 24 h comparison points (steps near 37,917 and 75,834) are just checkpoints on the way; the learning rate is not annealed there, and the next 12 h job resumes from the newest checkpoint. Run `JOBS=2` now, and add jobs later with `AFTER=<last id>`.
- `CKPT_FREQ=0.01121`: the trainer saves every round(34,167 x 0.01121) = 383 steps, on the Datakit run's grid (37,917 = 99 x 383, 75,834 = 198 x 383). A job boundary lands within 383 steps of those; the milestone copy made at the start of each job keeps the checkpoint at that boundary.

## What differs (read before comparing)
| | Datakit run | this run |
|---|---|---|
| data | Datakit SFT mixture, 6.5M rows (one epoch = 382,747 steps) | big pool, 2.93M rows, 4.37B tokens (34,167 steps per epoch) |
| epochs seen at 75,834 steps | 0.20 (no repeats) | **2.22 (repeated data)**; 11.2 epochs at the end |
| tokens per step | about 120k | about 128k |
| draft vocabulary | top 32,000 of the Datakit `token_freq` | top 32,000 of the big pool's `token_freq` (a different selection) |
| validation split | 0.1% | 0.5% |
| data workers | default | `--num-workers 48 --prefetch-factor 2` (from the big-pool settings) |
Each drafter has a home-set advantage on its own held-out set; SPEED-Bench and the five paper-style sets (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench) are the neutral comparison.

## Run it (from a LOGIN node)
```
cd .../experiments/dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw
bash vista/submit_chain.sh --dry-run
JOBS=2 bash vista/submit_chain.sh                       # 12 h + 12 h: the comparison point
AFTER=<last job id> JOBS=2 bash vista/submit_chain.sh   # later: continue along the same schedule (the whole run is about 117 h at 1.1 s per step, about 10 jobs)
```
Work dir `runs/grug_dspark_itw_drdoom_grug-layers6-taps5-bigmix` (override `GRUG_L6T5_BIGMIX_WORK_DIR`). Each job exits at once when `checkpoints/epoch11_end` exists.

## Evaluate
After job 1 and after job 2: the newest complete step checkpoint (copy it, as for the Datakit run; the milestone copy at the start of the next job also keeps it). Then `tools/eval_acceptance/eval_matrix.sh` with `BENCHES="speedbench datakit bigmix paper"` and `label=<dir>@dspark@8+overlay`.

## Risks
- Step time on the big pool is unmeasured for this drafter (about 1.1 s expected from the Datakit run; the earlier 3-layer big-pool run took 0.41 s per step). A slower step moves the 24 h checkpoint a little earlier than step 75,834.
- The 48 loader workers and 0.5% validation passes come from the big-pool settings and were last used with 3-layer drafters.
