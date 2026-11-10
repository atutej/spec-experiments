# targetgen DSpark drafter for Dr. Doom, 6 Grug layers, trained on Datakit-SFT conversations with Dr. Doom's own replies, about 24 h in all

Question: the targetgen counterpart of `../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw` (original completions). Written 2026-11-09.
Data: 281,000 rows: a random 281 of each shard's 400 parts (500 rows each) of the 400,000-row regeneration in `runs/grug_dspark_drdoom_targetgen/regen_parts` (Dr. Doom replies, `/think`, T=1; 17.2 h of regeneration), chosen by `data/select_parts.py` (seed 0, the list in `runs/grug_dspark_drdoom_grug6t5_datakit_targetgen281k/subset_parts.json`). Random parts, not a prefix: the rows that finish first are shorter. The source parts are only read. Nothing is regenerated here.

## Learning-rate schedule (identical to the original-completion 6-layer runs)
`--scheduler-total-steps 382747`, peak 3e-4, warmup 1% (3,827 steps), linear decay: the same learning rate at every step as the Datakit-original and BigMix-original runs. This run covers ONE epoch of its rows, which is far fewer steps than 382,747, so the rate ends near 0.94-0.95 of the peak (not annealed), just as the 12 h and 24 h original-completion checkpoints are not annealed. `EPOCHS=1`, no `--max-steps`.

## What is the same as the original-completion 6-layer runs
`draft_config.json` (6 Grug layers, FFN 7,936), taps 2/7/12/17/22 (+ final), block size 8, Markov rank 1024, 32,000-token draft vocabulary (here built from the regenerated text by the merge), `--draft-layer-type grug --grug-qk-mult 1.75`, 12,288 anchors, 32,768-token rows, loss, seed, peak learning rate. Differences: trained from the target's own `/think` replies; 0.5% validation split (targetgen convention); a checkpoint every 2% of the epoch.

## Time budget (about 24 h in all, merge excluded)
Regeneration 17.2 h (already spent, counted in full) + training about 2.7B tokens (about 22.5k steps at about 1.1 s) = about 6.8 h; total about 24 h. Time limit 07:10:00.
The job limit covers the training plus about 5 minutes of gather and merge. If the regeneration took longer or shorter than expected, change `TIME=hh:mm:ss` when submitting. A job cut by its time limit loses at most the last 2% of the epoch (checkpoints every 2%, learning rate unannealed), and can be resumed with `JOBS=1 bash vista/submit_chain.sh` (resumes from the newest checkpoint).

## Run it (from a LOGIN node)
```
bash vista/submit_chain.sh --dry-run
AFTER=<BigMix targetgen train job id> DEP=afterany bash vista/submit_chain.sh     # queued behind the BigMix job
```
Work dir `runs/grug_dspark_drdoom_grug6t5_datakit_targetgen281k`. Evaluate with `tools/eval_acceptance/eval_matrix.sh` (`label=<checkpoint dir>@dspark@8+overlay`; copy `checkpoints/epoch0_end` or `checkpoint_best` to `runs/eval_ckpts/` first).
