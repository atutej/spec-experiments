# targetgen-only DSpark drafter for Dr. Doom, from random initialisation

The target is `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom` for the rollouts, the hidden states and the lm_head, **not** 09.21. The drafter (3 layers, FFN 3840, block 8, 32k draft vocabulary) starts from random weights and sees only Dr. Doom's own replies: 400k Datakit rows (uniform sample, seed 0), each conversation replayed through Dr. Doom with the dataset's prompts. No itw text is mixed in.

It matches the compute of the warm-start run (itw 80k + targetgen, about 26 h): about 17 h of regeneration, about 4.6 h of training, about 2 h of overhead (estimates from the warm80k run, see `settings.sh`).

## Run it (from a LOGIN node; nothing is submitted by writing this folder)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen
bash vista/submit_chain.sh --dry-run ; bash vista/submit_chain.sh        # regen (12 h) -> regen again (afterany, resumes) -> train (afterok): 3 jobs
```
Work dir `runs/grug_dspark_drdoom_targetgen` (new; `settings.sh` refuses the old runs' dirs). W&B run name = this folder's name.

## Notes
- Regen is resumable per part; the second regen job exits quickly if the first finished everything. The train job starts with the merge (about 0.5 h), which builds the draft vocabulary from Dr. Doom's replies.
- Ran end to end as submitted (jobs 1085529 regen 12 h limit, 1085530 regen 5 h 09 min, 1085531 merge + train 5 h 40 min), no smoke test first. Regeneration: 400,000 rows, 2,467,745,475 regenerated tokens (6,169 per row), 10.2% of rows hit the 32,768-token limit, 0 errors. Training: 31,932 steps, W&B run `88vu0w3a`, final trainer validation accept length 3.625 (`checkpoints/0`, identical to `epoch0_end` and `checkpoint_best`).
- Comparison caveats: 3.8B training tokens against 9.7B for the itw 80k drafter; one epoch with the learning rate fully annealed.

## Result (serving evaluation, Dr. Doom target, T=1, 8 drafted tokens; one run per subset, no plain arm in these runs)
- RLVR (256 prompts): acceptance length 2.758 and 3,744 output tokens/s (1.87x vs the 2,006 tokens/s plain reference) in the repeat; the first run had 2.734 and 2,598 tokens/s (1.29x), a slow session.
- Datakit held-out (1,024 prompts): first turn 2.909 (1.95x vs 2,142), later turn 2.398 (1.44x vs 1,624).
- Raw results: `docs/results/dspark_eval/drdoom_targetgen_scratch/`. Comparison with the other Dr. Doom drafters: the model card in `hf_card/`.
