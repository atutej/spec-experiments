# targetgen DSpark drafter for Dr. Doom, 210k-row subset of the Datakit regeneration (about 12 h aggregate)

Parent (settings, drafter, recipe): [`../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen/) (3 Qwen3 layers, FFN 3840, 32k draft vocabulary, random init, one epoch, 4 ranks over 2 gb nodes).
Data: a random half of that run's regenerated replies: 420 of its 800 parts (210 of 400 per shard, 500 rows each) = 210,000 Dr. Doom /think replies to Datakit SFT conversations, chosen by `data/select_parts.py` (seed 0; the list is in `runs/grug_dspark_drdoom_targetgen_210k/subset_parts.json`).
The source run's parts (`runs/grug_dspark_drdoom_targetgen/regen_parts`) are only read.

## Why 210k rows
The 400k-row run took 12.0 h + 5.1 h of regeneration (jobs 1085529, 1085530) and 5.7 h of training (1085531) = 22.8 h (`sacct`). Per 1k rows that is 0.0429 h of regeneration + 0.0142 h of training = 0.0571 h; 12 h / 0.0571 h = 210k rows,
i.e. about 9.0 h of regeneration (reused, not rerun) + about 3.0 h of training (+ merge and startup, about 0.5 h). Counting regeneration, this is the Datakit counterpart of the 12 h itw runs.
Parts are random, not a prefix: a prefix would favour rows that finished first, which are shorter (in a 14-part-per-half sample the first 210 parts hold rows about 5% shorter than the last 190).

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-210k
bash vista/submit_chain.sh --dry-run ; bash vista/submit_chain.sh        # ONE train job (merge, then one epoch). Do NOT run the regen stage: it would sample and regenerate a different 210k rows.
```
Work dir `runs/grug_dspark_drdoom_targetgen_210k`; W&B run name = this folder's name.
