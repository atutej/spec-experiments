# DSpark drafter for Dr. Doom: itw + targetgen mixed into ONE dataset (about 12 h aggregate, about 7.5 h of wall time)

Parent (drafter, recipe, 4-rank layout, seed): [`../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/) (3 Qwen3 layers, FFN 3840, 32k draft vocabulary, random init, linear LR decay over the epoch, one epoch).
The apples-to-apples counterpart at 12 h aggregate of itw-80k (12 h of itw) and `..._targetgen-210k` (210k regenerated rows). Not two phases: one shuffled dataset, trained once.

## Data (`data/build_mix.py`, run by the train job on its batch host, `vista/run_mix.sh`)
- **itw part:** random whole Datakit shards (seed 0; each Datakit shard is a random sample of the mixture) until 4.85B tokens, about 23 of 214 shards, about 700k rows, with their ORIGINAL replies. 4.85B tokens is 6 h of the itw run (it saw 120k tokens per step at 0.523 s per step: 40.4k steps). The shards are symlinked, not copied. The last Datakit shard (the held-out rows the eval prompts come from) is never used for training.
- **targetgen part:** 105,000 rows of the warm80k run's regenerated parts (`runs/grug_dspark_drdoom_targetgen_warm80k/regen_parts`: 250k rows of Dr. Doom's own /think replies), random part order, skipping rows whose source Datakit row is in a chosen itw shard (no prompt appears twice). Converted to the Datakit layout (spans become the 0/1 `loss_mask`, int64, no extra columns). 105k rows is 6 h of the 400k run's cost (regeneration 0.0429 h + training 0.0142 h per 1k rows): about 4.5 h of regeneration, reused, not rerun.
- **validation:** the mix's last shard: 3,000 rows of the Datakit held-out shard + 500 held-out regenerated rows. `manifest.json["train_data_ratio"]` makes the trainer's validation split exactly that shard.
- Vocabulary: the Datakit dataset's own `token_freq.pt`, so the draft vocabulary is the one of itw-80k and warm80k.
- The on-disk order does not matter for training (the trainer's sampler permutes the rows); it matters only for the validation split, which is why validation is the last shard.

## Expected size and time (estimates)
Mean row lengths: itw about 7.1k tokens (2-shard sample), regenerated about 10.4k tokens (300-row sample). 4.85B + 105k x 10.4k = about 5.9B tokens, about 49k steps at 120k tokens per step, about 7.2 h at 0.523 s per step, plus the mix build (about 10-20 min) and startup. One 12 h job.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-targetgen-mix
bash vista/submit_chain.sh --dry-run ; bash vista/submit_chain.sh        # one job; JOBS=2 adds a resumable follow-up
```
Work dir `runs/grug_dspark_itw_targetgen_mix_drdoom`; W&B run name = this folder's name. A small test of `build_mix.py` (2 shards, 300 regenerated rows) passed on 2026-10-31; the full-size build has not been run.
