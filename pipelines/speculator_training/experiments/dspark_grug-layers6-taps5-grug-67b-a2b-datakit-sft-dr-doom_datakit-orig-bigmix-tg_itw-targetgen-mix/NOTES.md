# Dr. Doom drafter trained from scratch on Datakit original + BigMix targetgen together (Horizon)

Why: one drafter trained on Datakit original and BigMix targetgen together, instead of one drafter per dataset.
Architecture: the layers6-taps5 drafters (6 Grug layers, FFN 7,936, taps 2/7/12/17/22, block 8, Markov rank 1024, 32,000-token draft vocabulary), target `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`. Settings: `settings.sh` (self-contained).

## Data (`data/build_mix.py`)
- Natural proportions, NO upsampling: Datakit original (`spec_data/datakit_orig`, 214 of 215 shards, about 46.0B tokens, symlinked; the last shard is held out) + BigMix targetgen (`spec_data/bigmix_tg`, 296,000 rows, 3.22B tokens, rewritten into the Datakit layout: int64 `input_ids`, no `source_row`). Targetgen is about 6.5% of the tokens (about 4.4% of the rows).
- Shuffling: the trainer permutes all training rows every epoch (`distributed_batch_sampler.py`), so the on-disk order is irrelevant for training. The validation split is positional (the last `1 - train_data_ratio` rows), so the LAST shard holds 3,000 random Datakit-held-out rows + 300 random targetgen rows (about 1:10 instead of the natural 1:22, because 140 targetgen rows would be too noisy; the validation loss is a sanity check, the real evaluation is acceptance length).
- `token_freq.pt/.json` = the two sources' counts summed: the 32k draft vocabulary follows the combined data.
- Prompts of the two sources were not deduplicated against each other.

## Length: one epoch, stopped by hand at 24 h EFFECTIVE time
Original completions are free; a targetgen row costs its regeneration time (BigMix-TG: 0.0595 h per 1k rows on 2 nodes, from 10.7 h / 180k rows and 6.9 h / 116k rows). Effective time = training wall time + regeneration credit of the targetgen rows seen. `tools/effective_time.py` predicts the stop step and, during the run, `--step S --train-hours H` says how far the run is.
At 1.14 s per step (the Datakit-orig run; unmeasured on this mix) the stop point is about step 66.7k (8.0B tokens, 16% of the epoch, about 48k of 296k targetgen rows seen), learning rate 0.85 of peak, not annealed. The epoch is about 409k steps and the schedule decays over all of it.
Stopping: `scancel` the job; the newest checkpoint is at most about 409 steps (8 min) older. Milestone copies every 8,180 steps land in `$WORK_DIR/milestones`.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/<this folder>
bash horizon/submit.sh data                 # builds $WORK_DIR/data on a compute node (about 38 GB rewritten)
bash horizon/submit.sh train --dry-run ; bash horizon/submit.sh train      # JOBS=2 TIME=hh:mm:ss to chain resumable jobs
```
Work dir `runs/grug_dspark_drdoom_layers6-taps5_datakit-orig-bigmix-tg-mix`; W&B run name = this folder's name. Smoke test: see the header of `horizon/submit.sh`.

## Horizon specifics (probe job 96294, 2026-11-16)
Compute nodes: 4x GB200 (189 GB), 1.7 TB RAM, 157 GB `/tmp`; InfiniBand interfaces `ibs2`, `ibP2p1s0`, `ibP16s4`, `ibP18p1s0` (the launcher uses `ibs2`); `srun --export=ALL ...` is accepted. The training env is `envs/speculators` (job 96258), the hidden-state server runs in `envs/vllm_fork`.
UNTESTED until the smoke run: the hidden-state vLLM server (Grug fork) feeding the trainer through Mooncake on Horizon, multi-node NCCL (`NCCL_IB_HCA` is left to NCCL: mlx5_2 is a 25 Gb port, mlx5_3 is down), the trainer on GPUs here.
