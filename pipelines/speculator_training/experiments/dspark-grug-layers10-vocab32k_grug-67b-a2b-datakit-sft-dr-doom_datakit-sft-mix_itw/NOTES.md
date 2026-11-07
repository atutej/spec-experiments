# Big itw DSpark drafter for Dr. Doom: 10 Grug layers, 32,000-token draft vocabulary (one epoch of Datakit SFT)

Target: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`. Data: the full Datakit SFT mixture, itw (the mixture's own replies, no regeneration), ONE epoch (382,747 steps, about 46B tokens), learning rate 3e-4 decayed linearly over the whole epoch with 1% warmup (the original itw recipe).
Parent settings: [`../dspark-grug-layers_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark-grug-layers_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/) (same data, seed, loss 0.1 CE + 0.9 TV, 12,288 anchors, 32,768-token rows, Markov and confidence heads).

## Configuration (differences from the parent)
| | parent | this run |
|---|---|---|
| layers (Grug) | 3 | **10** |
| FFN (dense shared expert) | 3,664 | **7,936** |
| hidden / heads / head size | 2,560 / 20 query, 5 KV / 128 | same |
| vocabulary | 32,000 (frozen copy of the target's rows) | **32,000 (same selection; unchanged)** |
| tapped target layers | 2, 12, 22 (+ final) | **23, 24, 25 (+ final): the last three of 26** |
| block size | 8 | **5** |
| attention | full | full (sliding window left for later) |
About 848M trainable parameters (excluding the frozen head rows and the embedding): layers 787.0M (78.7M each), Markov head 41.0M (W1 over the target vocabulary 32.8M, W2 to the 32,000 draft tokens 8.2M), fc 19.7M. The checkpoint also stores the frozen 32,000 x 2,560 head rows (81.9M): about 930M saved.

## Estimates (not measured)
Step time about 1.5 s on 2 nodes (the 6-layer run measured 0.89 s per step; the per-layer cost model gives about 1.5 s): about 160 h for the epoch, 13-14 chained 12 h jobs. Unmeasured; memory at 12,288 anchors is the main risk (the 8-layer, FFN 7,680 run ran out of memory before the allocator setting).

## Run it (from a LOGIN node)
```
cd .../experiments/dspark-grug-layers10-vocab32k_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
bash vista/submit_chain.sh --dry-run
JOBS=1 bash vista/submit_chain.sh        # first job alone: look at its first steps (OOM, taps, step time) before queueing the rest
AFTER=<last job id> JOBS=2 bash vista/submit_chain.sh   # the gb QOS allows 3 submitted jobs: top up as jobs finish
```
Each job exits at once when `checkpoints/epoch0_end` exists. At the start of every job the newest step checkpoint is copied to `runs/grug_dspark_itw_drdoom_grug-layers10-vocab32k/milestones/` (about every 11% of the epoch; the rolling checkpoints keep only the newest 3).
W&B: `atutej/marin_speculator`, run name = this folder's name. No HF upload from chain jobs.

## Risks (first job is the test)
- Job 1120063 died in kernel compilation (shared Triton cache, stale file handle: the trainer now has node-local caches). Job 1120105 ran out of memory in the backward of the 128K loss at 12,288 anchors (157 GB allocated of 184 GiB; 14.7 GiB per logits tensor). Job 1120156 (9,216 anchors) was cancelled by hand: fewer anchors per 32,768-token row changes the anchor density (0.375 to 0.28 per token), which is a recipe change. `MAX_ANCHORS` stays 12,288 until the memory problem has a fix that keeps the recipe.
- Serving: the vLLM fork's loader should fall back to the target vocabulary and share its head when the checkpoint omits them (read in `qwen3_dspark.py`, untested with `grug_dspark.py`). Test on the first milestone checkpoint.
- `epoch0_end` as the finishing marker is taken from the Datakit driver script (`vista/drive_chain.sh`) and is not yet seen for this run.
- The Grug-layer code is uncommitted in speculators and the vLLM fork; the checkpoints record the diff, but commit before the run if you want a commit name.

## History
Started as 8 layers (job 1120105 OOM at 12,288 anchors); changed to 6 layers with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the trainer, anchors unchanged. Failed 8-layer attempts: `runs/grug_dspark_itw_drdoom_grug-layers8-fullvocab/` (logs only).

## History
The 6-layer sibling (`dspark-grug-layers6-fullvocab_...`, since removed from the repo) ran 12 h to step 47,875 (12.5%): Datakit-SFT held-out 3.011, SPEED-Bench 2.69. This folder is a NEW architecture (10 layers, FFN 7,936) trained from step 0.

## 2026-11-05: resuming from checkpoint-31789 runs out of memory
Jobs 1123392 (12,288 anchors, CUDA driver OOM), 1123406 (12,288; 166.3 GB allocated, 14.68 GiB request, 11.3 GB free) and 1123413 (11,264 anchors; 167.5 GB allocated, 13.46 GiB request, 10.0 GB free) all ran out of memory in the first backward after the resume; the first 31,789 steps (job 1121700, no resume, 12,288 anchors) fit. GPUs were clean at job start (1-3 MiB). Cutting the anchors by 8% did NOT lower the allocated memory, so the memory is not dominated by anchor-scaled tensors, and the cause of the extra allocation after a resume is unknown. `settings.sh` is back at 12,288 anchors (the 11,264 attempt is reverted).

## Why 32k
The full-vocabulary 10-layer sibling (`dspark-grug-layers10-fullvocab_...`, since removed from the repo) trained 12 h to step 31,789 (SPEED-Bench 2.68) but could not be resumed: out of memory in the first backward after a resume, three times (12,288 and 11,264 anchors). This folder repeats the architecture from step 0 with the 32,000-token draft vocabulary of the earlier drafters, whose 4x smaller loss tensors leave room.
