# itw DSpark drafter with a 3x wider FFN (457M parameters)

The itw drafter ([`../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/): 3 Qwen3-style layers, FFN 3,840, 280M parameters, trained on the Datakit SFT mixture's own replies against the 09.21 SFT model) rerun from step 0 with
**`intermediate_size` 11,520 instead of 3,840**, and nothing else changed: same target (09.21), data (the full Datakit dataset, itw), recipe (lr 3e-4, loss 0.1 CE + 0.9 TV, 12,288 anchors, 32,768-token rows, block 8, 32k draft vocabulary, Markov and confidence heads), seed,
and the same learning-rate schedule (linear decay over the FULL epoch of 382,747 steps, so a run stopped at step 80,813 is not annealed). The existing itw run, stopped by its 12 h limit at step 80,813, is the baseline; no separate baseline run is needed.

## Why
Whether a larger drafter raises acceptance. The current FFN is 1.5 x hidden, half the `speculators` default for a MoE target (3 x hidden = 7,680); 11,520 is 3 x the current FFN. A wider FFN adds no sequential layer and no key-value cache (a deeper drafter would).

## Size (exact: the decoder layers instantiated from the two configs)
| | itw | this run |
|---|---|---|
| decoder layers (3) | 137,641,728 | 314,588,928 |
| whole drafter (plus 142,614,273 fixed: input projection 19.7M, frozen `lm_head` slice 81.9M, Markov head 41.0M, confidence head, norms) | 280,256,001 | **457,203,201 (1.63x)** |

## What is different, and the one risk
Only `draft_config.json` (`intermediate_size`). `settings.sh` also keeps 8 checkpoints (`SAVE_TOTAL_LIMIT=8`, the itw run kept 3) so the checkpoint at step 80,813 (211 x 383, the itw run's last) survives. Its step time is unknown: the itw run took 0.523 s per step on 2 nodes
(about 82k steps in 12 h). If the wider drafter is slower, 12 h will not reach step 80,813, which is why two chained jobs are planned. Compare the two drafters AT THE SAME STEP (the checkpoint at 80,813 against the itw run's `checkpoint-80813`), not at the end of the time limit.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark-ffn3x_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw
bash vista/submit_chain.sh --dry-run
bash vista/submit_chain.sh                 # JOBS=2 (default): two 12 h jobs on the 2 gb nodes, the second afterany the first
```
Each job builds the data view (`vista/make_data_view.sh`: the Datakit shards symlinked, metadata copied; the trainer's vocabulary cache goes into the view, the shared dataset is not touched), then runs the itw folder's `run_2node_ddp.sh train` with this folder's settings (`EXP_DIR_OVERRIDE`).
A job whose run already has a checkpoint at or past step 80,813 exits at once (`STOP_AFTER_STEP`), so if the first job gets there the second does nothing. If the first job does NOT get there, the second runs to its own 12 h limit, far past the comparison point:
cancel it (`scancel`) once `checkpoint-80813` exists, or submit it with `TIME=<hh:mm:ss>` sized to the remaining steps (`TIME=... bash vista/submit_chain.sh` with `JOBS=1`).
W&B: project `atutej/marin_speculator`, run `dspark-ffn3x_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw`; local files in `runs/grug_dspark_itw_ffn3x/logs/tracker`. No HF upload (chain jobs never upload).

## Guards (lessons from earlier runs)
`settings.sh` assigns every fresh path outright after sourcing the itw settings and recomputes `LOGGER_ARGS` (the itw settings build it from the itw run's log directory). The data view refuses the source dataset and any directory it did not make. The job script names its variable `RUN_WORK_DIR`, not `WORK` (TACC's `$WORK` feeds `env.sh`).
Checked before launch (login node, nothing written): the resolved paths are all new, and the new draft config differs from the itw one only in `intermediate_size`.

## After training
1. Copy `checkpoint-80813` (`model.safetensors`, `config.json`, `config.py`) into a new `runs/eval_ckpts/<name>/`.
2. Compare with the itw drafter at step 80,813 on the same prompts and settings: the single-round probe on the Datakit and RLVR prompts, and chained serving on the 09.21 target (RL prompts: itw 3.141 acceptance length, 2.16x; Datakit validation prompts: 3.122, 1.99x), and the trainer's `accept_len` at that step (itw: 4.262 over the last 500 steps).
3. A single run per arm; repeats of one configuration differ by about 0.02 in acceptance length.
