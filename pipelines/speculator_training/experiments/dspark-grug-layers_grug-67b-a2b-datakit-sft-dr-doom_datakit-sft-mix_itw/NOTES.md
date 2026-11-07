# itw DSpark drafter for the Dr. Doom target with Grug layers (parameter-matched to the Qwen3 layers)

The Dr. Doom itw drafter ([`../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/): 3 Qwen3 layers, FFN 3,840, 280M parameters, trained on the Datakit SFT mixture's own replies against
[`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom)) rerun from step 0 with
**Grug-style decoder layers instead of Qwen3 layers**, sized to the same number of parameters, and nothing else changed: same target (Dr. Doom), data (the full Datakit dataset, itw), recipe (lr 3e-4, loss 0.1 CE + 0.9 TV, 12,288 anchors, 32,768-token rows, block 8, 32k draft vocabulary, Markov and confidence heads), seed,
and the same learning-rate schedule (linear decay over the FULL epoch of 382,747 steps, so a run stopped at step 80,813 is not annealed). The parent's 12 h job reached step 80,813; its checkpoint there (the `itw-80k` drafter on the Hub) is the baseline, so no separate baseline run is needed.

## Why
The target's own layer is not a Qwen3 layer (see the table). The question is whether a drafter built from the target's layer design predicts the target better than a Qwen3-style one of the same size.

## The Grug layer (speculators `layer_type="grug"`, `models/dflash/grug_definitions.py`)
| | Qwen3 layer (parent) | Grug layer (this run) |
|---|---|---|
| q/k norm | learned RMSNorm weights | parameter-free RMSNorm, then q x `qk_mult` 1.75 |
| attention output | straight into `o_proj` | XSA (the part along the token's own value removed), then a per-head `2 x sigmoid` gate that starts at identity |
| norms | RMSNorm | RMSNorm followed by a rank-128 sigmoid "gated norm" (before attention and before the FFN) |
| positions | full RoPE | half-RoPE (`partial_rotary_factor` 0.5 in the draft config, as in the target) |
| FFN | dense SwiGLU, width 3,840 | dense SwiGLU (the Grug shared expert), width 3,664; no routed experts yet |
| attention mask | full, bidirectional inside the draft block | same (no sliding window, no NoPE: the block is bidirectional, so NoPE would make the mask tokens position-blind) |

## Size (exact: the decoder layers instantiated from the two configs, counted on the meta device)
| | parent (Qwen3, FFN 3,840) | this run (Grug, width 3,664) |
|---|---|---|
| decoder layers (3) | 137,641,728 | 137,671,680 (+29,952, +0.022%) |
| whole drafter (plus the same 142,614,273 fixed parameters) | 280,256,001 | about 280,285,953 |

3,664 is the multiple of 16 nearest to the exact match (3,662.9; 3,663 would be +0.005% but is not GEMM-aligned). The draft config's `intermediate_size` is the Grug shared-expert width here.

## Baseline (the parent's checkpoint at step 80,813)
Trainer `accept_len` over the last 500 steps: 4.201. Serving on the Dr. Doom target (acceptance length / speedup over plain decoding; one run each):
| Prompts | 2026-10-17 evaluation | 2026-10-20 evaluation (Hub card) |
|---|---|---|
| RL prompts (256) | 2.697 / 1.81x | 2.689 / 1.77x |
| Datakit first turn (833) | | 2.983 / 1.96x |
| Datakit later turn (191) | | 2.522 / 1.52x |

Speedups depend on the session's plain-decoding speed (it varied by about 20% between sessions), so compare speedups only inside one session; acceptance length is stable (repeats differ by about 0.02). Raw results: `docs/results/dspark_eval/`.

## What is different, and the risks
Only `draft_config.json` (`intermediate_size` 3,664, `rope_parameters.partial_rotary_factor` 0.5) and the two flags `settings.sh` appends to the trainer command (`--draft-layer-type grug --grug-qk-mult 1.75`). `settings.sh` also keeps 8 checkpoints (`SAVE_TOTAL_LIMIT=8`, the parent kept 3) so the checkpoint at step 80,813 (211 x 383) survives.
- **Step time is unknown.** The Grug layer adds gated norms and XSA; if it is slower than the parent (0.523 s per step on 2 nodes, about 82k steps in 12 h), one 12 h job will not reach step 80,813, hence up to two chained jobs. Compare the two drafters AT THE SAME STEP (this run's `checkpoint-80813` against the parent's), not at the end of the time limit.
- **Only a tiny-scale check exists.** On a random-init 73M Grug target (300 steps, one seed) the Qwen3 drafter was slightly ahead of this layer at equal parameters (validation expected acceptance length 1.963 against 1.922; W&B `f4jrwyzs` against `drlmzjcr`). That target is not a real distribution, so it says nothing about Dr. Doom; treat the real run as an open question.
- **Provenance:** the trainer imports the speculators working tree (editable install) and records its commit and diff in the checkpoint directory. Commit the Grug-layer code (`models/dflash/grug_definitions.py`, `models/dspark/{config,core}.py`, `train/config/schema.py`) before launching so the checkpoints name a commit.
- **Serving needs a new vLLM build.** The Marin vLLM fork release in use (`marin-vllm-gpu-20260929-39e62869693c`) has no `GrugDSparkModel`; the draft architecture is on the fork's `grug-dspark-layers` branch (`vllm/model_executor/models/grug_dspark.py` plus small edits in the registry, `speculative.py`, the speculators config mapping and `qwen3_dspark.py`). Training does not need it; evaluation does.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark-grug-layers_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
bash vista/submit_chain.sh --dry-run
bash vista/submit_chain.sh                 # JOBS=2 (default): up to two 12 h jobs on the 2 gb nodes, the second afterany the first
```
Each job builds the data view (`vista/make_data_view.sh`: the Datakit shards symlinked, metadata copied; the trainer's vocabulary cache goes into the view, the shared dataset is not touched), then runs the itw folder's `run_2node_ddp.sh train` with this folder's settings (`EXP_DIR_OVERRIDE`).
A job whose run already has a checkpoint at or past step 80,813 exits at once (`STOP_AFTER_STEP`); if the first job does NOT get there, cancel the second (`scancel`) once `checkpoint-80813` exists, or submit with `TIME=<hh:mm:ss>` sized to the remaining steps (`JOBS=1`).
W&B: project `atutej/marin_speculator`, run `dspark-grug-layers_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`; local files in `runs/grug_dspark_itw_drdoom_grug-layers/logs/tracker`. No HF upload (chain jobs never upload).

## Guards (lessons from earlier runs)
`settings.sh` assigns every fresh path outright after sourcing the parent's settings and recomputes `LOGGER_ARGS` (the parent's settings build it from the parent's log directory). The data view refuses the source dataset and any directory it did not make. The job script names its variable `RUN_WORK_DIR`, not `WORK` (TACC's `$WORK` feeds `env.sh`).

## After training
1. Copy `checkpoint-80813` (`model.safetensors`, `config.json`, `config.py`) into a new `runs/eval_ckpts/<name>/`.
2. Evaluate with the same harness, prompts and settings as the parent's drafters (`tools/eval_acceptance/serve_eval.sh` with `MODEL` the Dr. Doom target; the RL prompts and the Datakit first-turn and later-turn prompts), with the vLLM build above, the parent's `itw-80k` drafter and a plain-decoding arm in the SAME session.
3. A single run per arm; the parent's `itw-160k` and `warm80k` drafters are further references.
