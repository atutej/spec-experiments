# itw DSpark drafter for the Dr. Doom target

Same experiment as [`../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/) (the drafter trained on the replies of the Datakit SFT mix as written, no regeneration; see `docs/naming_itw_targetgen.md` for the names), restarted **from step 0 for one 12 h job**, with one change: the target is
[`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) (Dr. Doom, an FTPO de-looping fine-tune of the 09.21 Snowball SFT model) instead of the 09.21 model itself.
The drafter learns from the target's hidden states (layers 2, 12, 22 and the final layer), its `lm_head` rows and its gated final norm, so this run gives a drafter matched to Dr. Doom.

## Why
The existing itw and targetgen drafters were trained on 09.21 hidden states. On Dr. Doom (RL prompts, 256 prompts, probabilistic drafting, concurrency 16) the itw drafter gets acceptance length **2.667 (1.79x)** against 3.141 (2.16x) on 09.21 (issue marin-community/marin#10227, table a2).
This run measures how much of that 0.47 drop a matched drafter recovers.

## What is the same / different
| | |
|---|---|
| Same | drafter (`../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/draft_config.json`: 3 layers, FFN 3840, block 8, 32k draft vocabulary), recipe (`settings.sh` of the itw run: lr 3e-4, loss 0.1 CE + 0.9 TV, 12,288 anchors, 32,768-token rows, checkpoint every 0.1% of the epoch), data (Datakit SFT mix, 6,471,159 rows, train ratio 0.999), 2-node DDP layout (`run_2node_ddp.sh` of the itw folder), seed and therefore the data order |
| Different | target (`MODEL` in `settings.sh`); a fresh work dir `runs/grug_dspark_itw_drdoom`; the data directory is a symlink view of the Datakit dataset (below); one 12 h job instead of a chain |

Dr. Doom is a drop-in verifier: its safetensors have the same 502 weight names, shapes and dtypes as 09.21 (checked), the same tokenizer and chat template, and the speculators fork's `GrugMoeConfig` loads its `config.json` (hidden 2560, 26 layers, vocab 128,256). Its `lm_head` differs from 09.21's by 0.07% (relative norm), so the 32k draft vocabulary (built from the dataset's token frequencies) and the draft head slice
are the same in structure; the slice is re-initialised from Dr. Doom's head. Dr. Doom's config also carries extra keys (`disable_long_rope`, `disable_pko`, `global_every`, `sconv*`, ...); vLLM served the model correctly with them in the evaluations, and the drafter pipeline reads only `hidden_size`, `num_hidden_layers` and the layer count.

## Budget and schedule
12 h at 0.523 s per step on 2 nodes is about 82k steps (the itw run reached 80,813). **Default: the itw run's own schedule**, a learning rate that decays linearly over the *full epoch* (382,747 steps) and is cut by the time limit, so the learning rate is not annealed at the end. This keeps the comparison to the 09.21-trained itw drafter like for like.
To anneal inside the 12 h instead (a better final drafter, but no longer the same schedule): export `MAX_STEPS=78000` (or `TOKEN_BUDGET=<tokens>`) before submitting; the trainer then decays to zero over exactly that many steps and stops. Leave margin for startup (about 10-15 min) and the final validation.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
bash vista/submit_chain.sh --dry-run     # prints the sbatch line
bash vista/submit_chain.sh               # one 12 h job on 2 gb nodes (CGAI-gb reservation), JOBS=n for resumable follow-ups
```
The job builds the data view (`vista/make_data_view.sh`, seconds, CPU), then runs the itw folder's `run_2node_ddp.sh train` with this folder's settings (`EXP_DIR_OVERRIDE`). Training resumes from `runs/grug_dspark_itw_drdoom/checkpoints` if the folder exists, so a job that hits its limit is continued by submitting again (`AFTER=<jobid> bash vista/submit_chain.sh`).
W&B: project `atutej/marin_speculator`, run `dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`; its local files go to `runs/grug_dspark_itw_drdoom/logs/tracker`.

## Data view (why `DATA_DIR` is not the dataset)
The trainer reads `token_freq.pt` from the data directory and writes its vocabulary cache (`d2t-32000.npy`, `t2d-32000.npy`) there. `vista/make_data_view.sh` builds `runs/grug_dspark_itw_drdoom/data` with the 215 shards as symlinks and `state.json`, `dataset_info.json`, `manifest.json`, `token_freq.pt` copied, so the shared dataset
(`/scratch/09749/atutej/marin_datakit/dataset_seq32768`) is never written by this run. Checked: the view loads with `datasets.load_from_disk`, has 6,471,159 rows, and rows 0, 123,456 and the last row equal the source's.

## Guards (lessons from the targetgen run)
`settings.sh` assigns every fresh path outright after sourcing the itw settings (a `${VAR:-default}` would have kept the itw run's `runs/grug_dspark_full`), and recomputes `LOGGER_ARGS`, which the itw settings build from the itw run's log directory (the targetgen run wrote W&B files into the itw run's `logs/tracker` because of this).
`make_data_view.sh` refuses to touch the source dataset or a directory it did not make. Resolved paths were printed and checked before submitting anything.

## After training
1. Copy the final checkpoint's `model.safetensors`, `config.json`, `config.py` into a new `runs/eval_ckpts/<name>/` (as for the other drafters; the run directory stays untouched).
2. Evaluate with the same harness and prompts as the issue: `tools/eval_acceptance/serve_eval.sh` with `MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`, the Nemotron RLVR1 prompts (`MAX_PROMPTS=256 MAX_TOKENS=49152 MAX_MODEL_LEN=102400`, probabilistic sampling flags, `PLAIN=1`), and the multi-turn Datakit validation prompts (`SUBSETS='first_turn later_turn'`). Reference numbers on Dr. Doom with the 09.21-trained drafters: itw 2.667 / 1.79x, targetgen 2.465 / 1.62x (plain 2,023 tok/s).
3. Compare also on the 09.21 target to see what the matched drafter loses there (reference: itw 3.141 / 2.16x).
4. Upload with `tools/upload_drafter_to_hf.py --stage-dir ...` (public repo `atutej/<this folder's name>-<date>`).

Caveats: a single run; an unannealed checkpoint with the default schedule; Dr. Doom's replies are longer (about 16k tokens on the RL prompts), so absolute throughput differs from 09.21's.

## Result (2026-10-17)
Training job 1076643 ran the 12 h limit (TIMEOUT) and reached step 80,813 (last complete checkpoint `checkpoint-80813`; train `accept_len` over the last 500 steps 4.201, against 4.263 for the 09.21-trained itw run). Evaluation: job 1077583, target Dr. Doom, the same 256 RL prompts and settings as the issue's table (a2), plain baseline reused from that run (2,023 tok/s):

| Drafter (all trained itw, step 80,813) | Acceptance length | Output tokens/s | Gain vs plain | Per-position acceptance, positions 0-3 |
|---|---|---|---|---|
| trained on 09.21 hidden states | 2.667 | 3,630 | 1.79x | 0.673, 0.418, 0.249, 0.145 |
| **trained on Dr. Doom hidden states (this run)** | **2.697** | 3,660 | **1.81x** | 0.684, 0.428, 0.255, 0.147 |
| targetgen, trained on 09.21 hidden states | 2.465 | 3,270 | 1.62x | 0.647, 0.374, 0.207, 0.112 |

The matched drafter is ahead by 0.03 in acceptance length (about 1%) and 0.8% in throughput: within what a single run on 256 prompts can resolve. Raw results: `docs/results/dspark_eval/rl_prompts_256_target-dr-doom/itw_drdoom_step80813/`.
Reading: the 0.47 drop that the 09.21-trained itw drafter shows on Dr. Doom (3.141 -> 2.667) is almost entirely NOT recovered by training on Dr. Doom's hidden states, so it does not come mainly from the hidden-state mismatch. It is more likely a property of Dr. Doom's text (a de-looping fine-tune: less repetitive output, so less to predict cheaply); not tested here.
