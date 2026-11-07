# targetgen DSpark drafter for Dr. Doom on Open-PerfectBlend + Nemotron-Terminal-Corpus + Nemotron-Post-Training-Dataset-v2 prompts, from random initialisation

Same chain, drafter (3 Qwen3 layers, FFN 3840, 280M parameters, block 8, 32k draft vocabulary) and recipe as [`../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen/); only the prompts differ:
rows are sampled from the "big dataset" below instead of Datakit conversations. Dr. Doom (`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`) writes every reply in **`/think` mode**, and the drafter trains on those replies only (targetgen).
The DFlash paper's recipe is the same idea on this kind of data: about 800K samples of Nemotron-Post-Training-Dataset-v2 + CodeAlpaca with the target's own replies, 6 epochs, sequences capped at 3k tokens.

## Why
The itw PerfectBlend run trained on other models' prompts AND replies (off-policy in both x and y*); it scored RLVR 1.655 / Datakit pooled 2.196 against 2.723 / 2.939 for the Datakit itw-80k drafter (see `../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw/archive_perfectblend_only/NOTES.md`). Here only the prompts x are off-policy; the replies are the target's own.

## The "big dataset" (`/scratch/09749/atutej/marin_big_pool/dataset_{itw,think}_seq32768`, built by `../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw/data/build_big_pool.py`, finalized with the Datakit `finalize_dataset.py`)
| Source | Rows | Tokens | Supervised | Clipped at 32,768 |
|---|---|---|---|---|
| `mlabonne/open-perfectblend` | 1,358,785 | 0.701B | 0.491B | 0 |
| `nvidia/Nemotron-Terminal-Corpus` (all rows of the Datakit pipeline's normalized store) | 174,064 | 1.723B | 0.692B | 1,692 |
| `nvidia/Nemotron-Post-Training-Dataset-v2`, `chat` | 627,389 | 1.336B | 1.039B | 942 |
| v2 `code` | 175,000 | 0.246B | 0.162B | 0 |
| v2 `math` | 239,467 | 0.187B | 0.161B | 0 |
| v2 `stem` | 355,000 | 0.174B | 0.098B | 0 |
| **together** | **2,929,705** | **4.367B** | **2.644B** | |

Terminal is 6% of the rows and 39% of the tokens; v2 is 48% of the rows and 44% of the tokens (chat alone 31%); PerfectBlend 46% of the rows and 16% of the tokens. The multilingual v2 splits (4.9M rows, about 24B tokens) are not included. Rows are shuffled together in 20 shards; nothing is filtered (v2 is used as is: 331 of its 1,397,187 rows have no trainable token and were dropped).
- PerfectBlend: 62,124 of 1,420,909 conversations dropped because the turns do not start with a user turn, alternate, and end with an assistant turn.
- v2 rows are single user/assistant exchanges with an empty system message. `chat` (Qwen3-235B/30B replies): 545,844 rows with reasoning "on" (a `<think>` block, turned into Marin's `<|start_think|>...<|end_think|>`) and 81,876 "off". `code`, `math`, `stem` (DeepSeek-R1-0528): all "off", plain answers (the empty `<think></think>` prefix of code and math is dropped).
- Terminal rows: the Datakit pipeline's own text (all `/nothink`, no reasoning blocks, no tool-call tags).
Everything is rendered with the Marin chat template and tokenized and masked with the Datakit code (assistant-only loss mask, reasoning block included). Mask checks against the template's generation spans: 278 of 278 PerfectBlend and 309 of 309 v2 rows.

The multilingual v2 splits are not included. In this folder the original replies in the dataset are discarded (see "The header").

## The header
Every prompt starts with `Reasoning: /think`: the model's own default at serving (the `else` branch of its inference `chat_template.jinja`) and the header of every eval prompt set. The `think` variant of the dataset has it on every row (sample of 25.6k rows: 100%); the Terminal rows' native `/nothink` is changed to `/think` (the only change to their text).
`tools/regenerate_token_rows.py` replays each assistant turn at the token level and replaces it with Dr. Doom's own generation (later user turns and tool outputs are kept as they are). The Datakit targetgen runs kept each row's own header (83% `/nothink`).

## Pilots (one node DP4 x EP4, replies capped at 12,288 tokens; the real cap is 32,768)
| | PerfectBlend prompts only (384 rows) | the big pool, uniform sample (512 rows) |
|---|---|---|
| generated tokens per row | 5,292 | **6,162** |
| total row length | median 3,170, mean 5,460, p90 12,606 | median 4,624, mean 6,707, p90 12,944 |
| rows that hit the 12,288 cap | 14% | **27%** |
| assistant turns replayed per row | 1.37 | 1.39 |
| errors | 0 | 0 |
| run | 4.8 min after a 151 s server start | 5.1 min after a 206 s server start, 10.4k generated tokens/s on one node |

The v2 STEM, math and code prompts and the Terminal sessions make `/think` replies much longer than PerfectBlend's; with the real 32k cap the mean is probably 8-11k generated tokens per row (the Datakit regeneration: 6.2k). Raw parts: `runs/targetgen_pool_pilot_2026-10-30`, `runs/targetgen_pblend_pilot_2026-10-30`.

## Size (estimates)
`REGEN_ROWS=180000` (settings.sh), sampled uniformly from the 2,929,705 rows of the big dataset, at an assumed 9k generated tokens per row: about 1.62B generated tokens = about **11.3 h of regeneration** on 2 nodes (about 40k generated tokens/s, as the Datakit regeneration; close to one 12 h job, the chained second regen job covers the rest), about 1.66B merged tokens = about 13.7k steps at 0.54 s = about 2 h of training,
plus about 1.5 h of merge, startup and queue gaps: **about 15 h in all** (13-18 h for 8-11k generated tokens per row). 150k rows would be about 12.5 h in all, 200k rows about 16-17 h, 100k rows about 9 h. The variable is read when the jobs start (`PBLEND_REGEN_ROWS=<n> bash vista/submit_chain.sh`).
The regeneration is resumable per part; the rows sampled depend on `--num`, so do not change it between the regen jobs. Training uses the default 12 dataloader workers: rows are long (about 9k tokens), so a step holds about 4 rows, like the Datakit runs (compute-bound).

## Run it (from a LOGIN node; nothing is submitted by writing this folder)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_targetgen
bash vista/submit_chain.sh --dry-run
bash vista/submit_chain.sh        # regen (12 h) -> regen again (afterany, resumes; exits quickly if finished) -> merge + train (afterok): 3 jobs
```
Work dir `runs/grug_dspark_drdoom_big_pool_targetgen` (new; `settings.sh` refuses the old runs' dirs). W&B: `atutej/marin_speculator`, run `dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_targetgen`. The train job starts with the merge, which also builds the draft vocabulary from Dr. Doom's own replies. `conda activate` is replaced by a hand activation in `settings.sh`.

## After training
Evaluate on the Dr. Doom target with the same harness and prompts as the other Dr. Doom drafters (`tools/eval_acceptance/serve_eval.sh`: RLVR 256, Datakit first/later turn, probabilistic drafting) and compare with the Datakit itw-80k drafter (RLVR 2.723, pooled 2.939), the Datakit targetgen drafter (RLVR 2.758, Datakit first 2.909, later 2.398) and the PerfectBlend itw runs.
