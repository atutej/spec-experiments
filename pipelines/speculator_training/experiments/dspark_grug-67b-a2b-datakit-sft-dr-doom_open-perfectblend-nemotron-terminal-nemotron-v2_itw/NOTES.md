# itw DSpark drafter for Dr. Doom on Open-PerfectBlend + Nemotron-Terminal-Corpus + Nemotron-Post-Training-Dataset-v2

Existing text, no regeneration (itw). Parent: [`../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/) (3 Qwen3 layers, FFN 3840, 280M parameters, 32k draft vocabulary, same recipe and 4-rank layout);
the data is the "big dataset" below instead of the Datakit mix. This replaces the earlier PerfectBlend-only and PerfectBlend + Terminal plans (v2 is the data the DFlash paper trains on: about 800K samples of v2 + CodeAlpaca with the target's own replies, 6 epochs, 3k-token cap, per its paper).
The first PerfectBlend run (rows with no system header, stopped by its time limit at 5.3 epochs, RLVR 1.655 / Datakit pooled 2.196) is documented in [`archive_perfectblend_only/NOTES.md`](archive_perfectblend_only/NOTES.md) and is not continued.

## The "big dataset" (`/scratch/09749/atutej/marin_big_pool/dataset_{itw,think}_seq32768`, built by `data/build_big_pool.py`, finalized with the Datakit `finalize_dataset.py`)
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

## Headers: each row keeps its own reasoning mode
PerfectBlend and the Terminal rows are `Reasoning: /nothink` (PerfectBlend's replies are direct answers; the Terminal rows carry it natively); v2 `chat` rows with reasoning "on" are `Reasoning: /think` with a reasoning block, all other v2 rows `/nothink`. In a 25.6k-row sample of the dataset 81% of the rows are `/nothink` (no reasoning block) and 19% `/think` (99.9% of those have a `<|start_think|>` block),
like the Datakit rows (83% / 16%). The eval prompts use `/think` (the model's serving default). The first PerfectBlend run had no header at all (the Marin template writes it only when a reasoning mode is passed).

## Budget
One epoch is about 4.37B tokens = about 35k steps (130k tokens per step on 4 ranks, about 95% packing). **`EPOCHS=2`** is about 70k steps = about **8.7B tokens** (the Datakit itw-80k run: 80,813 steps, 9.7B tokens); the learning rate decays over the 2 epochs.
At about 0.4-0.55 s per step (short v2 and PerfectBlend steps are fast, Terminal and v2-chat steps are compute-bound like the Datakit ones) that is about 8-11 h: up to two chained 12 h jobs. End-of-epoch checkpoints `0/` and `1/` (`epoch0_end` is the 1-epoch checkpoint) are never pruned; a step checkpoint every 0.03 epoch (newest 3 kept); 0.5% of the rows (about 14.6k) held out for per-epoch validation.

## Dataloader
`--num-workers 48 --prefetch-factor 2` (defaults 12 and 4): a step holds about 80 rows on average (about 21 per rank), many of them short, each fetched by its own hidden-state request in one worker; 12 workers stalled the first PerfectBlend run (mean step 1.48 s, median 0.29 s; see its notes).
Not yet tested at this setting: **watch the first 10 minutes** (startup time, memory, step time). Each worker creates its own Mooncake client (16 GiB local buffer + 32 GiB mounted segment, virtual); if the start is slow or memory runs out, lower `MOONCAKE_GLOBAL_GIB` / `MOONCAKE_LOCAL_GIB` (set before sourcing the parent's settings) or use 32 workers.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw
bash vista/submit_chain.sh --dry-run
bash vista/submit_chain.sh                 # up to two chained 12 h jobs on the 2 gb nodes; the second exits at once if epoch1_end exists
```
Work dir `runs/grug_dspark_itw_drdoom_big_pool` (new). W&B: `atutej/marin_speculator`, run `dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw`. `conda activate` is replaced by a hand activation in `settings.sh` (conda on /work hung on 2026-10-29).

## After training
Evaluate on the Dr. Doom target with the same harness and prompts as the other Dr. Doom drafters (`tools/eval_acceptance/serve_eval.sh`: RLVR 256, Datakit first/later turn, probabilistic drafting) and compare with the Datakit itw-80k drafter (RLVR 2.723, pooled 2.939, same session), the first PerfectBlend run and the targetgen runs.
