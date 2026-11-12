# Acceptance length vs. training-data row statistics (6-layer Dr. Doom drafters)

Written 2026-11-12. Target: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`. Drafter: 6 Grug layers, five taps (2/7/12/17/22), block size 8, Markov rank 1024, 32,000-token draft vocabulary, trained from scratch.
Question behind the figure: how do mean acceptance length (AL) and the per-row shape of the training data (how much of a row is reply text, how long rows are, how many supervised tokens were seen) relate across five training recipes?

## Files
| file | content |
|---|---|
| `method_summary.csv` | one row per method and checkpoint (7 rows; `used_in_figure` marks the 5 plotted): dataset statistics, steps, tokens seen, mean AL, AL per benchmark |
| `al_by_benchmark.csv` | long table, method x benchmark: AL, tokens/s (client concurrency 16; SPEED-Bench: total tokens / total benchmark time), source eval folder |
| `build_csvs.py` | rebuilds both CSVs from the eval folders (`runs/eval_matrix_*`) and the dataset manifests; run with the speculators env |
| `plot_al_vs_row_stats.py` | reads `method_summary.csv`, writes the figure |
| `al_vs_row_stats_paper.pdf` / `.png` | the figure: (a) AL vs supervised share, (b) vs supervised tokens per row, (c) vs total tokens per row, (d) vs supervised tokens seen |

## Methods (latest checkpoint of each approach in the figure)
| method | prompts | replies | step | note |
|---|---|---|---|---|
| Datakit original 24h | Datakit-SFT mixture (203 sources) | original | 75,834 | 0.20 epoch |
| NoFin original 14h | Datakit-SFT without `finance_v1` (326,695 rows removed) | original | 45,747 | 0.15 epoch of the filtered dataset; run in progress (resumed) when evaluated |
| BigMix original 24h | Open-PerfectBlend + Nemotron-Terminal + Nemotron-v2 | original | 95,146 | 2.78 epochs (repeated data) |
| BigMix targetgen | 296,000 BigMix rows | Dr. Doom's own replies (T=1, `/think`) | 27,081 | one epoch |
| Datakit targetgen | 281,000 Datakit rows | Dr. Doom's own replies | 22,472 | one epoch |
Also in the CSVs (not plotted): Datakit original 12h (37,917) and NoFin original 6h (17,986).

## Columns of `method_summary.csv`
- Dataset: `dataset_rows`, `dataset_tokens`, `dataset_supervised_tokens` are totals of the whole dataset (manifests; for the targetgen sets `token_freq.pt` and the merge log). `train_split_share` = share of rows the trainer trains on (0.999 or 0.995; the rest is validation).
- `epoch_length_steps` = steps of one pass over the train split. Exact for the targetgen runs (steps at `epoch0_end`), the BigMix run (34,167) and the full Datakit run (382,747); for the NoFin run (300,412) it is derived from the scheduler's learning rate (the same derivation returns 382,745 for the Datakit run). `epochs_seen` = `checkpoint_step` / `epoch_length_steps`.
- Per row (means over the whole dataset): `tokens_per_row_mean`, `supervised_per_row_mean` (tokens with loss mask 1), `supervised_share` = supervised / total tokens.
- Per step: a step packs whole rows to a budget of 32,768 tokens on each of 4 ranks (multipack sampler; attention is masked across the packed documents), so `rows_per_step` varies with row length while `tokens_per_step` stays near 118k to 127k. `supervised_per_step` = supervised tokens of the train split / epoch length.
- Seen: `supervised_seen_B` = `supervised_per_step` x steps (billions); `unique_supervised_seen_B` caps it at the train split's supervised tokens (differs only for the 2.78-epoch BigMix run: 7.33B seen, 2.63B unique); `total_tokens_seen_B` the same for all tokens. These assume a step consumes the epoch-average amount; for runs that did not finish an epoch they are estimates (about 1%), exact for the targetgen runs.
- `anchor_cap_share_of_supervised_per_step` = 4 x 12,288 anchors / supervised tokens per step (capped at 1): the trainer draws at most 12,288 anchors per packed sequence, so only that share of a step's supervised tokens can start a drafted block (my reading of `dflash/core.py`).
- `mean_AL_six_neutral` = mean AL over GSM8K, MATH-500, HumanEval, MBPP, MT-Bench, SPEED-Bench; the two home sets (Datakit and BigMix held-out) are left out because each favours one training set. `AL_<benchmark>` columns hold the per-benchmark values (also finance_v1 and the Datakit representative set where evaluated; blank = not run).
- `tokens_per_row_is_estimate`: true only for Datakit targetgen, whose total tokens per row (9,490) comes from a row sample (the supervised total is exact).

## Evaluation settings
Marin vLLM fork with the Grug-layer overlay, 8 drafted tokens, temperature 1, probabilistic draft sampling with standard rejection; concurrency 16 (SPEED-Bench 32, AIPerf); 4,096 new tokens (paper sets 2,048). One run per cell, no plain arm: tokens/s are not converted to speedups here. Each method's eval was run on different nodes and sessions, so tokens/s is only indicative.

## Caveats
- Five points, one training run each: the figure shows co-variation, not a fitted relation.
- Datakit TG and BigMix TG used different numbers of rows (281k, 296k) and ran fewer steps than the original-text runs; BigMix original repeats data.
- The NoFin number is for a checkpoint of a run that was still training (about 15% of its epoch, learning rate still about 0.85 of peak).
