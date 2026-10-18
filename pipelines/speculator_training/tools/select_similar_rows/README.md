# select_similar_rows

Pick the Datakit SFT rows nearest to a set of RL prompts, as a new training dataset in the Datakit format. Three stages, each writing to a new directory
and each resumable or cheap to rerun. Design and decisions: `memory`-style plan in the experiment's NOTES (itw-rlsimilar).

| stage | script | env | does |
|---|---|---|---|
| 1 | `embed_rows.py rl` / `datakit` | `embed` | embeds "system + first user" (left-truncated to 8192 EmbeddingGemma 2 tokens, symmetric task prefix) of the RL prompts and of every Datakit row, one process per GPU |
| 2 | `select_rows.py` | `embed` (torch) | exact top-K neighbours of every RL prompt, then round-robin selection (round r = every prompt's r-th nearest) up to a packed-token budget; the last partial round is a seeded random subset of the prompts |
| 3 | `write_subset.py` | `speculators` | writes the selected rows (shuffled) with `save_to_disk`, recomputes `token_freq.pt` / `manifest.json` from them, runs the experiment's `finalize_dataset.py` |

`common.py` holds what text is embedded. The auxiliary first system header of a Datakit row (reasoning mode + tools block) and the tools of the RL rows are not embedded.

Smoke test (2026-10-18, 3,000 RL prompts, 2 shards x 3,000 rows, budget 3.5e7 tokens): all three stages run and `finalize_dataset.py` passes. With a universe that
small, 70% of the selected rows were repeats (no dedup); the full-size rate is not known until the full embedding exists.
