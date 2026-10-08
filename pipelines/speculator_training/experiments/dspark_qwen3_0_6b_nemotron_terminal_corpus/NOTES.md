# DSpark drafter for Qwen3-0.6B on the Nemotron-Terminal corpus's own completions (no regeneration)

Sibling of `../dspark_qwen3_0_6b_nemotron_terminal` (read its `NOTES.md` for the corpus, the pipeline and the Vista
learnings). That experiment trains on Qwen3-0.6B's own regenerated responses (on-policy). This one trains on the
corpus's original completions instead, so the two runs differ only in the completions:

|  | on-policy (`..._nemotron_terminal`) | corpus completions (this one) |
|---|---|---|
| Conversations | 100k, seed 0 | the same file (linked, not re-exported) |
| Completions | Qwen3-0.6B's, regenerated turn by turn | DeepSeek-V3.2's (terminus-2 agent, thinking on) |
| Rows | one per assistant turn (prefix + new turn) | one per assistant turn, built by `prepare-data` via `/render` |
| Sequence length / anchors | 16384 / 6144 | 16384 / 6144 |
| Everything else (DSpark settings, LR, Mooncake, resume) | same | same |

## Why (and what it depends on)

Whether corpus completions are the right data depends on the model you will serve. Base Qwen3-0.6B writes differently
from DeepSeek-V3.2, so the on-policy data matches it better. A Qwen3-0.6B fine-tuned on this corpus (as the
Nemotron-Terminal models are) would produce text close to the corpus, and then these completions are near on-policy
for it. Note the hidden states used in training come from the target model named in `settings.sh` (base Qwen3-0.6B),
not from a fine-tuned one: for a fine-tuned target, change `MODEL`.

## How it works

- `prepare-data` gets raw `conversations` here, so it needs a live vLLM server for the target model to render them
  (`render_endpoint is required to convert natural-language conversations ...`). The `prepare` stage therefore runs on a gb
  node: it starts `mooncake_master` and the hidden-state server, renders through it, writes `$DATA_DIR.tmp` and renames it.
- Qwen3's chat template strips the `<think>` block of every earlier assistant turn from the history (checked on 298
  rendered rows: always exactly one `<think>`, the supervised turn's own). That turn's reasoning is supervised in its own row.
- Per turn, the boundary is where the full render extends the generation-prompt render, so only the new turn is supervised.
  Checked locally with the tokenizer on 60 conversations: 0 of 414 rows had an unstable boundary.

## Chain

`bash vista/submit_chain.sh` (login node) submits `export` (gg) -> `prepare` (gb) -> `train` (gb), time limits at the QOS
maximum. The qgb QOS allows 3 submitted jobs per user, so this plus the on-policy `train` job fits only if nothing else
is queued on gb (an idev counts). `AFTER=<jobid>` and subsets work as in the on-policy chain.

## Expected size (estimates from 60 conversations; refine after a run)

About 6.9 rows per conversation (plus ~8% of turns skipped because their context alone fills 16384 tokens), mean row
~6.2k tokens, ~42k tokens per conversation against ~25k for the regenerated rows: about 1.7x the training tokens.
At the on-policy run's measured ~8 steps/s that is roughly 4-5 h for one epoch, on top of the render time of `prepare`
(about 1.5-2 render calls per turn, ~1.5M for 100k conversations; not timed).

## Status

Written 2026-10-08. `export` tested here on a CPU node (links the shared sample; exports when none exists). Not yet
tested: `prepare` and `train` on a gb node (needs a 4-GPU idev: `SAMPLE_LIMIT=300 MAX_STEPS=10 WORK_DIR=<abs path> bash vista/run.sh`),
including that `prepare-data` accepts the corpus rows with their extra columns, how `/render` behaves on this data, and the
time of the render step. Not submitted.
