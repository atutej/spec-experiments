# DSpark drafter for Qwen3-0.6B on the Nemotron-Terminal corpus's own completions (no regeneration)

Sibling of `../dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy` (read its `NOTES.md` for the corpus, the pipeline and the Vista
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

- `prepare-data` gets raw `conversations` here, so it needs a vLLM render endpoint (`/v1/chat/completions/render`) to apply the
  chat template (`render_endpoint is required to convert natural-language conversations ...`). **The hidden-state server of this
  vLLM build does not register that route** (a first version rendered through it and got `404 Not Found` for all 300 test
  conversations), but the build has `vllm launch render`, **a GPU-less render server** (no GPU, no weights, ~25-50 s to start).
  So `prepare` is a CPU stage: it starts that server, runs `prepare-data`, writes `$DATA_DIR.tmp`, renames it and stops the server.
  On a gg node (no NVIDIA driver at all) the server first failed with `Failed to infer device type`; `VLLM_TARGET_DEVICE=cpu` fixes
  that (the render server only tokenizes), and `run.sh` sets it. The CPU-platform render produced exactly the same dataset as the
  GPU node's (2,272 rows, identical content hash on 300 conversations).
- Qwen3's chat template strips the `<think>` block of every earlier assistant turn from the history. Checked on the real rendered
  rows: every row has exactly one `<think>`, it is inside the supervised part, and the history carries actions only.
- Per turn, the boundary is where the full render extends the generation-prompt render, so only the new turn is supervised.
  On 300 conversations: 2,272 rows (7.6 per conversation), 0 zero-supervised rows, no boundary errors.

## Measured on the 300-conversation and 3,000-conversation samples (2026-10-08, gb idev, render server run on that node)

| | corpus rows (this experiment) | regenerated rows (on-policy) |
|---|---|---|
| Rows per conversation | 7.6 | 5.8 |
| Mean row tokens (cap 16384) | 6,443 (2.7% at the cap) | 4,292 |
| Supervised tokens per row | 1,396 | 1,062 |
| Total tokens per conversation | 48,795 | 24,833 |

So an epoch has about **2x the tokens of the on-policy epoch** (~5-6 h at the on-policy run's ~8 steps/s, if the step time is similar;
the on-policy epoch is ~2.7 h).
`prepare` speed: 300 conversations took 40 s in total; 3,000 conversations took 317 s (62,465 render calls, ~20.8 per conversation)
with only 3 of 27 map workers busy (1,000 conversations per batch), and the render server (6 API servers x 2 workers) at ~170% CPU, far from
saturated. For 100k conversations (100 batches over 27 workers, ~2.1M render calls) a rough projection is 0.5-2.6 h; the gg limit is 2 days.

## Chain

`bash vista/submit_chain.sh` (login node) submits two jobs, time limits at the QOS maximum: one gg job running `export prepare`,
then `train` on gb (`afterok`). `export` here only links the on-policy experiment's 100k sample (a symlink, seconds) unless that
file is missing, so it shares a job with `prepare` instead of queuing separately. Only `train` needs a gb node, so only it waits in
the long gb queue (the on-policy run waited 3.6 h and 8.8 h for its two gb jobs; its gg jobs waited minutes). The qgb QOS allows
3 submitted jobs per user; this uses 1. Each argument of `submit_chain.sh` is one job and stages are joined with commas
(default `export,prepare train`; `export prepare train` makes three jobs; stages that need different partitions cannot share
a job). `AFTER=<jobid>` and subsets work as in the on-policy chain. The job name of the first job is
`dspark-offpolicy-export-prepare`.

Names: outputs go to `runs/nemotron_qwen3_0_6b_corpus_offpolicy_mooncake_100k/` (checkpoints, data, logs), and the Slurm jobs are
`dspark-offpolicy-<stage>`, so their logs are `logs/slurm/dspark-offpolicy-<stage>-<jobid>.out`. `submit_chain.sh` prints the
run directory it will use (it follows a `WORK_DIR=...` override).

## Expected size (estimates from 60 conversations; refine after a run)

See the measurements above: about 2x the on-policy run's training tokens per epoch.

## Metric logging (Weights & Biases)

`train` logs with `--logger wandb`: **entity `atutej`, project `marin_speculator`, run name = this experiment's folder name**
(`dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy`). The entity and project are set outright in `settings.sh`: this account's *default* entity is the team `dogml`, so
leaving it unset would send the runs there. Logging is off for smoke runs (`MAX_STEPS` set) unless `LOGGER=wandb` is given, and
`LOGGER= bash ...` turns it off. Local files go to `$WORK_DIR/logs/tracker/`.

Before training starts, `run.sh` checks the W&B login and that the entity is one of the account's (`wandb_preflight`). If that
fails (no key, no network, wrong entity) it logs **offline** instead of crashing, with a warning in the job log; upload later with
`wandb sync <run dir>`. `wandb` is installed in the `speculators` env by the recipe and checked by `env_check`.

Tested here: the pre-flight (login ok, wrong entity, bad key), the trainer's own `WandbHandler` offline (entity, project and run
name resolved as above), and that `speculators.train` accepts the flags and starts the W&B run. Not yet run end to end on a gb node.


## Status

Written 2026-10-08. Tested with `SAMPLE_LIMIT=300`: all three stages on a gb idev (export, prepare, 10 training steps, checkpoint, no
Mooncake retries), and export + prepare on a real gg idev (no GPU driver; render server ready in 27 s, 0 failed conversations,
dataset identical to the gb run). Not tested: the render step at full scale (projected 0.5-2.6 h, limit 2 days on gg), a full-length
run, and whether this data suits your target model (see above). Not submitted.
