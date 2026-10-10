# sft_qwen3_0_6b_nemotron-terminal-corpus

Full fine-tune of `Qwen/Qwen3-0.6B` on the Nemotron-Terminal corpus's own completions (DeepSeek-V3.2 terminus-2 trajectories), the SAME
100k conversations as the speculator pipeline. One training example per assistant turn; loss only on that turn's thinking and response;
earlier turns' thinking is stripped from the context. The result is meant to be the target model of a new drafter experiment
(`pipelines/speculator_training`, off-policy, target = this checkpoint). Decisions: `../../README.md`.

## Run

```
bash vista/submit_chain.sh [--dry-run]     # from a login node: gg job (export, prepare), then gb job (train), 4 GPUs
bash vista/run.sh [export] [prepare] [train]   # as-is on an idev node
```

- `export`: links the speculator pipeline's 100k sample (`runs/nemotron_qwen3_0_6b_regen_online_think_mooncake_100k/source/...`), else
  exports it with the same script; always checks its sha256 (`791b7f07...`, in settings.sh).
- `prepare`: `tools/build_sft_dataset.py` -> `$WORK_DIR/data` (a `DatasetDict` with train and validation; the validation conversations
  are chosen by hash, so none has turns on both sides). Examples over 16,384 tokens are dropped, not truncated.
- `train`: `llamafactory-cli train` from the generated `$WORK_DIR/train_config.yaml`; output `$WORK_DIR/model` (`checkpoint-N/` while
  training, the final HF model at its root); an existing `checkpoint-N` is resumed.
- **Prebuilt data:** `PREBUILT_DATASET=<dir>` trains on an existing dataset of that format and skips prepare. To use the speculator
  pipeline's prepared data (one tokenized row per assistant turn, already made): `python tools/convert_prepared_data.py --prepared
  <run>/data --out DIR` (drops the rows clipped at 16384; validation = random rows, not by conversation).
- W&B is off (`REPORT_TO=none`) until its project and naming are decided; `REPORT_TO=wandb` turns it on (entity atutej).

## Status (2026-10-10)

Smoke-tested end to end on the first 300 conversations (gb idev, 4 GPUs, 3 steps, global batch 8, per-device batch 1): export, prepare,
train all ran, `checkpoint-3` and the final model were written; train loss 1.74, eval loss 0.975 (3 steps: meaningless as a result).
Not done yet: the 100k prepare (time and size unknown), throughput and per-device batch tuning, resume test of training, serving the
saved model with vLLM, hyperparameters (all placeholders in settings.sh: lr 2e-5, global batch 64, 1 epoch).

Known properties of the data: about 12% of the per-turn examples (the later turns, whose context alone is too long) exceed 16,384
tokens and are dropped; the rest is ~5B tokens. No packing: LLaMA-Factory packs during its own tokenization, which the pre-tokenized
path skips, so batches are padded (watch the waste when tuning `PER_DEVICE_BATCH`).
