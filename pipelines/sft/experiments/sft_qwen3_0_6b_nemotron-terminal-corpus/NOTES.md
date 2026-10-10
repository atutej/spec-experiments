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
- `convert` (alternative to `prepare`, writes the same `$WORK_DIR/data`): `bash vista/submit_chain.sh export,convert train` makes the
  dataset from the speculator pipeline's prepared data with `tools/convert_prepared_data.py` (below): no tokenization, minutes.
- **Prebuilt data:** `PREBUILT_DATASET=<dir>` trains on an existing dataset of that format and skips prepare. To use the speculator
  pipeline's prepared data (one tokenized row per assistant turn, already made): `python tools/convert_prepared_data.py --prepared
  <run>/data --out DIR` (drops the rows clipped at 16384; validation = random rows, not by conversation).
- W&B is off (`REPORT_TO=none`) until its project and naming are decided; `REPORT_TO=wandb` turns it on (entity atutej).

## Status (2026-10-10)

End to end through `run.sh` on a gb idev node (4 GPUs), `export convert train` with SAMPLE_LIMIT=300, 6,000 prepared rows (5,869 kept after
dropping 131 clipped at 16384), global batch 64 (2 per device x 8 accumulation x 4 GPUs), 10 steps: ran, eval loss 1.067 (step 5) ->
1.012 (step 10). Resume: rerun with MAX_STEPS=14 continued from `checkpoint-10` ("global step 10", fast-forwarded 80 batches), saved
`checkpoint-14`. The final model at `$WORK_DIR/model` loads and serves with vLLM and answers corpus prompts in the corpus format
(`<think>` then JSON with analysis/commands) after 14 steps. Bug found and fixed on the way: the converter inherited the prepared data's
saved format `torch`, which crashed the training dataloader workers (datasets 4.0.0's torch formatter imports torchvision.io.VideoReader,
removed in torchvision 0.28; datasets 5.x fixes it but LLaMA-Factory refuses it); `convert_prepared_data.py` now resets the format.

Throughput (rough, from steps 6-10 of that run): about 8 s per step at global batch 64, i.e. ~54k tokens/s on 4 GPUs, a few percent of the
GPUs' peak. One epoch is ~772k examples = ~12k steps = ~27 h at that rate: more than one 12 h gb job, so the train stage must be chained
(each job resumes from the last checkpoint; save_steps 500 is ~1.1 h) unless throughput is improved first.
Not done yet: the full 100k convert/prepare run, throughput tuning (per-device batch, gradient checkpointing, padding waste), hyperparameters
(all placeholders in settings.sh: lr 2e-5, global batch 64, 1 epoch).

Known properties of the data: about 12% of the per-turn examples (the later turns, whose context alone is too long) exceed 16,384
tokens and are dropped; the rest is ~5B tokens. No packing: LLaMA-Factory packs during its own tokenization, which the pre-tokenized
path skips, so batches are padded (watch the waste when tuning `PER_DEVICE_BATCH`).
