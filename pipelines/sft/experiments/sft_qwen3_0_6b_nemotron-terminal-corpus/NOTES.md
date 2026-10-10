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

Throughput tuning (`vista/tune.sh`, 4 GPUs of a gb node, global batch 64, 40 steps each unless noted, same data and seed; step time from
step 3 on). The data is 5,869 examples of the prepared set (shuffled, so the same length mix as the full set):

| per-device batch | gradient checkpointing | batches | s/step | samples/s | peak GiB/GPU |
|---|---|---|---|---|---|
| 2 | on | random | 6.03 | 10.6 | 60 |
| 2 | on | grouped by length | 4.54 | 14.1 | 60 |
| 4 | on | grouped by length | **3.24** | **19.7** | 109 |
| 4 | on | random (8 steps only) | 7.0 | 9.1 | 111 |
| 8 | on | random | out of memory (a 71.5 GiB allocation) | | |
| 2 / 4 | off | random (8 steps only) | 6.4 / 7.0 | 10.0 / 9.1 | 172 / 108 |

Padding is the bottleneck: turning gradient checkpointing off changes nothing, a larger batch alone does not help, and grouping batches by
length does (an 8-step run showed 2.6 s/step for batch 4 grouped; 8 steps were too few, the 40-step figure is 3.24). Defaults are now
per-device batch 4 + `group_by_length`, and global batch 32 = 4 per device x 2 accumulation x 4 GPUs (the table is at global batch 64 = 4 x 4 x 4; the
step time should be about half at 32, the epoch time the same, with twice as many optimizer steps: ~24k; evals and saves every 1000 steps). At 3.24 s/step the full epoch (~772k examples = ~12k steps) is ~11 h plus evals and saves: it
barely fits one 12 h gb job, so chain a second `train` job (it resumes). The step time is still only ~5% of the GPUs' peak: attention
with a padding mask is probably the next limit (packing with varlen attention would avoid the mask, not done). The grouped sampler computes
all lengths at startup by iterating the dataset: a few minutes expected on 772k examples, not measured. Grouping makes the examples of a
step similar in length, which changes the batch statistics somewhat compared with random batches.

Not done yet: the full 100k convert/prepare run, hyperparameters
(all placeholders in settings.sh: lr 2e-5, global batch 32, 1 epoch).

Known properties of the data: about 12% of the per-turn examples (the later turns, whose context alone is too long) exceed 16,384
tokens and are dropped; the rest is ~5B tokens. No packing: LLaMA-Factory packs during its own tokenization, which the pre-tokenized
path skips, so batches are padded (watch the waste when tuning `PER_DEVICE_BATCH`).
