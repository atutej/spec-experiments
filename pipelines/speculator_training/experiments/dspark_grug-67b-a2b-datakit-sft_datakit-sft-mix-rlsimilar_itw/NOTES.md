# itw DSpark drafter trained on the Datakit rows nearest to the RL prompts ("rlsimilar")

Same drafter and recipe as the full itw run ([`../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/): 3 layers, FFN 3840, block 8, 32k draft vocabulary, lr 3e-4, loss 0.1 CE + 0.9 TV, 12,288 anchors, 32,768-token rows),
trained **from step 0** on a different 9.7B-token slice of the Datakit SFT mixture: the rows whose first turn is most similar to the Nemotron RL Ultra RLVR1 prompts, instead of a prefix of the whole mixture.
It is itw (existing text, no regeneration; see `docs/naming_itw_targetgen.md`). Compute equals the itw run's step 80,813 (9.70B tokens, 80,813 steps x 120,043 tokens), so the two are comparable.

## Why
The RL prompts are the use case (speculative decoding for Snowball RL rollouts), but the itw drafter was trained on a uniform slice of the whole mixture. Serving acceptance length of the full itw drafter in separate evaluations (different prompt sets and settings, so not one controlled comparison): 3.141 on the 256 RL prompts, 2.973 on first replies of mix validation prompts, about 2.2 on R2E-Gym agent first turns. This tests whether spending the same compute on the rows nearest to the RL prompts raises the RL-prompt number.

## Data creation (three stages; code in `tools/select_similar_rows/`, README there)
1. **Embed** (`embed` env, `vista/build_data.sbatch` + `embed_node.sh`, both gb nodes, 8 workers): `google/embeddinggemma-2`, bf16, symmetric prefix `task: sentence similarity | query: `, the text `system + "\n\n" + first user turn`
   of the 84,403 RL prompts (`RL_DATASET` train.parquet) and of every one of the 6,471,159 Datakit rows, **left-truncated to 8,192 embedder tokens**. For Datakit rows the first system header (reasoning mode + tools block) is skipped,
   and the RL rows' tool definitions (kept in `request_json`, not in the messages) are not embedded either. Measured on 2,150 rows: median 286 tokens, p99 33k, 8.6% over 8k; speed about 73k tokens/s per GPU, so about 4.8 h on the 8 GPUs of both nodes (estimate from the 2,000-row sample, untuned).
2. **Select** (same job, on the first node, `embed` env): exact top-64 neighbours of every RL prompt (GPU brute force, no index), then round-robin: round r takes every RL prompt's r-th nearest row until the budget `SELECT_TOKEN_BUDGET` (9.7e9
   packed tokens) is reached. The round the budget runs out in is filled with a seeded random subset of the RL prompts. No similarity cutoff, no supervised-token matching, **no dedup** (a row can be selected for several prompts and then appears
   several times; `--dedup` / `SELECT_DEDUP=1` skips taken rows; the selection summary reports the duplicate count either way).
3. **Write** (same job, `speculator` env): the selected rows, shuffled, as a `save_to_disk` dataset (`runs/grug_rlsimilar_data/dataset`), `token_freq.pt` / `manifest.json` recomputed from the SELECTED rows (the 32k draft vocabulary is built from the training data's supervised tokens, so it differs from the itw run's), then `finalize_dataset.py`.

All of it lives under `runs/grug_rlsimilar_data/{emb,sel,dataset}`; the Datakit dataset is read only. Every stage refuses a non-empty output directory.

## Training
Fresh work dir `runs/grug_dspark_itw_rlsimilar`, W&B run `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-rlsimilar_itw` in `atutej/marin_speculator`. `TOKEN_BUDGET=full` = one epoch of the filtered dataset (about 80.8k steps, about 11.7 h at 0.523 s/step on 2 nodes),
the learning rate decaying over that epoch (the itw run's schedule was cut at 21% of its epoch, so this one is annealed). Checkpoint every 1% of the epoch (about 808 steps), `CKPT_FREQ` in `settings.sh`: the itw value (0.1%) assumed a 382k-step epoch.
`submit_chain.sh` submits one 12 h job by default; the epoch plus startup is close to 12 h, so `JOBS=2` adds an `afterany` follow-up that resumes from the last checkpoint (not tried with a run that has already finished: check that case before relying on it).

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-rlsimilar_itw
# 1. smoke tests (on an allocation, not a login node): a small dataset through the real pipeline, then 10 training steps
bash vista/smoke_data.sh      # runs/grug_rlsimilar_smoke (3,000 RL prompts, 2 shards x 3,000 rows, 3.5e7 tokens)
bash vista/smoke_train.sh     # runs/grug_dspark_itw_rlsimilar_smoke
# 2. the full data creation: ONE job on both gb nodes (embedding on 8 GPUs, then selection and dataset; DATA_JOBS=n chains n resumable jobs)
bash vista/submit_data.sh --dry-run ; bash vista/submit_data.sh
#    when it is done read runs/grug_rlsimilar_data/sel/summary.json (duplicate_rows) before training
# 3. training: 2 nodes
bash vista/submit_chain.sh --dry-run ; bash vista/submit_chain.sh        # or AFTER=<data job id> to queue it behind the data job (afterok)
```

## Guards (lessons from the targetgen run)
`settings.sh` assigns every fresh path outright after sourcing the itw settings, recomputes the variables the itw settings derive from the itw run's values (`TRAIN_EXTRA`, `LOGGER_ARGS`) and resets `CKPT_FREQ` / `TRAIN_DATA_RATIO`. The smoke scripts refuse the full run's directories.
The only environment overrides are `RLSIM_BUILD_DIR` / `RLSIM_WORK_DIR`, names no other experiment uses.

## Open points
- **Duplicates.** With no dedup and a 0.1% validation split of the written dataset, a validation row can also be in training (it matters only if duplicates are frequent: the smoke run had 70% because its universe was tiny). The full-size rate is known after the selection job (`sel/summary.json`).
- Left truncation drops the instruction at the start of very long prompts (finance 10-K tasks, ARC grids); head + tail is the alternative if matching looks off.
- Ideas, not done: dedup across RL prompts; using the 40k RL rows that carry prefilled assistant / tool turns (multi-turn matching).
- Evaluation after training: as for the other drafters (`tools/eval_acceptance`, the 256 RL prompts, target 09.21), copy the final checkpoint into a new `runs/eval_ckpts/<name>/` first.
