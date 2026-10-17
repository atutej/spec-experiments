# DSpark drafter for Grug-67B-A2B (Datakit SFT 09-21), itw, on a 100k-row sample

A first, small experiment: the same drafter and pipeline as the full-mixture plan (`../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/PLAN.md`), trained on **100,051 rows (0.709B tokens, 0.306B
supervised, 32K context)** sampled from the Datakit SFT sources that had statistics on 2026-10-14. One job, ~4 hours of training, no resume needed.

## What it is

| | |
|---|---|
| Target | `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` (hidden 2,560, 26 layers, 256 experts, ~2B active) |
| Drafter | DSpark, 3 layers, FFN 3,840, hidden 2,560 (`draft_config.json`), draft vocab 32,000, Markov rank 256, block 8; 280M parameters saved, 198M trainable |
| Taps | layers 2, 12, 22 + the final layer (26) |
| Sequence length | 32,768 (`--total-seq-len` and the dataset's clip), `MAX_ANCHORS` 12,288 |
| Mask token | 128004 (`<|finetune_right_pad_id|>`; Grug's tokenizer has none) |
| Data | Datakit SFT sample: 87 sources (shard 1 of the first Datakit run), token shares equal to Grug's allocation shares; clipping at 32K removed 11.3% of the sampled tokens (91M) and dropped 250 rows whose supervised tokens all lie past 32K |
| Mode | itw (in-the-wild): the Datakit completions stand in for Grug's own; the server only does prefill with hidden-state extraction (`--on-missing generate` = generate hidden states) |

## Run it

1. **Dataset** (built, at `/scratch/09749/atutej/marin_datakit/dataset_100k_seq32768`; `bash data/make_dataset.sh` rebuilds it): `select --rows 100000` -> `write --seq-length 32768` -> `finalize_dataset.py`,
   with the tools of the full-mix folder (`../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data/`). It needs `fetch` + `stats` for the sources in the list (done for the 87 sources of `shards/shard-1-of-2.txt`).
2. **Train** (from a login node): `bash vista/submit_chain.sh` (`--dry-run` first). One job on a gb node of the CGAI-gb reservation, 12 h; the script refuses to submit without the reservation. Uploads the best and
   latest checkpoint to a public Hub repo named `<this folder>-<date>` afterwards (`HF_REPO= ` turns it off). Resume after a time limit: `AFTER=<jobid> bash vista/submit_chain.sh train`.
3. **Smoke test** (idev, 4 GPUs): `DATA_DIR=/scratch/09749/atutej/marin_datakit/write_test MAX_STEPS=10 WORK_DIR=<dir> bash vista/run.sh train`.

The chain has no `prepare` stage: the dataset is already tokenized and masked. `speculators` is the main checkout `speculators/` on branch `grug-dspark-gate` (gated verifier head in the DFlash/DSpark targets, `grug_moe` config registration, a `launch_vllm.py` that works without Transformers
knowing the model type); the env's editable install points at it. `run.sh` checks for the patch before starting the server and stops with a message on any other branch. (The first run, 2026-10-14, used a separate
worktree on `PYTHONPATH`; that worktree has been removed and the main checkout switched.)

## Smoke test, 2026-10-14 (idev, 4 GPUs, 4,146-row dataset from 3 sources, 10 steps)

- Server (TP=1, DP=2, expert-parallel, GPUs 0,1) ready after 250 s; Mooncake 32 / 16 GiB; trainer on GPUs 2,3; the gated head was enabled.
- 10 steps, a checkpoint, a validation pass, clean shutdown (about 6 minutes after the server was up). Train loss 1.255 -> 0.947; accept length ~1.0 (untrained, as expected).
- Throughput at step 9: 54.9K tokens/s, 2.3% of the step spent waiting for hidden states (36% at step 0). Memory: ~178 GB on the server GPUs, ~71 GB on the trainer GPUs (of 189 GB).
- The checkpoint records `DSparkDraftModel`, `mask_token_id` 128004, taps [2, 12, 22], draft vocab 32,000, verifier `GrugMoeForCausalLM`.

## Expected for the real run (estimates)

~0.709B tokens at ~55K tokens/s: about 3.6 h of training plus the validation pass (~10% of the rows: ~20 min) and the 4-min server start. **Not measured at this scale**: sustained throughput, Mooncake behaviour over hours,
memory with the long rows of `open_swe_traces` / `openthoughts4-code`, and the accept length it reaches.

## Not done / open

- The branch `grug-dspark-gate` is pushed to the fork (`atutej/speculators`) but not merged into its `main`; `setup/repos.txt` points at it.
- Serving the trained drafter in vLLM (the fork has a Qwen3-based `dspark` method) and measuring acceptance with Grug generating: untested.
- The sample covers 87 of the 203 sources: not the 26 unfetched ones (multilingual, science, opencode, math proofs / v4, swe_v2) nor the 90 fetched sources that have no stats yet.
