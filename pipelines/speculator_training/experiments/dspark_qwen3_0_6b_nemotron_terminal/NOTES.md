# DSpark drafter for Qwen3-0.6B on Nemotron-Terminal-Corpus

Scripts: `genai/run.sh` (the genai launch script) and `vista/` (Vista smoke tests and sbatch scripts). Run outputs go to `$PROJECT_ROOT/runs/`, not here.
It's launch-ready on genai but hasn't been run anywhere.

It trains a DSpark speculative-decoding drafter for Qwen/Qwen3-0.6B (thinking on) on
**on-policy** data: Qwen3-0.6B rewrites every assistant turn of 100k randomly sampled
conversations from `nvidia/Nemotron-Terminal-Corpus` (subset `dataset_adapters`, about 226k
terminal-agent trajectories covering code, math and SWE).

The pipeline:
- **Step 0:** export a seeded sample with `export_registry_dataset.py`.
- **Step 1:** `vllm serve` plus `speculators regenerate-responses` on the sample.
- **Step 2:** `prepare-data`.
- **Step 3:** hidden-state server (`launch_vllm.py`, layers `2 14 25`).
- **Step 4:** online DSpark training over Mooncake.

On genai, steps 1 and 3–4 used four GPUs and two plus two GPUs.

Settings agreed with the user: `MAX_GEN_TOKENS=8192` (= `SEQ_LENGTH`),
`REGEN_MAX_MODEL_LEN=32768`, Qwen's thinking-mode sampling, `SAMPLE_SEED=0`,
`REGEN_LIMIT=100000`, and the DSpark settings in the script.

Learned on genai:
- **`datasets` cannot read this corpus.** Each parquet file is a single row group with 2–7 GB
  of nested text, and Arrow fails with `Nested data conversions not implemented for chunked
  array outputs`, in any mode or batch size. Step 0 reads with `pyarrow` `iter_batches`, then
  uses `Dataset.from_generator`, HF `.shuffle(seed)`, `.select` and `.to_json`, copying rows
  verbatim and unifying the files' columns (`code.parquet` alone has `source`). Its cache is
  about 13 GB.
- **Context length:** at `max-model-len 12288`, 15% of conversations failed with HTTP 400,
  because the next terminal-output turn can push a prompt past `MAX_GEN_TOKENS`. 32768 fixes
  it.
- **Truncation is expected:** about 43% of conversations stop past 8192 tokens. Their last
  row can exceed 8192 and gets clipped by `prepare-data`, losing about 4% of supervised
  tokens. The user accepted this, and also accepted that regenerated turns may react to
  terminal output from commands the model didn't run.
- **`--resume` in step 1** relies on an unchanged step 0 file (same seed, subset and limit).
  `prepare-data --overwrite` refuses directories holding other files.
- Read regenerated JSONL by iterating over the file object, never with `str.splitlines`
  (U+2028 appears inside records).

Reference results:
- **Step 0 with `--limit 300 --seed 0`:** 43 rows from code, 218 from math and 39 from swe,
  all identical to their source rows. The first three `(trial_name, episode)` values are
  `(task_90527__CMJA82Z, episode-8)`, `(task_101944__NX3GDrK, episode-4)` and
  `(task_17047__7eZ59DU, episode-10)`. The file's sha256 was
  `32f16db6c70dbc1f7a7563942b8bacf52747f26d678f1f7fcfb65326fd71ae68`.
- **Step 1 on the first 100 rows** (one H100 at memory cap 0.12, concurrency 32): 0 failed,
  43 truncated, 563 training rows, a median of 6 turns per conversation, a median row of
  3,743 tokens and a median reply of 500 tokens. It ran at about 1.9 requests/s. It's
  stochastic, so expect similar numbers rather than identical ones.
- **Steps 2–4** have not been run on this dataset. The same pipeline on Open-PerfectBlend
  (an earlier run) trained in about 35 minutes on two GPUs and reached a validation accept
  length of 3.99.
- **Scale:** about 560k regeneration requests (on genai, perfectblend did about 15 requests/s
  on four H100s), and about 560k training rows.

On Vista:
- **Smoke tests** run in a GPU idev session.
  - On **gb**, the script's genai layout fits one node: step 1 data-parallel on 4 GPUs, and
    steps 3–4 on 2 + 2, probably with only `GPU_MEM_UTIL` revisited.
  - On **gh**, use a one-GPU variant. Step 3's hidden-state server and step 4's
    `torchrun --nproc_per_node 1` share the GPU, with vLLM memory capped at roughly 0.3
    (a guess).
- **The full 100k run** is always an `sbatch` job, possibly across several gh or gb nodes.
  Its configuration (node type and count, how step 1 is split, how steps 3–4 are placed,
  time limits) **is not decided yet**. Work it out with the user in the Vista session, then
  record it here.

## Vista smoke-test log

- **Step 0 (2026-10-07, gb node):** `vista/smoke_step0_export.sh` (`--limit 300 --seed 0`) reproduced
  the genai reference exactly: same sha256 and same first three `(trial_name, episode)` values.
  The run took 2 min and cached the whole corpus in `$HF_HOME` (19 GB, more than the ~13 GB guessed).
- **Steps 1-4:** not run yet.
