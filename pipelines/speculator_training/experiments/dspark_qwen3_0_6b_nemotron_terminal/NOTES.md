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

## Vista full run (100k conversations)

Files: `settings.sh` (shared settings, identical to the genai script's config block on all 39 shared
variables), `vista/run.sh` (the pipeline as stages), `vista/run.sbatch` (one stage per job),
`vista/submit_chain.sh` (submits the chain from a login node). Policy: the `.sh` runs as-is on idev, sbatch wraps it.

- **Stages:** `export` (gg, CPU) -> `regen` (gb, 4 GPUs) -> `prepare` (gg, CPU) -> `train` (gb: hidden-state
  server on GPUs 0,1, training on GPUs 2,3, as on genai). Each waits for the previous (`afterok`); time limits
  are the QOS maxima (gb 12 h, gg 2 days). The chain uses 2 of the 3 gb submit slots.
- **Submit:** on a login node, `bash pipelines/speculator_training/experiments/dspark_qwen3_0_6b_nemotron_terminal/vista/submit_chain.sh`
  (`--dry-run` first prints the commands). A subset reruns one stage: `... submit_chain.sh train` resumes training
  from `$WORK_DIR/checkpoints` (the trainer resumes by default); `regen` resumes with `--resume`.
- **Settings:** same as genai (`MAX_GEN_TOKENS=8192`, `REGEN_MAX_MODEL_LEN=32768`, thinking-mode sampling, seed 0,
  `REGEN_LIMIT=100000`, DSpark settings). Only performance knobs differ: regen concurrency is 1024 on Vista (genai
  512), because a gb GPU has 189 GB against an H100 NVL's 94 GB. **Not tuned:** check `rps` in the regen log
  of the first full run.
- **Estimates (extrapolated from smoke tests, order of magnitude):** ~563k rows, ~2.35 G training tokens; regen
  ~6 h on one gb node (24.5 rows/s measured at concurrency 256 on 300 conversations); training ~6.5 h for one
  epoch (from ~1e5 tokens/s over the 10 smoke steps, so shaky). Both fit the 12 h gb limit, and both resume.
- **Tested on idev (2026-10-07):** all four stages with `REGEN_LIMIT=300 MAX_STEPS=10`: export, regen (300
  conversations, 0 errors, 138 truncated, 62 s at concurrency 1024), prepare, train (stopped at the 10-step limit,
  checkpoint written, no tracebacks). The sbatch wrapper and chain were checked with `--dry-run` only: no job submitted yet.
- **Not done:** backup of results to `$WORK` (deferred), the `gh` variant, a first real submission.

## Tuning on a gb node (2026-10-07, `vista/tune.sh gen train`)

Performance-only knobs, on small data (the 3,000-conversation sample). The number of training GPUs stays 2
(it sets the effective batch size, so changing it would change what is trained). Results lines are in
`logs/smoke/tune-results.txt`; the script was written by another Claude session, which I found in the repo and used unchanged.

| Knob | Values | Result (4x GB200) |
|---|---|---|
| Regen client concurrency | 256, 512, 1024, 2048, 4096 | 29.7, 29.3, 32.3, 32.0, 30.3 rows/s: flat (within slice noise) |
| vLLM `--max-num-seqs 2048 --max-num-batched-tokens 32768` | at concurrency 4096 | 32.6 rows/s: no gain over defaults |
| Training dataloader workers / prefetch | 12/4 vs 24/8 | 90.6k vs 89.9k tokens/s: no gain |
| Mooncake global/local buffer | 4/2 vs 32/8 GiB | 90.6k vs 88.5k tokens/s: no gain |
| Hidden-state server GPUs | 2 vs 1 | 90.6k vs 86.9k tokens/s (-4%): one server GPU is nearly enough |

- **Training is compute-bound on the 2 training GPUs:** `fetch_frac` is only ~7% of a 77 ms step, so nothing on the
  Mooncake or server side speeds it up. The baseline settings are as good as the variants; keep them.
- **Regeneration saturates the 4 GPUs at or below ~256 concurrent requests** (~30 rows/s). `vista/run.sh` keeps
  `REGEN_CONCURRENCY=1024`: harmless, not faster. **Caveat:** each trial used only 400 conversations, and a
  conversation's turns run one after another, so at concurrency above ~400 everything is already in flight and
  the 1024-4096 trials mostly repeat each other. A longer test with more conversations than the concurrency
  would show whether 1024 helps at 100k scale; the flat 256 vs 512 result suggests not.
- **No change made to `vista/run.sh` from this pass.** The only idea left is freeing a GPU (the server is
  nearly as fast on one), but nothing else can use it without changing the training setup.

### GPU utilization and packed sequence length (2026-10-07; `vista/profile_gpu.sh`, `vista/tune_seqlen.sh`)

A training step is one packed sequence of `--total-seq-len` tokens per GPU (no batch-size flag; effective batch =
2 GPUs x length). At the genai setting (8192) the two trainer GPUs use only **19 GiB of ~183 (10%)** and are ~70% busy
when active; the two hidden-state-server GPUs hold ~168 GiB (the preallocated KV cache) but are only ~25% busy.
Sweeping the length on the same 13,885 rows, 100 steps each (`--max-anchors` scaled with the length; LR unchanged):

| `--total-seq-len` | tokens/s | step | trainer mem | busy util | rows clipped | supervised tokens lost | Mooncake retries |
|---|---|---|---|---|---|---|---|
| 8192 (genai) | 87k | 77 ms | 19 GiB | 74% | 1,052 / 13,885 | 4.90% | 0 |
| 16384 | 125k (+44%) | 120 ms | 31 GiB | 77% | 3 | 0.02% | 0 |
| 32768 | 147k (+69%) | 215 ms | 60 GiB | 96% | 0 | 0% | 325 |

- **This changes what is trained** (tokens per step, the clipping of long rows, steps per epoch), so it is not part of the
  genai-identical settings. Not evaluated: loss or acceptance quality, the right LR for a larger step.
- **At 32768 Mooncake rejected puts** (`status=-800`, 325 retries, a few exhausting 3 attempts): the 4/2 GiB buffers are too
  small for that many long samples in flight. Needs larger buffers (e.g. 32/8 GiB, untested at this length).
- **Epoch estimates** (2.35 G tokens): ~7.5 h at 8192, ~5.2 h at 16384, ~4.4 h at 32768, with 1x, 0.5x, 0.25x the optimizer steps.

## Vista smoke-test log

- **Step 0 (2026-10-07, gb node):** `vista/smoke_step0_export.sh` (`--limit 300 --seed 0`) reproduced
  the genai reference exactly: same sha256 and same first three `(trial_name, episode)` values.
  The run took 2 min and cached the whole corpus in `$HF_HOME` (19 GB, more than the ~13 GB guessed).
- **Step 1 (2026-10-07, gb node, 1 GPU, `vista/smoke_step1_regen.sh`, first 100 rows):** 0 failed, 41
  truncated, 569 rows (genai: 0 / 43 / 563), median 5.5 turns per conversation (6), median row 3,993
  tokens (3,743), median reply 521 tokens (500). 91 s, ~6.2 rows/s (genai ~1.9 requests/s on a shared
  H100 at 12% memory; not comparable). The 4-GPU layout is tested below.
- **Step 1, real 4-GPU layout (`DP=4 LIMIT=300 CONCURRENCY=256 GPU_MEM_UTIL=0.9 vista/smoke_step1_regen.sh`,
  `--data-parallel-size 4`, whole 300-row sample):** server ready in 141 s; 300 conversations, 0 errors,
  133 truncated (44%), 1,673 rows; median 6 turns, row 3,620 tokens, reply 519 tokens. 68 s of regeneration,
  24.5 rows/s, 3.94x the 1-GPU run (6.2 rows/s): near-linear data-parallel scaling. Rough extrapolation
  for the full 100k conversations on one gb node: hours (about 6 h, order of magnitude only; plan the run
  with a job array).
- **Step 2 (`vista/smoke_step2_prepare.sh`):** `prepare-data` on those 569 rows took 43 s, dropped 1
  row with no supervised tokens, flagged a few rows clipped at 8192 (expected; see the learnings above).
- **Steps 3-4 (`vista/smoke_step3_4_train.sh`, gb node, genai layout: hidden-state server on GPUs 0-1
  with data-parallel 2, training on GPUs 2-3):** `mooncake_master` (tcp) and the hidden-state server
  came up (207 s, compile cache warm); `speculators.train` ran 10 steps (`--max-steps 10`, 2 GPUs),
  fetched hidden states through Mooncake with 0 error records (~12 ms per step, ~1e5 tokens/s), ran a
  validation epoch and wrote `checkpoint_best` (`model.safetensors`, 525 MB with optimizer state).
  Exit code 0 in 156 s. Loss and accuracy after 10 steps at warmup LR are meaningless
  (`val/eal` 1.0, accuracies ~0); this only shows the pipeline runs end to end.
- **Noise at exit (fixed):** with `TMPDIR` on shared Lustre, Python's multiprocessing printed
  `OSError: [Errno 16] Device or resource busy` for `$TMPDIR/pymp-*` while cleaning up (exit code
  still 0). Vista now uses node-local `TMPDIR=/tmp`; the rerun of steps 3-4 (89 s, exit 0) had no
  such errors and no tracebacks.
- **Not tested:** `gh` nodes (1 GPU: the one-GPU variant, with the hidden-state server and training
  sharing the GPU), multi-node, a real-length training run.
