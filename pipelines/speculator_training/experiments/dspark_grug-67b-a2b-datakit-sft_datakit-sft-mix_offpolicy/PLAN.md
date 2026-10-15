# DSpark drafter for Grug-67B-A2B (Datakit SFT, 2026.09.21), trained offline on the Datakit SFT mixture

Status: plan only (2026-10-13). Nothing in this folder runs yet. Sibling to
`../dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_offpolicy` (same pipeline shape: vLLM hidden-state server
-> Mooncake -> online drafter training); read its `NOTES.md` for the pipeline and the Vista learnings.

## Fixed decisions

| | |
|---|---|
| Target (verifier) | `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` (`grug_moe`): hidden 2560, 26 layers, 20 heads / 5 KV, head_dim 128, vocab 128,256, 256 experts top-4 + shared expert, ~2B active, sliding window 2048 in 19 of 26 layers |
| Drafter | DSpark, **3 layers**, FFN `intermediate_size` = 1.5 x hidden = **3840**, hidden = 2560 (must equal the verifier's), draft vocab 32,000 (revisit 16k), est. ~250M params. `--draft-config` is a `qwen3` decoder config (DSpark inherits DFlash) |
| Sequence length | **32,768** (decided 2026-10-13; the sibling uses 16,384). `--seq-length` (prepare clip) and `--total-seq-len` (packed training sequence) are the same number in `speculators`; `MAX_ANCHORS` scales with it at 0.375 per token (3072 at 8K, 6144 at 16K), so **12,288**. See "Sequence length" below |
| Other DSpark settings | start from the sibling's: block size 8 (maybe 6), Markov rank 256, vanilla head, loss 0.1 CE + 0.9 TV, LR 3e-4. Taps: 3 spread over 26 layers (e.g. 2, 12, 22) + the final layer, to be confirmed |
| Data | the exact Datakit SFT mixture used for the 09-21 SFT: 203 sources, 53.7B SFT tokens (the 80% SFT part; no pretraining replay) |
| Mode | offline: Datakit completions stand in for Grug completions (no rollouts). Hidden states extracted online, as in the sibling |
| Compute | 2 x GB200 nodes (8 GPUs): ~6 GPUs serve the target, ~2 train the drafter (to be measured) |

## Where things are

- Mix definition: `experiments/grug_sft/special_token_1000_mix.json` in Marin (commit `82620ad62bbe23b9b12a89d656c24e6f80aff91a`, branch
  `held/grug-sft-special-token-1000`): per-source token allocations, `dropped_sources`, `sft_fraction` 0.8. Same numbers as `training_data_mix.json` on the
  model repo.
- Marin clone and its env (separate from the workspace envs): `/scratch/09749/atutej/marin_datakit/{marin,venv,uv_cache}`. Built with
  `uv sync --package marin-core --python 3.12 --no-dev` plus `uv pip install -e marin/lib/dupekit`. (Without `--no-dev`, `pycosat` fails to build.)
  `all_sft_sources()` registers 249 sources and all 203 allocated ones resolve.
- Per-source steps live in `lib/marin/src/marin/datakit/download/<source>.py`: `download_hf_step` (pinned revision) -> source-specific transform with its keep/drop
  rules -> `normalize_chat_step` (Harmony messages, exact-id dedup) -> render with the Marin chat template -> tokenize. We want the chat-normalized output, before rendering and tokenizing.
- Which Marin commit holds the data code (checked 2026-10-13): the 1,000-step run was added on 09-20 on `held/grug-sft-special-token-1000-cp-fix` (`8a0ec6cf`), which does not have the final SFT registry. The registry and the token-store
  pipeline were committed on 09-21 on `held/grug-sft-special-token-1000` (`8c3be8f` "Register the current chat source inventory", `add0379` "Add the current token-store build pipeline"; branch head `82620ad`). Only that lineage contains
  `ultrachat-persona-conversations`, which is in the model's allocations, so **`82620ad` is the commit to use**; all 203 allocated sources resolve there. `special_token_1000_mix.json` is identical on both branches.
- The mix is 203 leaf sources in 15 families (123 penfever-traces, 62 nemotron_sft_v3, 6 open_swe_traces, 12 single-source families); the registry has 249 sources and the mix JSON excludes 15 by name.
- Data scripts (this folder, `data/`; run with the datakit venv): `plan_datakit_sft_mix.py` (resolve every source to HF repo + pinned revision + size; no download), `build_datakit_sft_mix.py` with `fetch`, `stats`, `select`
  (below), `fetch.sbatch` + `submit_fetch.sh` (one gg CPU node, 2 days, fetch -> stats -> select; submit from a login node). Plan result: ~628 GB across 160 unique downloads (486 GB Nemotron v3, 38 GB penfever, 26 GB
  synthetic-1, 25 GB superior-reasoning, 23 GB open_swe_traces, 15 GB coderforge, 8 GB nemotron-terminal, ...). `massive_function_calling` has no Hub download step (its source, `AmazonScience/massive`, is fetched
  differently); check it separately.

### Running the Datakit command on the reservation's gb nodes (2026-10-13)
gg had no start in sight, so the Datakit command can run on the two `CGAI-gb` reservation nodes (`c672-006`, `c672-007`, until 2026-10-27, user atutej):
`PARTITION=gb SHARDS=2 bash data/submit_fetch.sh` (from a login node; `--dry-run` prints the sbatch lines).
- **Sharding**: `build_datakit_sft_mix.py shards --n 2` splits the 203 sources into balanced lists (sources sharing a Hugging Face download stay together; packed by download size and estimated tokens):
  shard 0 = 116 sources, 324 GB, ~51B tokens; shard 1 = 87 sources, 304 GB, ~55B tokens (`plan.json` in `data/` is the input; lists go to `/scratch/09749/atutej/marin_datakit/shards/`). Each shard job runs `fetch` and `stats`
  for its list, then `select --only-if-complete`: whichever shard finishes last writes `selection.json`. No third job is needed (the gb QOS allows 3 submitted jobs per user; an idev counts as one).
- **Limits**: 12 h per gb job (everything resumes: a rerun skips finished Marin steps; `ONLY=1` resubmits just shard 1); the nodes are `--exclusive`, so while the shards run no GPU job (serving test, training) can use the reservation; this idev
  (c672-006) holds one node until it ends or is cancelled, so the second shard waits for it.
- **Bugs found while testing the sharded job script**: `set -u` before `source ~/.bashrc` aborted the job silently (fixed in `fetch.sbatch`, `write.sbatch`); `WORKERS` computed as 0 on a host with 1 CPU (fixed). Tested end to end on a
  3-source shard (fetch skipped, stats + mask checks pass, `select --only-if-complete` correctly waits).

### First run of the Datakit command on the gb nodes: measured, and the resume (2026-10-13/14)
Submitted `PARTITION=gb SHARDS=2` (jobs 1070519 = shard 0 on c672-007, 1070520 = shard 1 on c672-006; the gg job 1070303 was cancelled).
- **Shard 1 finished** (87/87 sources normalized, 87 stats files, exit 0): 10 h 29 min = **9 h 39 min fetch + ~50 min stats**; Slurm accounting: 4 d 18 h CPU = ~114 core-hours over 10.5 h wall = **~11 of 144 cores busy on average (7.5%)**,
  peak RSS 570 GB (at `--max-concurrent 12`), 1.6 TB read and 1.3 TB written.
- **Shard 0 did not finish in 12 h**: at 11 h 42 min, 90 of 116 sources were normalized (~20B of its ~51B tokens), 26 remained (all `nemotron_sft_v3`: multilingual_v1 x13, opencode_v1 x5, science_v2 x4, swe_v2 x2, math_proofs lean, math_v4), and `stats` had not started.
  At the pace of the first 11.7 h the rest needs ~18 h of fetch.
- **My estimate was wrong.** I predicted 2-4 h per shard (CPU work ~115 core-hours, assumed >= 25-50% of the cores busy). The CPU work was right (~114 core-hours); the utilization was not (7.5%). The single-source calibration (9.3 cores busy) had
  already shown the node mostly idle, but I assumed 12 concurrent sources would fill it; they gave a ~3.5x speed-up (31 MB/s vs 8.9 MB/s), not 12x. The likely causes are the ones I listed as risks: serial tails on the large Nemotron v3 sources and the
  per-step startup / shuffle I/O. Not diagnosed further.
- **Resume mechanism** (added): Marin steps use lease locks (heartbeat 30 s, stale after 90 s), so a dead job's interrupted steps are taken over by the next run within ~90 s, and finished steps are skipped. `RESUME=1 TAG=resume1 PARTITION=gb SHARDS=2 CONCURRENT=24 bash data/submit_fetch.sh`
  re-splits only what is left (`shards --only-todo`: sources with no `SUCCESS` normalized step cost full fetch + stats, fetched sources with no stats file cost 0.25 of that) into `shard-<k>-of-<n>-resume1.txt`; the first split's lists are untouched.
  Dry run: 115 sources have work left (26 need fetch, 89 only stats) = shard 0: 16 fetch sources (142 GB, 15.4B tokens), shard 1: 10 fetch + 89 stats-only (105 GB, 35.8B tokens); no duplicates, all within shard 0's original list.
- **Concurrency**: 12 concurrent steps peaked at 570 GB RAM; a gb node has 1.69 TB, so 24 is about the safe doubling (~1.1 TB). `fetch.sbatch` now defaults to 16; pass `CONCURRENT=24` for the resume.

### Correction and first real statistics (2026-10-14, from the 88 stats files of shard 1)
- **Correction.** The "allocation is the same ~0.507 fraction of every store; stores total ~105.9B; `select` should show ratio ~0.507 for every uncapped source" claim in the fidelity audit below was an over-generalization from two small sources
  and is **wrong**. The allocation tracks the registry's *rough* token counts: median allocation / `rough_token_count_b` = 0.507 and median allocation / measured store tokens = 0.507 (so our token counts agree with the registry's rough counts at the median), but
  allocation / store ranges from 0.049 (`nemotron_sft_v3/math_proofs_v2/train`) through 0.136 (`massive_function_calling`) and 0.235 (`arc_agi_v1`) up to 1.000 (sources taken whole: misconceptions, wildchat, some instruction-following). The 87 sources of shard 1 hold
  67B store tokens (7.26M rows) against 27.7B allocated; there is no "~105.9B" total to check. **The check that stands**: no source has store tokens below its allocation (the original raised in that case): 0 of 88 violate it, and the sources taken whole land exactly on
  their allocation (misconceptions: 1,704,859).
- **Row statistics, weighted by the allocation** (what training would see): average 8,074 tokens per row; 3.43M rows in the allocation of these 88 sources; 53% of tokens supervised across the store (agentic families ~30-60%, finance 5%, massive 0%).
- **Clip loss at 32,768 tokens is large for the long-trace families** (this replaces the 2-source finding "32K keeps >= 99%"): store-wide 16.7% of tokens, **11.4% weighted by the allocation**. By family (share of tokens lost): open_swe_traces 51% (84% of rows exceed 32K, 64K tokens
  per row), openthoughts4-code 67% (91K tokens per row), math_proofs_v2 50%, swe_v1 31%, arc_agi_v1 26%, superior-reasoning 14.5%, agenttrove 7%; penfever traces 0.1%, instruction following, synthetic-1, massive 0%. The cutoff is a decision to revisit (64K, or windowing for these families).
- **A 100k-row sample in proportion to the allocation** is ~0.81B tokens: ~3.9 h of hidden-state server time at the measured 57K tok/s (2 GPUs), i.e. one gb job.

- **Resume on one node while the other trains (2026-10-14).** `RESUME=1 TAG=resume1 PARTITION=gb SHARDS=1 CONCURRENT=24 bash data/submit_fetch.sh` (a resume now goes through the shard logic even with one shard, so it skips finished work per source):
  dry run lists 115 sources with work left (25 need fetch, 90 only stats; 247 GB of repositories, partly downloaded already). Expected ~10-11 h on one node (fetch of the 25 heavy sources at ~2x the first run's pace, then stats of 115 sources ~1 h); if it hits the
  12 h limit, run the same command again. When every source has stats, the job's last `select --only-if-complete` writes `selection.json` for all 203 sources. The 100k-row experiment does not depend on it.

### Fidelity audit of the data scripts against the original construction (2026-10-13)

The original path is `experiments/datakit/sft_chat_tokenize.py` (per source: `all_sft_sources()[name].normalized`, then `tokenize_attributes_step`, then a Levanter store) and the mixture in
`experiments/grug_sft/special_token_lr.py` at the run commit `8a0ec6cf`. The first version of my script stopped at the chat-normalized rows and tokenized with a plain tokenizer call; the audit found it diverged in three ways, now fixed:
- **Text normalize.** The text that was tokenized is the output of `normalize_step` over the rendered text: empty text dropped, **whitespace runs longer than 128 characters truncated to 128**, `id` = xxh3(text), and **global exact-text
  dedup**. `fetch` now runs the full `source.normalized` chain; `stats` reads its `outputs/main` (what the original read as `main_output_dir`), not the chat-normalized rows or my own rendering.
- **Tokenization.** Marin's `tokenize_batches_with_id` sets `_long_string_workaround = True`: texts over 10,000 characters are split at whitespace and encoded in pieces, which changes token ids at the split points; each document is
  `[BOS] + ids(text + " " + eos)`. `stats` calls Levanter's own `load_tokenizer` / `BatchTokenizer` with the same switches, instead of `tokenizer.encode`.
- **Loss mask.** Because the text is compacted, spans from the template are no longer aligned; the mask is derived from the token stream (assistant header ... `<|eot_id|>`), and `--verify-masks` checks it against the template's own
  `{% generation %}` spans (3 of 3 test sources: 100/100, 39/39, 93/93 rows identical).
- Tokenizer checks: vocab size 128,256, the nine reserved token ids, BOS/EOS ids (the original `verify_tokenizer`), and the repo's `training_chat_template.jinja` equal to Marin's `MARIN_CHAT_TEMPLATE`. (The original's
  `apply_chat_template` probe cannot be used on the Hub copy: its default template is the inference variant.)
- Same code, same commit: between the registry commit `8c3be8f` and `82620ad` the only data-code change is the added `sft_chat_tokenize.py`.
- **Evidence the counts match the original.** (1) `synthetic-misconceptions-conversations`: our store tokens = **1,704,859**, exactly its allocation (a whole-store source). (2) For the two sources that were partially used
  (`glm-5.2-kernelgym-rollouts`, `exp_rpt_issue`), allocation / our store tokens = 0.5070686 and 0.5070683: a constant factor, i.e. the allocation is the same fraction of each store (SFT budget 53,687,091,200 /
  total store tokens, so the stores total about 105.9B tokens; sources below the one-sample-per-block floor are taken whole). **The full `stats` run must reproduce this: the sum over the 203 sources should be ~105.9B and
  allocation / store ~0.5070 for every uncapped source.** `select` reports exact matches and any source below its allocation (an error in the original).
- Not reproduced exactly: the original drew packed 262,144-token sequences with a seeded shuffle inside each source (levanter mixture: `shuffle=True`, block size 64,000, `RESTART_STRATEGY`, one-epoch limit). `select` draws a seeded
  random subset of conversations with the same per-source token total; per-source token shares match, the particular conversations do not (the original seeds and order are not available).
- Superseded: the char-span prototype in step 2 / `scripts/mask_proto.py` tokenized the *un*-compacted rendering with a plain `tokenizer.encode`; keep it only as the cross-check inside `--verify-masks`.

- No decontamination in this SFT path (`datakit/decon.py` is not called by the SFT launcher). Say so in the final write-up.

## Sequence length (decided: 32,768; check on long-context families)

- `speculators` clips each row to `--seq-length` by plain tail truncation (`_append_row`: `input_ids[:max]`, `loss_mask[:max]`). The decoder is causal, so the hidden states of the kept prefix are identical with or
  without the tail: clipping only loses the tail's supervision (a row can be cut mid-assistant-turn; the preprocessing warns). A row whose supervised tokens are all beyond the clip is dropped.
- The same number is `--total-seq-len`: the trainer packs several rows into one sequence of that length (multipack, with `document_ids` for per-row attention), and `--max-anchors` is per packed sequence.
  A longer sequence length therefore means longer packed sequences, not just fewer clipped rows.
- Consequences of 32,768: `MAX_ANCHORS` 12,288; hidden states per packed sequence are 32,768 x (4 taps x 2,560) bf16 = ~671 MB (vs ~335 MB at 16K), so the Mooncake global/local segments (4 / 2 GiB
  in the sibling) must be re-sized for several in-flight rows; the vLLM server needs `--max-model-len` >= 32,768 (Grug is configured for 262,144); extraction cost per token rises a little in the 7 full-attention
  layers (the other 19 use a 2,048 window). Drafter training cost per token is unchanged.
- Rows longer than 32K (about 1% on kernelgym): clip, or split into windows. Decide after measuring the long-context families.

## Steps

### 1. Datakit smoke test on one tiny source: DONE 2026-10-13
- Ran the chat chain for `synthetic-misconceptions-conversations` locally with `scripts/smoke_chat.py` (in `/scratch/09749/atutej/marin_datakit/scripts/`): `StepRunner` with
  `MARIN_PREFIX=/scratch/09749/atutej/marin_datakit/out`; Zephyr falls back to `LocalClient`, so no cluster is needed. Raw download (one pinned `train.parquet`) -> chat transform ->
  chat normalize took ~5 s. 3,450 rows in, 3,450 out, 0 duplicates. Output: `out/normalized-chat/<name>_<hash>/outputs/main/*.parquet` with columns `id, messages (Harmony), source,
  source_id, chat_template_kwargs`. This source has only user/assistant-final messages (2-12 per row), `chat_template_kwargs` = `{"enable_thinking":false}`.
- Rendering + tokenizing + masking prototype (`scripts/mask_proto.py`): `render_jinja_template(..., return_assistant_tokens_mask=True)` on Datakit's grouped messages gives
  the `{% generation %}` char spans; tokenizing with offsets and marking tokens inside the spans gives the assistant-only mask. On all 3,450 rows: span count == assistant-turn count in
  every row, supervised tokens decode to exactly the span text (checked on 300 rows), 68% of tokens supervised, 1.70M tokens (matches the registry's 0.0017B estimate), max row 2,064 tokens,
  ~407k tokens/s on one core (so 53.7B tokens is ~37 core-hours of tokenizing, not the bottleneck).
- BOS/EOS confirmed with the real Grug tokenizer (200 rows): `[BOS]+ids(text)` == `ids("<|begin_of_text|>"+text)` in 200/200 rows; a stored Grug document ends `<|eot_id|>`, `" "`, `<|end_of_text|>`.
- Two agentic sources, same chain + same prototype (2026-10-13), no failures and no decode mismatches:
  - `glm-5.2-kernelgym-rollouts` (16 shards, ~100 MB, chain 14 s): 1,251 rows, 1,251 built; user + assistant analysis (thinking) + final; `enable_thinking` true. 16.1M tokens, 88% supervised,
    median row 12,323 tokens, p95 25,316, max 69,477; **21% of rows exceed 16,384**.
  - `penfever-traces/qwen35-122b-131k-opencode/exp_rpt_issue`: 93 rows, 93 built; system + user + assistant analysis/commentary/tool calls + tool responses; `tools` (10) and `enable_thinking` in
    `chat_template_kwargs`. 1.19M tokens, only 19% supervised (tool outputs and the long system prompt are unsupervised), median 11,545, max 29,252; 19% of rows exceed 16,384.
  - Clipping at 16,384 drops 5.5-8.7% of tokens (about 9% of supervised tokens) on these sources. Measured keep rate (tokens / supervised tokens) at each clip length:
    kernelgym 91.3% / 90.7% at 16K, 99.3% / 99.2% at 32K, 100% at 64K; penfever opencode 94.5% / 91.1% at 16K, 100% at 32K. Only two small sources: re-measure on the long-context families
    (`open_swe_traces`, `nemotron-terminal`, coderforge, Nemotron agentic) before finalizing 32K vs 64K.
  - Extraction cost is paid on all tokens but only ~20% are supervised for tool-heavy agentic traces.
- Still untested: the chain on a source with a big download (Nemotron v3 partitions, coderforge), and parallel runs of many sources at once.

### 2. Rendering and loss mask (largely settled by reading the templates)
- Checked 2026-10-13: Grug's training template keeps reasoning from **every** assistant turn (it emits `reasoning_content` for each assistant message; no stripping like
  Qwen's), and Datakit's `render_marin_chat` says "retaining reasoning from every turn". Marin's `MARIN_CHAT_TEMPLATE` is byte-identical to the model repo's
  `training_chat_template.jinja`. So: **one row per conversation**, no per-turn prefix duplication.
- Assistant-only loss mask: the template wraps each assistant message in `{% generation %}...{% endgeneration %}` (reasoning, text, tool calls, `<|eot_id|>`; not the
  assistant header), so `return_assistant_tokens_mask` gives it directly.
- Templates: the model repo ships `training_chat_template.jinja` and `chat_template.jinja`, identical except one line: when `enable_thinking` is undefined, training renders no
  `Reasoning:` header and inference renders `Reasoning: /think`. This never occurs in the training data: Datakit's `normalize_chat` always sets `enable_thinking` per row (true if any
  assistant message has an analysis channel, else false), so every training row has a `Reasoning: /think` or `/nothink` header, and both templates render every row identically.
  So render with the row's `chat_template_kwargs` (it carries `enable_thinking` and the row's `tools`) and either template gives the same text. Marin's `MARIN_CHAT_TEMPLATE` is the training one.
- For serving and for the acceptance eval (step 10) use the **inference** template: an omitted `enable_thinking` becomes `/think`, which matches the thinking training rows. A client that omits it
  with the training template would produce a header-less prompt the model never saw in SFT.
- Reuse Datakit's `_inference_messages` grouping (consecutive assistant messages, i.e. analysis + commentary + tool calls, become one turn with a single `<|eot_id|>`); do not
  convert Harmony messages with a generic OpenAI-style converter.
- BOS/EOS (read from the code, then confirmed with the real tokenizer on 2026-10-13, see step 1): Datakit renders text with `bos_token=""`. Levanter's `BatchTokenizer`
  (`enforce_bos`/`enforce_eos` default True) encodes the text with `add_special_tokens=False`, **inserts the BOS id at position 0**, and **appends `" " + eos_token`** (a space, then
  `<|end_of_text|>`) to each document before encoding. So a stored Grug document is `[BOS] + ids(text) + ids(" <|end_of_text|>")`. Our rows: `[BOS] + ids(rendered text)`; the trailer is outside every
  assistant span, so it needs no mask and can be left off. Still to verify with the real tokenizer: `[BOS] + ids(text)` == `ids("<|begin_of_text|>" + text)` (what vLLM serving produces from the
  template's `{{ bos_token }}`), and what the space+EOS trailer tokenizes to. (The Hub returned 500 for `tokenizer.json` on 2026-10-13; retry.) Also compare our token ids with Datakit's own render +
  tokenize on a few hundred rows per source family.
- Check that Grug was trained on all tokens (the launcher uses `TextLmDatasetFormat` on rendered text) or only assistant tokens; we mask to assistant tokens either way (that is what the
  drafter drafts at serving time).

### 3. Serve Grug MoE in the vLLM fork and get hidden states: DONE 2026-10-13 (smoke test passed)
Run with `vista/serve_smoke.sh` (+ `serve_smoke_client.py`, `serve_logit_check.py`, `serve_smoke_client_util.py`) on a gb idev node, GPUs 0,1, from the pinned fork build (`envs/vllm`, 39e62869).
- **Serving layout.** The fork's Grug GPU path requires `--tensor-parallel-size 1` ("GrugMoE expert-parallel GPU serving"); use `--data-parallel-size N --enable-expert-parallel` (N=2 here: each GPU held ~69 GB of the 134 GB). The server was up
  150-210 s after launch with the checkpoint on local disk.
- **Environment pitfalls hit and fixed.** `launch_vllm.py train` called `AutoConfig` only for the layer count (fails on `grug_moe`): fixed on the speculators branch (96699f9, falls back to config.json). The script must source the
  workspace `env.sh` (Vista's default `nvidia` module sets `CC=nvc`, which breaks inductor's warm-up: "nvc-Error-Unknown switch: -Wno-psabi") and `conda activate vllm` (puts `ninja`/`CUDA_HOME` on PATH). The hidden-state file is written
  asynchronously; wait for the file and its `.lock` (as speculators' client does).
- **Generation.** Thinking on: correct reasoning ("17 x 23 = 391") returned in `reasoning`, answer "391" in `content` (Grug reasoning parser); thinking off: correct, short answers.
- **Hidden states** (`file` backend, `--target-layer-ids 2 12 22`, final layer appended): `hidden_states` shape `(n_tokens, 4, 2560)` bf16, all finite, plus `token_ids`.
- **Prefill throughput with hidden states on, 2 GPUs (DP2 x EP2):** ~56-58K tokens/s for a single 16K-32K request (4K: 11.7K, partly warm-up); 4 concurrent 16K requests: 33K tokens/s aggregate (lower than one stream: DP ranks run in
  lock-step under expert parallelism, so concurrency across ranks is not free). Not yet measured on 6 GPUs, with Mooncake, or as sustained load. First estimate for the full mix at ~3x this rate (6 GPUs, ~170K tok/s):
  53.7B tokens ~ 3.7 days; 106B ~ 7 days. Re-measure before planning.
- **The tapped final layer and the gated head, checked against the live model** (`serve_logit_check.py`, 700 prompt tokens): recomputing the server's own prompt log-probs from the layer-26 tap with the checkpoint's weights:
  `lm_head(norm(h))` mean |diff| 0.0148 nats (max 4.59); **`lm_head(gate(norm(h)))` mean 0.0024, max 0.046, correlation 1.0000**; no norm 7.26; `norm(1+w)` 0.067; gate before norm 0.097. So (1) the layer-26 tap is the
  **pre-norm** residual stream (RMS ~1.9 per token vs norm weight mean 13.8), (2) the verifier's logits are exactly `lm_head(gate(norm(h)))`, which is what the DSpark patch (f93d1a5) computes for the TV-loss targets; skipping the gate is
  right on average but wrong by up to 4.6 nats on some tokens.

### 4. Choose tap layers and check the speculators fork supports this verifier
- Confirm `speculators` handles `grug_moe` verifier configs (hidden size, head counts, `get_verifier_config`) and the sliding window.
- Pick taps (3 + final) over the 26 layers. If time allows, compare two choices on a small run.
- Write the `--draft-config` JSON from the real verifier config; verify the parameter count (~250M) by instantiating the drafter.

### 5. Build the data at scale: per-source subsets, proportional by allocation
- Script (in this folder, `tools/` or `data/`) that for each selected source runs download -> transform -> chat-normalize on the needed shards, then converts to
  `conversations` and writes JSONL/Parquet shards with the source name in each row.
- The Datakit steps process whole sources (Nemotron v3 alone is 36.8B tokens). Plan disk and download size first; use `hf_urls_glob` or process a shard subset when a full source is
  more than we need. Pin the same dataset revisions as the registry.
- Interleave sources by their allocations when writing/reading so that any prefix of the stream has the right mix (the run must be stoppable at any point).
- Keep per-source filters from the transforms; note which sources were dropped (`dropped_sources`) and any that need auth.

### 6. Writing the training dataset: `write` + `finalize` (the "prepare" of this experiment): BUILT and tested 2026-10-13
Decided: no `prepare` stage in the training chain. The Datakit command ends in a `write` stage that produces the trainer's dataset directory; the chain starts at `train`.
- **Pipeline (all in `data/`)**: `fetch -> stats -> select` (`submit_fetch.sh`, as before) -> *look at the `stats` output, choose the cutoff* -> `write` + `finalize` (`SEQ_LENGTH=N bash submit_write.sh`, one gg node,
  `write.sbatch`). The dataset goes to `dataset_seq<N>/`, so several cutoffs can coexist; re-running `write` for another cutoff only re-tokenizes the selected rows.
- **What `speculators prepare-data` does for pre-tokenized rows, measured:** clip to `--seq-length`, drop rows with no supervised token, int64 `input_ids` / `loss_mask` + `seq_len`, shuffle, count supervised-token frequencies (`token_freq.pt`),
  write Arrow. It changes no id or mask besides the clip. Its cost is memory (~35 bytes per token held at once, so ~1.9 TB for the full mix) and 16 bytes per token on disk (int64): usable only chunk by chunk. So `write` reproduces its
  output directly instead of calling it.
- **`build_datakit_sft_mix.py write`** (datakit env; plain pyarrow, no torch): per normalized parquet file (parallel): read only the selected ids, tokenize with the same Levanter path as `stats`, build the mask, clip, drop unsupervised rows,
  count supervised-token frequencies, and bucket the rows by hash of the id into temp Arrow files (int32 / int8, ~5 bytes per token). Then per bucket (own, smaller pool): concatenate, shuffle (seeded), cast to int64 and write one shard
  `data-<k>-of-<N>.arrow` in `datasets`' schema with its Hugging Face metadata. Every shard holds every source, in proportion, so a run stopped early still trains on the right mix. Also writes `state.json`, `dataset_info.json`,
  `manifest.json` (per-source rows kept / clipped / dropped and token totals) and `token_freq.json`. Resumable (finished files and shards are skipped). **Refuses to run on a partial selection** (all 203 sources must be present and none below
  its allocation) unless `--allow-partial`.
- **`finalize_dataset.py`** (speculators env; needs torch): writes `token_freq.pt`, loads the directory with `datasets.load_from_disk` as the trainer does, and checks it against the manifest (row count, columns and dtypes,
  `seq_len == len(input_ids) <= seq_length`, mask in {0,1}, every row supervised; `--full-check` also recomputes the frequencies from every row).
- **Tested on the 3 small sources (4,146 rows, 4 shards, `--seq-length 32768`)**: the guard refuses the partial selection; the output loads under the training env's `datasets` 5.0.1; `--full-check` passes; the draft vocabulary mapping
  builds (`d2t` 32,000, `t2d` 128,256) and `build_client_item` works; **against `speculators prepare-data` on the same rows the output is identical**: rows (ids, mask, seq_len), `token_freq.pt`, Arrow schema + HF metadata, `state.json`
  format flag. Not yet tested: scale (many files / shards, memory under load), and a resumed run.
- **Scale estimates (not measured at scale)**: tokenizing ~33 core-hours for 53.7B tokens; output ~860 GB (int64) + ~270 GB temp (deleted after); memory ~3 GB per tokenizing worker (default 32) and ~7 GB per shard worker
  (default 8, `--shard-tokens 250M`). The gg node RAM is unknown (Slurm reports 1 MB); this gb node has 1.69 TB.
- **Superseded design** (shard writer that calls `prepare-data` per chunk; `prepare` stage in the chain): dropped.

### 7. End-to-end smoke training on the real target
- 2 nodes, a small sample (e.g. 300 conversations across all source families), `MAX_STEPS=10`: server up, Mooncake across nodes, trainer consumes, checkpoint written,
  validation runs.
- Measure: extraction tokens/s, trainer tokens/s, Mooncake occupancy and transfer rate (~20 KB/token at 4 taps), GPU split (6+2 vs 4+4).
- Fix the Mooncake sizing (the sibling's 4 GiB global / 2 GiB local was for a tiny target).

### 8. Pick the token budget and the schedule
- From step 7's rates: tokens per day -> days for the full mix. Estimate before measuring: ~2-4 days for 53.7B tokens if rows are not inflated; weeks if they are.
- Run to the full mix if it fits; otherwise a token budget with held-out accept length as the stopping signal. The proportional stream (step 5) makes a partial run valid.
- Slurm: chained resumable jobs at the QOS time limit (the sibling's `submit_chain.sh` pattern), W&B project `marin_speculator`, run name = this folder's name.

### 9. Full run
- `vista/run.sh` + `submit_chain.sh` for 2 GB200 nodes (new scripts; the sibling's are single-node).
- Monitor: loss, accept length on a held-out slice that covers every source family (not only the first sources in the stream), throughput, preemption/resume.

### 10. Evaluate and publish
- Rollout acceptance with Grug generating (`tools/eval_acceptance`), on prompts from the agentic and reasoning families; compare 1 vs 3 layers only if time allows.
- End-to-end tokens/s in vLLM with and without the drafter at the batch sizes we will serve at: this is the number that decides whether the drafter is worth its cost.
- Upload the best and latest checkpoints (public repo named `<experiment folder>-<YYYYMMDD>`; scan the diff before pushing; the token is `HUG_TOKEN` in `~/.bashrc`).

## Open questions / risks

- Does the vLLM fork serve Grug MoE correctly, and with hidden states from arbitrary layers? (Step 3: blocking.)
- Does `speculators` accept a `grug_moe` verifier config? (Step 4.)
- Row inflation from per-turn rows: resolved, one row per conversation (step 2). Still measure total tokens after clipping.
- Is the Datakit mix "completions" close enough to Grug's own text for offline training? Grug was SFT'd on exactly these, so probably, but acceptance on Grug rollouts (step 10) is the real test.
- Disk and download volume for the full mix (tens of TB?). Estimate in step 5 before downloading.
- GB200 queue wait and the QOS time limit.
- No decontamination in the source data: the drafter may see eval items. Harmless for speed, but note it if drafter evals reuse benchmark prompts.

## Full run: setup (2026-10-14 night). Scripts ready and dry-run checked; the data write is queued; training NOT submitted
- **Datakit selection done:** `selection.json` = 6,480,049 rows from all 203 sources, 53.689B tokens, none below allocation (the resume job 1071607 had fetched everything; its last `select` step had a bug,
  `sources_arg` read an `args.sources` the `select` command does not define, now fixed).
- **Write job:** `data/submit_write.sh` (now `-A ASC26088`, partition gg, 2 days) -> job 1072916, queued 2026-10-14 ~22:55. Never run at this scale (tested on 3 small sources and, with `--allow-partial`, on the 100k sample).
  Estimates in step 6 above: about 860 GB output, 33 core-hours of tokenizing. Resubmitting continues where it stopped.
- **Training scripts** (copied from the tested 100k experiment): `settings.sh`, `draft_config.json`, `vista/{run.sh,run.sbatch,submit_chain.sh}`. Changes: data and work dirs for the full mix; the run is a TOKEN BUDGET
  (`TOKEN_BUDGET`, default 12.5B tokens = 231,481 steps at 54k tokens/step, about 32.8 h at the 100k run's 0.51 s/step, i.e. 3 jobs) with `--max-steps` so the LR decays linearly over exactly that many steps and an early stop is annealed;
  `--train-data-ratio 0.999` (the default 0.9 would hold out 5.4B tokens: hours of validation at the end; this holds out about 6.5k rows, about 54M tokens); `--checkpoint-freq 0.01` (every 1% of the epoch, about 85 min,
  the resume granularity); no automatic Hugging Face upload in chain jobs (upload the finished run by hand: `HF_REPO=auto bash vista/run.sh upload`); `SMOKE=1` replaces the old "MAX_STEPS set means smoke" convention.
- **Chain:** `JOBS=3 AFTER=<write job id> bash vista/submit_chain.sh` (first job `afterok` the write, later jobs `afterany`; account ASC26088, partition gb, reservation CGAI-gb). Dry run checked 2026-10-14.
- **Not tested / caveats:** the write at scale; the 0.51 s/step and 54k tokens/step come from the 100k run (one node, 2 GPUs serving hidden states + 2 training) and may differ on the full mix (longer rows, more varied families);
  validation happens only at the end of the run (the trainer validates per epoch), so mid-run monitoring is the per-step training accept length; a finished budget still starts later chain jobs (server start, about 5 min) before they exit.
- **Scheduling conflict:** a 3-job chain occupies one reserved gb node for about 33 h and uses all 3 of the user's QOS gb submission slots (an RL job or an idev needs one), so it blocks other gb submissions, 2-node RL runs included,
  while it drains. Submit it deliberately.

### Decision (2026-10-14 night): the budget is the full epoch; training starts only after the RL pipeline is confirmed working with the drafter
- Budget: one epoch over the whole mixture (`TOKEN_BUDGET=full`, the default in `settings.sh`), as close to Grug's own SFT as we can get. Assumption to check: that Grug's SFT saw the full allocation once (53.7B tokens before
  the 32,768 clip; about 47.6B after it, since 11.4% of the allocation is clipped away). Estimate: about 880k steps, about 125 h, 11 jobs of 12 h at the 100k run's 0.51 s/step (not measured at this scale). The 12.5B-token
  budget in the section above is still available as `TOKEN_BUDGET=12500000000`.
- The epoch is too long to queue at once (3 gb submissions per user at a time), so `vista/drive_chain.sh` submits one 12 h job at a time from a login node (tmux/nohup) until the trainer's `epoch0_end` checkpoint
  link appears; it resumes from checkpoints, stops after two jobs that die within 10 minutes, and never uploads. Only its `--dry-run` path has been exercised.
- Launch order (decided by the user): write job 1072916 -> wait for the RL pipeline to be confirmed with the drafter -> then `AFTER=<write job> bash vista/drive_chain.sh` in tmux. Nothing is submitted for training yet.
