# SPEED-Bench (qualitative split) with AIPerf

NVIDIA's SPEED-Bench, run with NVIDIA's AIPerf against a vLLM server. The serving and reporting scripts are in `../eval_acceptance/` (`serve_speedbench.sh`, `speedbench_arms.sbatch`, `eval_matrix.sh`). This folder holds the helpers around them.

- `run_prepare_qualitative.sh` rebuilds the 494 masked prompts of the qualitative split with NVIDIA's `prepare_data.py` (Model-Optimizer, `examples/specdec_bench`). It needs a Hugging Face token that has accepted the terms of the gated `cais/hle` dataset.
- `fake_server.py` is a fake OpenAI-compatible server with vLLM-style speculative-decoding counters. It lets the AIPerf flags, the multi-turn dispatch and the acceptance report be tested without a GPU.
- `fixtest.sh` runs AIPerf against the fake server with each dispatch variant and counts which prompts and turns it sends.

What the dispatch test showed: `--request-count` (set to the number of turns) repeated about 4% of the requests and dropped the same number of follow-up turns, and `--no-allow-dataset-wrap` did not change that. `--num-conversations <samples in the category>`, with no `--request-count`, sends every turn of every sample exactly once. `serve_speedbench.sh` uses it by default and writes a per-category `coverage.json` that checks this against AIPerf's per-request export.

Not saved here: the step that converted the prepared parquet into AIPerf's JSONL layout (one row per sample with `question_id`, `category` and `messages`) was run inline. The paths in these scripts are the ones on our cluster.
