# agentic_eval: live agentic evals with a speculative-decoding drafter

Runs a Marin eval-policy Harbor benchmark (OpenThoughts-TBLite 2.0 or Terminal-Bench 2.1, from [Marin issue #10197](https://github.com/marin-community/marin/issues/10197)) against the Dr. Doom target served on the job's own node, with or without a DSpark drafter, and records the server's speculative-decoding counters over the run.

## How it works

- The agent is Terminus-2. It runs in the Harbor process on the GPU node and calls the model at `http://127.0.0.1:8000/v1`; only shell commands run in the Daytona sandbox. Nothing has to reach the node from outside.
- Serving follows the policy's Dr. Doom model config (73,728-token window, 7,168 batched tokens, 32 sequences, `hermes` tool parser), data parallel over the node's GPUs with expert parallelism.
- `spec_metrics.py` takes two scrapes of the server's `/metrics` (before and after) and writes `spec_metrics.json`: acceptance length (1 + accepted draft tokens per draft), acceptance per draft position, and the token counts.
- Harbor's own results (rewards, trajectories) are under `$OUT/harbor_jobs/<arm>`.

SWE-bench Verified is not covered: its agent (mini-swe-agent) runs inside the sandbox and would need an endpoint the sandbox can reach.

## Needs

- A GPU node with outbound internet (the job checks Daytona, GitHub and the Hugging Face Hub first and stops with a message if one is unreachable).
- A conda env with a vLLM that supports the Grug-layer DSpark drafter ([marin-community/vllm#128](https://github.com/marin-community/vllm/pull/128)); set `VLLM_ENV`. A vLLM without it needs `VLLM_OVERLAY`.
- A Daytona key: a file with `export DAYTONA_API_KEY=...` (`KEYS_ENV`, default `~/keys.env`). The job reads only that variable and never prints it.
- `uv` (installed with pip if missing). The pinned Harbor environment, `envs/harbor_policy`, is built on the first run from the policy's `marin-community/harbor` commit; the policy's Harbor configs are fetched from the pinned Marin commit.

## Run

From a login node:

```
# smoke: 3 tasks, one attempt
ARM=smoke BENCH=ot-tblite HARBOR_EXTRA="--n-tasks 3 --n-attempts 1" TIME=02:00:00 \
  DRAFTER=atutej/dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-filtered_itw-20261112 \
  VLLM_ENV=vllm_fork KEYS_ENV=/path/to/keys.env bash submit_agentic_eval.sh

# plain decoding on the same benchmark (omit DRAFTER); DRY=1 prints the sbatch command only
```

`DRAFTER` is a checkpoint directory or a Hugging Face repo id. Results go to `$RESULTS/<bench>/<arm>` (default `runs/agentic_eval_20261113/results`) and are never overwritten.

Machine settings, all environment variables of the submit command: `PARTITION`, `ACCOUNT`, `RESERVATION`, `GPUS`, `TIME`, `ROOT`. Defaults are Vista's `gb` partition with 4 GPUs.

- **Vista:** `KEYS_ENV` is `/scratch/<user>/keys.env` where the OT repo's notes put it; `VLLM_ENV=vllm_fork` is the env built from the PR branch. The `CGAI-gb` reservation expired on 2026-11-12; `RESERVATION=` submits without one.
- **Horizon:** `PARTITION=gpu` (or `gpu-dev`), `ACCOUNT=<project>` if you have several, `RESERVATION=`. See `setup/machines/horizon/NOTES.md`, a draft written without access to the machine. Build the vLLM env first: clone `https://github.com/atutej/vllm` (branch `grug-dspark-layers-main`) and run `setup/setup.sh` with `VLLM_ENV_NAME=vllm_fork VLLM_SRC=<that clean checkout> VLLM_SHA=<its HEAD>`; `setup/machines/vista/slurm/build_vllm_fork_env.sbatch` is the Vista version of this (a full node, hours), and Horizon reuses its recipe untested.

## Comparing drafters

At temperature 1 the trajectories differ between arms, so compare per-token numbers (acceptance length, tokens per second) and report task rewards separately, with their sampling noise.
