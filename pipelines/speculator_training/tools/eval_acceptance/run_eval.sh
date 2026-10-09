#!/bin/bash
# Rollout acceptance length of one drafter checkpoint: serve the target WITH the draft on one GPU, let the target generate on the
# held-out prompts, read the speculative-decoding counters (run_eval.py). Runs as-is on a gb idev node.
#
#   bash run_eval.sh CHECKPOINT_DIR [OUT_DIR]
#
# env: PROMPTS_DIR (default $PROJECT_ROOT/runs/eval_prompts_500, made by build_prompts.py), TARGET (Qwen/Qwen3-0.6B),
#      NUM_SPEC_TOKENS (8 = the draft's block size), MAX_PROMPTS (testing: only the first N per subset), MAX_TOKENS (8192),
#      CONCURRENCY (32), GPU (0), PORT (8100), WANDB_NAME (log the results to W&B as an eval run)
set -euo pipefail
set -E; trap 'echo "run_eval.sh: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../../../.." && pwd)"   # this file: pipelines/<pipeline>/tools/eval_acceptance/
source "$REPO_DIR/env.sh"
use_env() { set +u; conda activate "$1"; set -u; }

CKPT=${1:?usage: run_eval.sh CHECKPOINT_DIR [OUT_DIR]}
CKPT=$(cd "$CKPT" && pwd)
OUT=${2:-$CKPT/../eval_rollout_acceptance}
PROMPTS_DIR=${PROMPTS_DIR:-$PROJECT_ROOT/runs/eval_prompts_500}
TARGET=${TARGET:-Qwen/Qwen3-0.6B}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-8}
MAX_TOKENS=${MAX_TOKENS:-8192}
CONCURRENCY=${CONCURRENCY:-32}
GPU=${GPU:-0}
PORT=${PORT:-8100}
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_speculator
mkdir -p "$OUT"
[[ -f "$CKPT/config.json" && -f "$PROMPTS_DIR/manifest.json" ]] || { echo "missing $CKPT/config.json or $PROMPTS_DIR/manifest.json" >&2; exit 1; }

use_env vllm
SPEC=$(printf '{"model": "%s", "num_speculative_tokens": %s, "method": "dspark"}' "$CKPT" "$NUM_SPEC_TOKENS")
CUDA_VISIBLE_DEVICES=$GPU vllm serve "$TARGET" --port "$PORT" --max-model-len 32768 --speculative-config "$SPEC" \
    > "$OUT/server.log" 2>&1 &
echo $! > "$OUT/server.pid"
trap 'kill "$(cat "$OUT/server.pid")" 2>/dev/null || true' EXIT
echo "serving $TARGET + $CKPT (log $OUT/server.log)"
for _ in $(seq 1 180); do   # up to 30 min (first start compiles kernels)
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 "$(cat "$OUT/server.pid")" 2>/dev/null || { echo "server died; tail of $OUT/server.log:" >&2; tail -30 "$OUT/server.log" >&2; exit 1; }
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server not healthy after 30 min" >&2; exit 1; }

use_env speculators
EXTRA=()
[[ -z "${MAX_PROMPTS:-}" ]] || EXTRA+=(--max-prompts "$MAX_PROMPTS")
[[ -z "${WANDB_NAME:-}" ]] || EXTRA+=(--wandb-name "$WANDB_NAME")
python "$HERE/run_eval.py" --target "http://127.0.0.1:$PORT" --model "$TARGET" --prompts-dir "$PROMPTS_DIR" --output-dir "$OUT" \
    --max-tokens "$MAX_TOKENS" --concurrency "$CONCURRENCY" --checkpoint "$CKPT" "${EXTRA[@]}"
cat "$OUT/acceptance.csv"
