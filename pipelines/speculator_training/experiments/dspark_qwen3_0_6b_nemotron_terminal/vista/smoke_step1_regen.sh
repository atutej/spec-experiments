#!/bin/bash
# Smoke test of step 1 (on-policy regeneration) on a GPU node, same flags as genai/run.sh. Default: ONE
# GPU, first 100 rows of the step 0 smoke sample; compare with the genai reference in ../NOTES.md (0 failed,
# 43 truncated, 563 training rows, ~1.9 requests/s on one shared H100). Run after smoke_step0_export.sh.
#   nohup bash <this file> > "$PROJECT_ROOT/logs/smoke/step1-regen.log" 2>&1 &
# The real script's layout, data-parallel over 4 GPUs, on the whole 300-row sample:
#   DP=4 LIMIT=300 CONCURRENCY=256 GPU_MEM_UTIL=0.9 nohup bash <this file> > "$PROJECT_ROOT/logs/smoke/step1-regen-dp4.log" 2>&1 &
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "$REPO_DIR/env.sh"
[[ "$NUM_GPUS" -gt 0 ]] || { echo "no GPU on this node (NODE_KIND=$NODE_KIND); run on gh or gb" >&2; exit 1; }

MODEL="Qwen/Qwen3-0.6B" PORT=8078 DP=${DP:-1} LIMIT=${LIMIT:-100} CONCURRENCY=${CONCURRENCY:-32} GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3}
[[ "$NUM_GPUS" -ge "$DP" ]] || { echo "DP=$DP needs $DP GPUs (have $NUM_GPUS)" >&2; exit 1; }
TAG=$([[ "$DP" -eq 1 ]] && echo "" || echo "_dp$DP")   # DP=1 keeps the paths step 2 reads
SRC=$PROJECT_ROOT/runs/smoke_step0/nemotron-terminal_300_seed0.jsonl
OUT=$PROJECT_ROOT/runs/smoke_step1$TAG/regen.jsonl
VLOG=$PROJECT_ROOT/logs/smoke/step1_vllm$TAG.log
GPUS=$(seq -s, 0 $((DP-1)))
DP_ARGS=(); [[ "$DP" -gt 1 ]] && DP_ARGS=(--data-parallel-size "$DP")
[[ -f "$SRC" ]] || { echo "missing $SRC (run smoke_step0_export.sh first)" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"; rm -f "$OUT" "${OUT%.jsonl}.errors.jsonl"
SAMPLING='{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": true}}'

set +u; conda activate vllm || exit 1; set -u
CUDA_VISIBLE_DEVICES=$GPUS setsid vllm serve "$MODEL" --port "$PORT" "${DP_ARGS[@]}" --max-model-len 32768 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" > "$VLOG" 2>&1 &
PGID=$!
trap 'kill -TERM -- -$PGID 2>/dev/null; sleep 5; kill -KILL -- -$PGID 2>/dev/null' EXIT
echo "node=$(hostname) gpus=$GPUS ($(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)) DP=$DP limit=$LIMIT concurrency=$CONCURRENCY mem=$GPU_MEM_UTIL vllm log: $VLOG"
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 "$PGID" 2>/dev/null || { echo "vLLM exited:"; tail -n 20 "$VLOG"; exit 1; }
    sleep 5
done
echo "vLLM ready after ${SECONDS}s"

set +u; conda activate speculators || exit 1; set -u
START=$SECONDS
speculators regenerate-responses --dataset "$SRC" --limit "$LIMIT" \
    --endpoint "http://127.0.0.1:$PORT/v1/chat/completions" --max-tokens 8192 \
    --concurrency "$CONCURRENCY" --sampling-params "$SAMPLING" --seed 0 --outfile "$OUT"
rc=$?
echo "regenerate-responses exit code $rc, $((SECONDS-START))s"
ERR=${OUT%.jsonl}.errors.jsonl
echo "rows: $( [[ -f "$OUT" ]] && wc -l < "$OUT" || echo 0 )   failed conversations: $( [[ -f "$ERR" ]] && wc -l < "$ERR" || echo 0 )"
