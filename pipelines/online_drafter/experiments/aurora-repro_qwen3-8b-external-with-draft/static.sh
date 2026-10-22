#!/bin/bash
# The STATIC baseline of the same request stream: the same server and the same pretrained draft, never updated, nothing captured and no trainer.
#   bash static.sh <NEW run dir> <data dir>        (GPU 2 by default; SERVER_GPU=<n> to change)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TOOLS="$HERE/../../tools"; source "$TOOLS/lib.sh"
init_run "${1:?usage: static.sh <new run dir> <data dir>}"; DATA=${2:?data dir}; shift 2
port_free 30001 || { echo "port 30001 in use" >&2; exit 1; }
TARGET=Qwen/Qwen3-8B; BATCH=${BATCH:-48}; MAX_TOKENS=${MAX_TOKENS:-512}; N=$(wc -l < "$DATA/dataset.jsonl"); REQUESTS=${REQUESTS:-$N}
cleanup() { touch "$RUN/stop"; for n in client server; do stop_one $n; done; }
trap cleanup EXIT INT TERM
run_in vllm server CUDA_VISIBLE_DEVICES=${SERVER_GPU:-2} HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING PYTHONPATH="$TOOLS" TOKENIZERS_PARALLELISM=false -- \
    python "$TOOLS/vllm_capture_server.py" --target Qwen/Qwen3-8B --draft Tengyunw/qwen3_8b_eagle3 --num-spec-tokens 5 --max-model-len 8192 --gpu-memory-utilization 0.85 --port 30001 ${SERVER_ARGS:-}
wait_http "http://localhost:30001/health" 1500 server || exit 1
run_in aurora client HF_HUB_OFFLINE=1 -- python "$TOOLS/rollout_client.py" --server http://localhost:30001 --prompts "$DATA/dataset.jsonl" --tokenizer "$TARGET" --chat-template-kwargs '{"enable_thinking": true}' --max-prompt-tokens 6000 --batch "$BATCH" --max-tokens "$MAX_TOKENS" --temperature 0 \
    --stop-file "$RUN/stop" --log "$RUN/rollouts.jsonl" --max-batches $(( (REQUESTS + BATCH - 1) / BATCH ))
while alive client; do sleep 20; done; echo "static run finished: $RUN/rollouts.jsonl"
