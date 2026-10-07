#!/bin/bash
# GPU smoke test of the serving stack: start `vllm serve` on one GPU, check /health and a
# chat completion with return_token_ids (regenerate-responses needs prompt_token_ids), stop.
# Run on a GPU node. Uses whatever GPU CUDA_VISIBLE_DEVICES selects (default: GPU 0).
#   MODEL=Qwen/Qwen3-0.6B PORT=8077 GPU_MEM_UTIL=0.3 setup/check_gpu.sh
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}" PORT="${PORT:-8077}" GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.3}"
LOG="$TMPDIR/check_gpu_vllm.log"

set +u; conda activate vllm || exit 1; set -u
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" setsid vllm serve "$MODEL" --port "$PORT" \
    --max-model-len 32768 --gpu-memory-utilization "$GPU_MEM_UTIL" > "$LOG" 2>&1 &
PGID=$!
trap 'kill -TERM -- -$PGID 2>/dev/null; sleep 5; kill -KILL -- -$PGID 2>/dev/null' EXIT

echo "starting vllm serve $MODEL on port $PORT (log: $LOG)"
for _ in $(seq 120); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 "$PGID" 2>/dev/null || { echo "vLLM exited:"; tail -n 20 "$LOG"; exit 1; }
    sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "vLLM not healthy after 10 min"; exit 1; }

curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL\", \"max_tokens\": 32, \"return_token_ids\": true,
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hi.\"}],
    \"chat_template_kwargs\": {\"enable_thinking\": false}}" |
python -c '
import json, sys
d = json.load(sys.stdin)
ok = bool(d.get("prompt_token_ids")) and bool(d["choices"][0].get("token_ids"))
print("reply:", d["choices"][0]["message"]["content"][:80].replace("\n", " "))
print("return_token_ids:", "ok" if ok else "MISSING")
sys.exit(0 if ok else 1)'
