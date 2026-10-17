#!/bin/bash
# Smoke test of the Grug-67B-A2B target in the Marin vLLM fork: does it load, generate sensibly, and emit hidden states from
# chosen layers, and how fast does it prefill? Run on a GPU node (idev or sbatch). The weights must already be in
# $HF_HOME (cache/hf): `snapshot_download` resumes; see logs/grug_serving/download.py.
#   bash vista/serve_smoke.sh                  # DP=2 x EP=2 on GPUs 0,1
#   DP=4 GPUS=0,1,2,3 TARGET_LAYER_IDS="2 12 22" bash vista/serve_smoke.sh
# The fork's Grug GPU path requires tensor_parallel_size=1 ("GrugMoE expert-parallel GPU serving"): the experts are sharded by
# expert parallelism across the data-parallel ranks (the layout Luke's eval used: TP1 x DP4 x EP4).
# Uses the `file` hidden-states backend (no Mooncake): the point is the model, not the transport.
source ~/.bashrc 2>/dev/null || true   # before `set -u`: bashrc uses unset variables
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator
# The workspace env: CC=gcc CXX=g++ (Vista's default `nvidia` module sets CC=nvc, which breaks torch inductor's compile
# warm-up: "nvc-Error-Unknown switch: -Wno-psabi"), the Triton/HF caches, TMPDIR.
set +u; source $ROOT/spec-experiments/env.sh
conda activate vllm   # as the Qwen runs' use_env: puts ninja and the env's nvcc (CUDA_HOME) on PATH; conda's scripts are not nounset-safe
set -u
SPEC_DIR=${SPEC_DIR:-$ROOT/speculators}      # the speculators checkout whose scripts/launch_vllm.py is used
MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21}
DP=${DP:-2}; GPUS=${GPUS:-0,1}; PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-33792}                       # 32,768 + room for the generation prompt
TARGET_LAYER_IDS=${TARGET_LAYER_IDS:-"2 12 22"}   # the final layer (26) is appended by launch_vllm.py
OUT=${OUT:-$ROOT/runs/grug_serving_smoke}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME=$ROOT/cache/hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_TOKEN=${HF_TOKEN:-${HUG_TOKEN:-}}
mkdir -p "$OUT/hs"; LOG="$OUT/server.log"
PY=$ROOT/envs/vllm/bin/python
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=$GPUS $PY $SPEC_DIR/scripts/launch_vllm.py train "$MODEL" \
    --hidden-states-backend file --hidden-states-path "$OUT/hs" --target-layer-ids $TARGET_LAYER_IDS \
    -- --tensor-parallel-size 1 --data-parallel-size "$DP" --enable-expert-parallel --port "$PORT" --max-model-len "$MAXLEN" --gpu-memory-utilization 0.9 \
       --reasoning-parser grug > "$LOG" 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
echo "server pid $SERVER (TP=1 DP=$DP EP=$DP GPUs=$GPUS), log $LOG; waiting for it to load the 134 GB checkpoint ..."
for i in $(seq 1 180); do                     # up to 30 minutes
    curl -sf "http://localhost:$PORT/health" >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "server died; tail of the log:"; tail -30 "$LOG"; exit 1; }
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "server not healthy after 30 min"; tail -30 "$LOG"; exit 1; }
echo "server up after $((i*10)) s"
$PY "$HERE/${CLIENT:-serve_smoke_client.py}" --port "$PORT" --model "$MODEL" --layers $TARGET_LAYER_IDS --hs-dir "$OUT/hs" | tee "$OUT/client.log"
