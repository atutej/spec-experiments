#!/bin/bash
# Smoke test of steps 3-4 on one gb node (4 GPUs), the genai layout: mooncake_master, the hidden-state
# vLLM server on GPUs 0,1 (data-parallel 2), and online DSpark training on GPUs 2,3 for a few steps
# (--max-steps). Uses the step 2 output. Settings otherwise as in genai/run.sh. Needs 4 GPUs.
#   nohup bash <this file> > "$PROJECT_ROOT/logs/smoke/step3-4-train.log" 2>&1 &
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "$REPO_DIR/env.sh"
[[ "$NUM_GPUS" -ge 4 ]] || { echo "needs 4 GPUs (NODE_KIND=$NODE_KIND, NUM_GPUS=$NUM_GPUS); use a gb node" >&2; exit 1; }

SPEC_DIR=$PROJECT_ROOT/speculators
MODEL="Qwen/Qwen3-0.6B" PORT=8079 MC_PORT=50061 MAX_STEPS=${MAX_STEPS:-10} GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.3}
DATA=$PROJECT_ROOT/runs/smoke_step2/data CKPT=$PROJECT_ROOT/runs/smoke_step4/checkpoints
LOGS=$PROJECT_ROOT/logs/smoke
[[ -d "$DATA" ]] || { echo "missing $DATA (run smoke_step2_prepare.sh first)" >&2; exit 1; }
rm -rf "$CKPT"; mkdir -p "$CKPT"
TARGET_LAYER_IDS="2 14 25"
MASTER="" SERVER=""
cleanup() {
    for g in $SERVER $MASTER; do kill -TERM -- "-$g" 2>/dev/null; done; sleep 5
    for g in $SERVER $MASTER; do kill -KILL -- "-$g" 2>/dev/null; done
}
trap cleanup EXIT

set +u; conda activate vllm || exit 1; set -u
setsid mooncake_master --rpc_port "$MC_PORT" > "$LOGS/step3_mooncake_master.log" 2>&1 &
MASTER=$!
until (exec 3<>"/dev/tcp/127.0.0.1/$MC_PORT") 2>/dev/null; do
    kill -0 "$MASTER" 2>/dev/null || { echo "mooncake_master exited:"; tail -n 20 "$LOGS/step3_mooncake_master.log"; exit 1; }
    sleep 1
done
echo "mooncake_master ready"

# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0,1 setsid python "$SPEC_DIR/scripts/launch_vllm.py" "$MODEL" \
    --hidden-states-backend mooncake --mooncake-master "127.0.0.1:$MC_PORT" --mooncake-protocol tcp \
    --mooncake-global-segment-gib 4 --mooncake-local-buffer-gib 2 \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size 2 --port "$PORT" --gpu-memory-utilization "$GPU_MEM_UTIL" \
    > "$LOGS/step3_hs_vllm.log" 2>&1 &
SERVER=$!
until curl -sf "http://localhost:$PORT/health" >/dev/null; do
    kill -0 "$SERVER" 2>/dev/null || { echo "hidden-state server exited:"; tail -n 25 "$LOGS/step3_hs_vllm.log"; exit 1; }
    sleep 5
done
echo "hidden-state vLLM ready after ${SECONDS}s"

set +u; conda activate speculators || exit 1; set -u
START=$SECONDS
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node 2 -m speculators.train \
    --verifier-name-or-path "$MODEL" --speculator-type dspark --data-path "$DATA" \
    --vllm-endpoint "http://localhost:$PORT/v1" \
    --hidden-states-backend mooncake --mooncake-master "127.0.0.1:$MC_PORT" --mooncake-protocol tcp \
    --mooncake-global-segment-gib 4 --mooncake-local-buffer-gib 2 \
    --save-path "$CKPT" --draft-vocab-size 32000 --epochs 1 --max-steps "$MAX_STEPS" --lr 3e-4 \
    --total-seq-len 8192 --block-size 8 --max-anchors 3072 --num-layers 3 \
    --target-layer-ids $TARGET_LAYER_IDS --markov-rank 256 --markov-head-type vanilla \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn '{"ce": 0.1, "tv": 0.9}' --confidence-head-alpha 1.0 \
    --save-best --checkpoint-freq 0.1 --on-missing generate
rc=$?
echo "training exit code $rc after $((SECONDS-START))s"; ls -R "$CKPT" | head -20
exit $rc
