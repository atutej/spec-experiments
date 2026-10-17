#!/bin/bash
# How much of each GPU does training use? Runs the real train layout (hidden-state server on GPUs 0,1, training on
# GPUs 2,3) for a few steps on already-prepared data, sampling nvidia-smi every 2 s, and prints peak memory and mean
# utilization per GPU. Needs a 4-GPU gb node and a prepared dataset (default: runs/vista_tune_b/data from tune.sh).
#   STEPS=150 DATA=<prepared dir> bash profile_gpu.sh
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"
use_env() { set +u; conda activate "$1"; set -u; }
source "$EXP_DIR/settings.sh"
[[ "$NUM_GPUS" -ge 4 ]] || { echo "needs a 4-GPU gb node" >&2; exit 1; }
DATA=${DATA:-$PROJECT_ROOT/runs/vista_tune_b/data}; STEPS=${STEPS:-150}; OUT=$PROJECT_ROOT/logs/smoke/profile-gpu
[[ -d "$DATA" ]] || { echo "missing $DATA" >&2; exit 1; }
mkdir -p "$OUT"; CSV=$OUT/gpu.csv; rm -f "$OUT"/*
SERVER="" MASTER="" SAMPLER=""
stop() { for g in $SERVER $MASTER; do kill -TERM -- "-$g" 2>/dev/null; done; [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null; sleep 5
         for g in $SERVER $MASTER; do kill -KILL -- "-$g" 2>/dev/null; done; }
trap stop EXIT
use_env vllm
setsid mooncake_master --rpc_port "$MOONCAKE_PORT" > "$OUT/master.log" 2>&1 & MASTER=$!
until (exec 3<>"/dev/tcp/127.0.0.1/$MOONCAKE_PORT") 2>/dev/null; do sleep 1; done
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=0,1 setsid python "$SPEC_DIR/scripts/launch_vllm.py" "$MODEL" --hidden-states-backend mooncake \
    --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
    --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_GIB" --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_GIB" \
    --target-layer-ids $TARGET_LAYER_IDS -- --data-parallel-size 2 --port "$VLLM_PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" > "$OUT/server.log" 2>&1 & SERVER=$!
until curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null; do kill -0 "$SERVER" 2>/dev/null || { tail "$OUT/server.log"; exit 1; }; sleep 5; done
echo "server ready; sampling GPUs while training $STEPS steps"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > "$CSV" & SAMPLER=$!
use_env speculators
# shellcheck disable=SC2086
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node 2 -m speculators.train \
    --verifier-name-or-path "$MODEL" --speculator-type "$SPECULATOR_TYPE" --data-path "$DATA" \
    --vllm-endpoint "http://localhost:$VLLM_PORT/v1" --hidden-states-backend mooncake \
    --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
    --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_GIB" --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_GIB" \
    --save-path "$OUT/ckpt" "${VOCAB_ARGS[@]}" --epochs 1 --max-steps "$STEPS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" --num-layers "$NUM_LAYERS" --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE" --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" --on-missing generate > "$OUT/train.log" 2>&1
echo "training rc=$?"
kill "$SAMPLER" 2>/dev/null; SAMPLER=""
python3 -I - "$CSV" <<'PY'
import csv, sys, collections
by = collections.defaultdict(list)
for r in csv.reader(open(sys.argv[1])):
    if len(r) == 4: by[int(r[1])].append((float(r[2]), float(r[3])))
role = {0: "hidden-state server", 1: "hidden-state server", 2: "trainer rank 0", 3: "trainer rank 1"}
print("GPU role                 samples  peak mem (GiB / 183)  mean util   util when busy (>5%)")
for g in sorted(by):
    m = [x[0] for x in by[g]]; u = [x[1] for x in by[g]]; busy = [x for x in u if x > 5]
    print(f"{g}   {role[g]:22s} {len(u):6d}  {max(m)/1024:8.1f}               {sum(u)/len(u):5.1f}%     {(sum(busy)/len(busy) if busy else 0):5.1f}%")
PY
