#!/bin/bash
# DSpark drafter for Qwen3-0.6B trained on the Nemotron-Terminal corpus's OWN completions (DeepSeek-V3.2), with no
# regeneration, on Vista. Sibling of ../../dspark_qwen3_0_6b_nemotron_terminal/vista/run.sh (on-policy, Qwen3-0.6B's
# own regenerated responses); same data sample, same training settings, so the two runs differ only in the completions.
#
#   bash vista/run.sh [stage ...]       stages: export prepare train   (default: all, in order)
#
#   export   CPU        link the on-policy experiment's 100k sample if it exists, else export it  (skipped if present)
#   prepare  CPU        `speculators prepare-data --render-endpoint` on the raw conversations. Rendering is done by
#                       `vllm launch render`, vLLM's GPU-less render server (no GPU, no weights): it applies Qwen3's chat
#                       template, one row per assistant turn (reasoning of earlier turns is stripped from the history, as at
#                       inference). Runs on a gg node. Skipped if $DATA_DIR exists.
#   train    4-GPU gb   mooncake_master, hidden-state vLLM on GPUs 0,1, online training on GPUs 2,3 (resumes from
#                       $WORK_DIR/checkpoints if present). Needs $DATA_DIR (run prepare first).
#
# Why a separate render server: the hidden-state server of this vLLM build does not register the
# `/v1/chat/completions/render` route (a first version that rendered through it got a 404 for every conversation).
#
# Runs as-is on an idev node (prepare and train need a gb node). Small test of everything on idev:
#   SAMPLE_LIMIT=300 MAX_STEPS=10 WORK_DIR=<abs path> bash vista/run.sh
set -euo pipefail
# Never die silently under `set -e` (a batch job's log would stay empty): say which command failed.
set -E; trap 'echo "run.sh: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"   # repo root (this file: pipelines/<pipeline>/experiments/<name>/vista/)
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"
use_env() { set +u; conda activate "$1"; set -u; }   # conda's activate scripts are not nounset-safe
source "$EXP_DIR/settings.sh"

VLLM_GPUS="0,1" NUM_VLLM_GPUS=2           # hidden-state server (also renders during prepare) ...
TRAIN_GPUS="2,3" NUM_TRAIN_GPUS=2         # ... and training side by side
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(export prepare train)
for s in "${STAGES[@]}"; do case "$s" in
    export|prepare) ;;
    train) [[ "$NUM_GPUS" -ge 4 ]] || { echo "stage '$s' needs a 4-GPU gb node (NODE_KIND=$NODE_KIND, NUM_GPUS=$NUM_GPUS)" >&2; exit 1; } ;;
    *) echo "unknown stage '$s' (export prepare train)" >&2; exit 2 ;;
esac; done
mkdir -p "$LOG_DIR" "$(dirname "$SOURCE_FILE")"
echo "node=$(hostname) kind=$NODE_KIND gpus=$NUM_GPUS stages=${STAGES[*]} WORK_DIR=$WORK_DIR SAMPLE_LIMIT=$SAMPLE_LIMIT MAX_STEPS=${MAX_STEPS:-none}"

# ---- server helpers: start in its own process group, stop the whole group ----
SERVER_PGID=""
start_server() {  # start_server <gpus> <logfile> <command...>
    local gpus="$1" logfile="$2"; shift 2
    CUDA_VISIBLE_DEVICES="$gpus" setsid "$@" > "$logfile" 2>&1 &
    SERVER_PGID=$!
    echo "Waiting for vLLM (log: $logfile)..."
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        kill -0 "$SERVER_PGID" 2>/dev/null || { echo "vLLM exited; see $logfile" >&2; tail -n 30 "$logfile" >&2; exit 1; }
        sleep 5
    done
    echo "vLLM ready after ${SECONDS}s."
}
stop_server() {
    [[ -n "$SERVER_PGID" ]] || return 0
    echo "Stopping vLLM server..."
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 -- "-$SERVER_PGID" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    SERVER_PGID=""
    sleep 10   # let GPU memory drain before the next server claims the same GPUs
}
MASTER_PGID=""
start_master() {
    setsid mooncake_master --rpc_port "$MOONCAKE_PORT" > "$LOG_DIR/mooncake_master.log" 2>&1 &
    MASTER_PGID=$!
    until (exec 3<>"/dev/tcp/127.0.0.1/$MOONCAKE_PORT") 2>/dev/null; do
        kill -0 "$MASTER_PGID" 2>/dev/null || { echo "mooncake_master exited; see $LOG_DIR/mooncake_master.log" >&2; exit 1; }
        sleep 1
    done
    echo "mooncake_master ready."
}
stop_master() {
    [[ -n "$MASTER_PGID" ]] || return 0
    kill -TERM -- "-$MASTER_PGID" 2>/dev/null || true
    sleep 2
    kill -KILL -- "-$MASTER_PGID" 2>/dev/null || true
    MASTER_PGID=""
}
cleanup() { stop_server; stop_master; stop_render_server; }
trap cleanup EXIT

RENDER_PGID=""
start_render_server() {  # vLLM's GPU-less render server: preprocessing only, no GPU and no model weights
    use_env vllm
    CUDA_VISIBLE_DEVICES="" setsid vllm launch render "$MODEL" --port "$RENDER_PORT" \
        --api-server-count "$RENDER_API_SERVERS" --renderer-num-workers "$RENDER_WORKERS" \
        > "$LOG_DIR/render_vllm.log" 2>&1 &
    RENDER_PGID=$!
    echo "Waiting for the render server (log: $LOG_DIR/render_vllm.log)..."
    until curl -sf "http://localhost:${RENDER_PORT}/health" > /dev/null 2>&1; do
        kill -0 "$RENDER_PGID" 2>/dev/null || { echo "render server exited; see $LOG_DIR/render_vllm.log" >&2; tail -n 30 "$LOG_DIR/render_vllm.log" >&2; exit 1; }
        sleep 3
    done
    echo "render server ready after ${SECONDS}s."
}
stop_render_server() {
    [[ -n "$RENDER_PGID" ]] || return 0
    kill -TERM -- "-$RENDER_PGID" 2>/dev/null || true
    sleep 3
    kill -KILL -- "-$RENDER_PGID" 2>/dev/null || true
    RENDER_PGID=""
}

start_hidden_state_server() {  # mooncake_master + the hidden-state vLLM on VLLM_GPUS (train only; stopped at exit)
    use_env vllm
    start_master
    # shellcheck disable=SC2086  # TARGET_LAYER_IDS is intentionally word-split
    start_server "$VLLM_GPUS" "$LOG_DIR/hs_vllm.log" \
        python "$SPEC_DIR/scripts/launch_vllm.py" "$MODEL" \
        --hidden-states-backend mooncake \
        --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
        --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_GIB" --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_GIB" \
        --target-layer-ids $TARGET_LAYER_IDS \
        -- --data-parallel-size "$NUM_VLLM_GPUS" --port "$VLLM_PORT" \
        --gpu-memory-utilization "$GPU_MEM_UTIL"
}

stage_export() {
    if [[ -f "$SOURCE_FILE" ]]; then echo "=== Step 0: $SOURCE_FILE exists, skipping ==="; return 0; fi
    if [[ -f "$SHARED_SOURCE_FILE" ]]; then
        echo "=== Step 0: linking the on-policy experiment's sample $SHARED_SOURCE_FILE ==="
        ln -s "$SHARED_SOURCE_FILE" "$SOURCE_FILE"; return 0
    fi
    echo "=== Step 0: Exporting a random sample of $DATASET ==="
    use_env speculators
    python "$REPO_DIR/pipelines/speculator_training/tools/export_registry_dataset.py" \
        --dataset "$DATASET" "${EXPORT_ARGS[@]}" \
        --limit "$SAMPLE_LIMIT" --seed "$SAMPLE_SEED" --out "$SOURCE_FILE"
}

stage_prepare() {
    [[ -f "$SOURCE_FILE" ]] || { echo "missing $SOURCE_FILE (run the export stage first)" >&2; exit 1; }
    if [[ -d "$DATA_DIR" ]]; then echo "=== Step 1: $DATA_DIR exists, skipping (delete it to redo) ==="; return 0; fi
    echo "=== Step 1: Preparing data from the corpus's own completions (render via the GPU-less render server) ==="
    start_render_server
    use_env speculators
    # Written to $DATA_DIR.tmp and renamed, so an interrupted run never leaves a half-written $DATA_DIR behind.
    speculators prepare-data \
        --model "$MODEL" \
        --data "$SOURCE_FILE" \
        --render-endpoint "http://localhost:${RENDER_PORT}" \
        --output "$DATA_DIR.tmp" \
        --seq-length "$SEQ_LENGTH" \
        --overwrite
    rm -rf "$DATA_DIR"; mv "$DATA_DIR.tmp" "$DATA_DIR"
    stop_render_server
}

stage_train() {
    [[ -d "$DATA_DIR" ]] || { echo "missing $DATA_DIR (run the prepare stage first)" >&2; exit 1; }
    echo "=== Step 2: Launching hidden-state vLLM server ==="
    start_hidden_state_server
    echo "=== Step 3: Training ==="
    use_env speculators
    # No --save-best: with it the trainer writes NO mid-epoch checkpoints (only at the end of an epoch), so a time
    # limit in this single-epoch run would lose all progress. --checkpoint-freq 0.1 saves every 10% of the epoch and
    # a rerun of this stage resumes from the last one. Training itself is unchanged.
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
        --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
        -m speculators.train \
        --verifier-name-or-path "$MODEL" \
        --speculator-type "$SPECULATOR_TYPE" \
        --data-path "$DATA_DIR" \
        --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
        --hidden-states-backend mooncake \
        --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
        --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_GIB" --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_GIB" \
        --save-path "$CKPT_DIR" \
        "${VOCAB_ARGS[@]}" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --total-seq-len "$SEQ_LENGTH" \
        --block-size "$BLOCK_SIZE" \
        --max-anchors "$MAX_ANCHORS" \
        --num-layers "$NUM_LAYERS" \
        --target-layer-ids $TARGET_LAYER_IDS \
        --markov-rank "$MARKOV_RANK" \
        --markov-head-type "$MARKOV_HEAD_TYPE" \
        --enable-confidence-head \
        --confidence-head-with-markov \
        --loss-fn "$LOSS_FN" \
        --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
        --checkpoint-freq 0.1 \
        --on-missing generate \
        "${TRAIN_EXTRA[@]}"
    echo "Done. Checkpoints saved to $CKPT_DIR/"
}

for s in "${STAGES[@]}"; do "stage_$s"; done
echo "Stages finished: ${STAGES[*]}"
