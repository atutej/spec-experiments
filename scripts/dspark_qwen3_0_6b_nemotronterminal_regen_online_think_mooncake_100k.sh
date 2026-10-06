#!/bin/bash
# Online DSpark training for Qwen3-0.6B on Nemotron, with on-policy regeneration.
# DSpark settings follow the official examples/train/dspark_qwen3_0_6b_sharegpt_online.sh;
# this script adds a generation step so the training data is Qwen3-0.6B's own output:
#
#   Step 1  plain `vllm serve` + `speculators regenerate-responses`
#           -> Qwen3-0.6B rewrites every assistant turn (turn by turn, on its own history)
#           -> pretokenized per-turn rows (input_ids + loss_mask)
#   Step 2  `speculators prepare-data` packages those rows (no render endpoint needed)
#   Step 3  hidden-state vLLM server (scripts/launch_vllm.py) exposing TARGET_LAYER_IDS
#   Step 4  online DSpark training against that server
#
# Run from the root of a speculators checkout, in an environment with both vllm and
# speculators (>= 0.8.0) installed, like the official examples.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
SPEC_DIR="$PROJECT_ROOT/speculators"
use_env() { set +u; conda activate "$1"; set -u; }   # conda's activate scripts are not nounset-safe

# ============ Configuration ============
MODEL="Qwen/Qwen3-0.6B"
DATASET="nemotron"      # regenerate-responses preset (Nemotron)
REGEN_LIMIT=100000                # conversations to regenerate; remove the flag below for all ~1.4M
WORK_DIR="$PROJECT_ROOT/runs/main"
REGEN_FILE="$WORK_DIR/regen/nemotron_qwen3_0_6b.jsonl"   # step 1 output
DATA_DIR="$WORK_DIR/data"        # step 2 output; kept separate because prepare-data
                                 # --overwrite refuses directories holding other files
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
VLLM_PORT=8000
MOONCAKE_PORT=50051              # hidden states move through Mooncake's in-RAM store, not files
MOONCAKE_GLOBAL_GIB=4 
MOONCAKE_LOCAL_GIB=2
GPU_MEM_UTIL=0.5                 # vLLM default 0.92 does not fit next to other users' jobs on this shared machine

# Generation settings -- match how you will SERVE the model.
ENABLE_THINKING="true"           # "true" if you serve Qwen3 with thinking on
MAX_GEN_TOKENS=6144              # per-request max_tokens; also stops a conversation once its
                                 # total length passes this (regenerate-responses behaviour)
REGEN_MAX_MODEL_LEN=12288        # >= prompt + MAX_GEN_TOKENS, or vLLM rejects the request
REGEN_CONCURRENCY=512            # scale with NUM_ALL_GPUS so every replica stays busy
MAX_ERROR_FRAC=0.02              # abort if more conversations than this fail

# Training settings (from the official DSpark Qwen3-0.6B example)
SEQ_LENGTH=8192                  # training rows longer than this are clipped by prepare-data
SPECULATOR_TYPE="dspark"
EPOCHS=1
LR=3e-4
BLOCK_SIZE=8                     # tokens drafted per step
MAX_ANCHORS=3072
NUM_LAYERS=3
TARGET_LAYER_IDS="2 14 25"       # Qwen3-0.6B has 28 layers; launch_vllm.py also appends layer 28
DRAFT_VOCAB_SIZE=32000           # reduced vocab: a full 152K head would dwarf a 3-layer draft
                                 # for this small model. Set to "" to use the full vocabulary.
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"       # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

ALL_GPUS="0,1,2,3"               # step 1 runs alone, so it uses every GPU
NUM_ALL_GPUS=4
VLLM_GPUS="0,1"                  # steps 3-4: hidden-state server ...
NUM_VLLM_GPUS=2
TRAIN_GPUS="2,3"                 # ... and training side by side
NUM_TRAIN_GPUS=2
# =======================================

if [[ "$ENABLE_THINKING" == "true" ]]; then   # Qwen3's recommended sampling per mode
    SAMPLING_PARAMS='{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": true}}'
else
    SAMPLING_PARAMS='{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}'
fi
VOCAB_ARGS=()
[[ -n "$DRAFT_VOCAB_SIZE" ]] && VOCAB_ARGS=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
mkdir -p "$(dirname "$REGEN_FILE")" "$LOG_DIR"

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
    echo "vLLM ready."
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
cleanup() { stop_server; stop_master; }
trap cleanup EXIT

# Step 1: On-policy regeneration with a plain vLLM server (not the hidden-state server)
echo "=== Step 1: Regenerating $DATASET responses with $MODEL ==="
use_env vllm
start_server "$ALL_GPUS" "$LOG_DIR/regen_vllm.log" \
    vllm serve "$MODEL" --port "$VLLM_PORT" \
    --data-parallel-size "$NUM_ALL_GPUS" --max-model-len "$REGEN_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL"

use_env speculators
speculators regenerate-responses \
    --dataset "$DATASET" \
    --limit "$REGEN_LIMIT" \
    --endpoint "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    --max-tokens "$MAX_GEN_TOKENS" \
    --concurrency "$REGEN_CONCURRENCY" \
    --sampling-params "$SAMPLING_PARAMS" \
    --seed 0 \
    --outfile "$REGEN_FILE" \
    --resume
stop_server

# Failed conversations go to <outfile>.errors.jsonl, not the exit code: check them.
ROWS=$(wc -l < "$REGEN_FILE")
ERR_FILE="${REGEN_FILE%.jsonl}.errors.jsonl"
ERRS=$( [[ -f "$ERR_FILE" ]] && wc -l < "$ERR_FILE" || echo 0 )
echo "Regenerated rows: $ROWS   failed conversations: $ERRS"
if awk -v e="$ERRS" -v r="$ROWS" -v m="$MAX_ERROR_FRAC" 'BEGIN{exit !(r == 0 || e > m * (e + r))}'; then
    echo "Too many failures; rerun this script to resume, or inspect $ERR_FILE" >&2
    exit 1
fi

# Step 2: Prepare data. Rows are already split per turn and tokenized, so no render endpoint.
echo "=== Step 2: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$REGEN_FILE" \
    --output "$DATA_DIR" \
    --seq-length "$SEQ_LENGTH" \
    --overwrite                  # rebuild from the (possibly resumed/extended) regen file

# Step 3: Launch the hidden-state vLLM server with DSpark's target layers
echo "=== Step 3: Launching hidden-state vLLM server ==="
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

# Step 4: Train DSpark against the live vLLM server
echo "=== Step 4: Training ==="
use_env speculators
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
    --save-best \
    --checkpoint-freq 0.1 \
    --on-missing generate

echo "Done. Checkpoints saved to $CKPT_DIR/"
echo "Serve with: vllm serve $MODEL --speculative-config '{\"model\": \"$CKPT_DIR/checkpoint_best\", \"num_speculative_tokens\": $BLOCK_SIZE, \"method\": \"dspark\"}'"
