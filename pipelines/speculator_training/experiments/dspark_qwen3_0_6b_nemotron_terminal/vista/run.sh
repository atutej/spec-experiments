#!/bin/bash
# Online DSpark training for Qwen3-0.6B on Nemotron-Terminal-Corpus with on-policy regeneration, on Vista.
# The same pipeline as ../genai/run.sh (settings in ../settings.sh), run as stages so each can use the right
# node type (CPU stages on gg, GPU stages on gb) and be restarted on its own:
#
#   bash vista/run.sh [stage ...]       stages: export regen prepare train   (default: all, in order)
#
#   export   step 0    CPU        seeded sample of the dataset -> $WORK_DIR/source/*.jsonl      (skipped if present)
#   regen    step 1    4 GPUs     vLLM data-parallel 4 + `speculators regenerate-responses --resume`
#   prepare  step 2    CPU        `speculators prepare-data`
#   train    steps 3-4 4 GPUs     mooncake_master, hidden-state vLLM on GPUs 0,1, online training on GPUs 2,3
#                                 (resumes from $WORK_DIR/checkpoints if present)
#
# Runs as-is on an idev node (the GPU stages need a gb node). run.sbatch and submit_chain.sh wrap it for Slurm.
# Small test of everything on idev:  REGEN_LIMIT=300 MAX_STEPS=10 WORK_DIR=$PROJECT_ROOT/runs/vista_test bash vista/run.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"   # repo root (this file: pipelines/<pipeline>/experiments/<name>/vista/)
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"
use_env() { set +u; conda activate "$1"; set -u; }   # conda's activate scripts are not nounset-safe

# Vista overrides, set before the shared settings. Only performance knobs: a gb GPU has 189 GB, twice a
# genai H100 NVL's 94 GB, so regeneration can keep twice as many requests in flight. Not measured yet:
# check the `rps` in the regen log of the first full run and tune.
REGEN_CONCURRENCY=${REGEN_CONCURRENCY:-1024}
source "$EXP_DIR/settings.sh"

ALL_GPUS="0,1,2,3" NUM_ALL_GPUS=4         # regen runs alone, so it uses every GPU (a gb node has 4)
VLLM_GPUS="0,1" NUM_VLLM_GPUS=2           # train: hidden-state server ...
TRAIN_GPUS="2,3" NUM_TRAIN_GPUS=2         # ... and training side by side
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(export regen prepare train)
for s in "${STAGES[@]}"; do case "$s" in
    export|prepare) ;;
    regen|train) [[ "$NUM_GPUS" -ge 4 ]] || { echo "stage '$s' needs a 4-GPU gb node (NODE_KIND=$NODE_KIND, NUM_GPUS=$NUM_GPUS)" >&2; exit 1; } ;;
    *) echo "unknown stage '$s' (export regen prepare train)" >&2; exit 2 ;;
esac; done
mkdir -p "$(dirname "$REGEN_FILE")" "$LOG_DIR"
echo "node=$(hostname) kind=$NODE_KIND gpus=$NUM_GPUS stages=${STAGES[*]} WORK_DIR=$WORK_DIR REGEN_LIMIT=$REGEN_LIMIT MAX_STEPS=${MAX_STEPS:-none}"

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
cleanup() { stop_server; stop_master; }
trap cleanup EXIT

stage_export() {
    if [[ -f "$SOURCE_FILE" ]]; then echo "=== Step 0: $SOURCE_FILE exists, skipping ==="; return 0; fi
    echo "=== Step 0: Exporting a random sample of $DATASET ==="
    use_env speculators
    python "$REPO_DIR/pipelines/speculator_training/tools/export_registry_dataset.py" \
        --dataset "$DATASET" "${EXPORT_ARGS[@]}" \
        --limit "$REGEN_LIMIT" --seed "$SAMPLE_SEED" --out "$SOURCE_FILE"
}

stage_regen() {
    [[ -f "$SOURCE_FILE" ]] || { echo "missing $SOURCE_FILE (run the export stage first)" >&2; exit 1; }
    echo "=== Step 1: Regenerating $DATASET responses with $MODEL (concurrency $REGEN_CONCURRENCY) ==="
    use_env vllm
    start_server "$ALL_GPUS" "$LOG_DIR/regen_vllm.log" \
        vllm serve "$MODEL" --port "$VLLM_PORT" \
        --data-parallel-size "$NUM_ALL_GPUS" --max-model-len "$REGEN_MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEM_UTIL"
    use_env speculators
    speculators regenerate-responses \
        --dataset "$SOURCE_FILE" \
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
    local rows errs err_file="${REGEN_FILE%.jsonl}.errors.jsonl"
    rows=$(wc -l < "$REGEN_FILE")
    errs=$( [[ -f "$err_file" ]] && wc -l < "$err_file" || echo 0 )
    echo "Regenerated rows: $rows   failed conversations: $errs"
    if awk -v e="$errs" -v r="$rows" -v m="$MAX_ERROR_FRAC" 'BEGIN{exit !(r == 0 || e > m * (e + r))}'; then
        echo "Too many failures; rerun this stage to resume, or inspect $err_file" >&2
        exit 1
    fi
}

stage_prepare() {
    [[ -f "$REGEN_FILE" ]] || { echo "missing $REGEN_FILE (run the regen stage first)" >&2; exit 1; }
    echo "=== Step 2: Preparing data ==="
    use_env speculators
    # Rows are already split per turn and tokenized, so no render endpoint.
    speculators prepare-data \
        --model "$MODEL" \
        --data "$REGEN_FILE" \
        --output "$DATA_DIR" \
        --seq-length "$SEQ_LENGTH" \
        --overwrite                  # rebuild from the (possibly resumed/extended) regen file
}

stage_train() {
    [[ -d "$DATA_DIR" ]] || { echo "missing $DATA_DIR (run the prepare stage first)" >&2; exit 1; }
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
        --on-missing generate \
        "${TRAIN_EXTRA[@]}"
    echo "Done. Checkpoints saved to $CKPT_DIR/"
    echo "Serve with: vllm serve $MODEL --speculative-config '{\"model\": \"$CKPT_DIR/checkpoint_best\", \"num_speculative_tokens\": $BLOCK_SIZE, \"method\": \"dspark\"}'"
}

for s in "${STAGES[@]}"; do "stage_$s"; done
echo "Stages finished: ${STAGES[*]}"
