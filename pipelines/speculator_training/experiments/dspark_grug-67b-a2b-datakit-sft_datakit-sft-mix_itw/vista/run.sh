#!/bin/bash
# DSpark drafter for Grug-67B-A2B (Datakit SFT 09-21) trained ITW (in-the-wild text, no regeneration) on the full Datakit SFT mixture (a token budget, see settings.sh), on Vista.
# Sibling of ../../dspark_qwen3_0_6b_nemotron-terminal-corpus_itw/vista/run.sh (same pipeline shape: hidden-state vLLM -> Mooncake -> online training).
#
#   bash vista/run.sh [stage ...]       stages: train upload   (default: train)
#
#   train    4-GPU gb   mooncake_master, hidden-state vLLM on GPUs 0,1 (TP=1, DP=2 x expert-parallel: the fork's Grug GPU path requires TP=1), online
#                       training on GPUs 2,3 (resumes from $CKPT_DIR if present). Needs $DATA_DIR: built by the Datakit command, not by this chain
#                       (NOTES.md: select --rows 100000, write --seq-length 32768, finalize). No `prepare` stage: the dataset is already tokenized and masked.
#   upload   CPU        send the best and latest checkpoint to the Hugging Face Hub (also runs after `train` when HF_REPO is set; see settings.sh)
#
# `train` (itw): --on-missing generate means "generate the hidden states" (one prefill per row, max_tokens=1): the server never samples text.
# Runs as-is on an idev node with 4 GPUs. Small test of everything on idev:
#   DATA_DIR=/scratch/09749/atutej/marin_datakit/write_test MAX_STEPS=10 WORK_DIR=<abs path> bash vista/run.sh train
set -euo pipefail
# Never die silently under `set -e` (a batch job's log would stay empty): say which command failed.
set -E; trap 'echo "run.sh: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"   # repo root (this file: pipelines/<pipeline>/experiments/<name>/vista/)
EXP_DIR="${EXP_DIR_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # EXP_DIR_OVERRIDE: another experiment (its settings.sh) trains with this script, e.g. the Dr. Doom sibling
set +u; source "$REPO_DIR/env.sh"; set -u     # env.sh: CC=gcc (Vista's nvidia module sets nvc, which breaks inductor), the caches, PROJECT_ROOT, NUM_GPUS
use_env() { set +u; conda activate "$1"; set -u; }   # conda's activate scripts are not nounset-safe
source "$EXP_DIR/settings.sh"
source "$REPO_DIR/pipelines/speculator_training/tools/upload_drafter.sh"   # stage_upload, upload_after_training

VLLM_GPUS="0,1" NUM_VLLM_GPUS=2           # hidden-state server (DP x EP over these GPUs) ...
TRAIN_GPUS="2,3" NUM_TRAIN_GPUS=2         # ... and training side by side
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(train)
for s in "${STAGES[@]}"; do case "$s" in
    upload) ;;
    train) [[ "$NUM_GPUS" -ge 4 ]] || { echo "stage '$s' needs a 4-GPU gb node (NODE_KIND=$NODE_KIND, NUM_GPUS=$NUM_GPUS)" >&2; exit 1; } ;;
    *) echo "unknown stage '$s' (train upload)" >&2; exit 2 ;;
esac; done
mkdir -p "$LOG_DIR" "$WORK_DIR"
# Work from the run directory, not from wherever the job was submitted (torch's DataLoader workers re-enter the parent's working directory).
cd "$WORK_DIR"
echo "node=$(hostname) cwd=$PWD kind=$NODE_KIND gpus=$NUM_GPUS stages=${STAGES[*]} DATA_DIR=$DATA_DIR SPEC_DIR=$SPEC_DIR MAX_STEPS=${MAX_STEPS:-none}"

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
    sleep 10   # let GPU memory drain
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

start_hidden_state_server() {  # mooncake_master + the hidden-state vLLM on VLLM_GPUS (train only; stopped at exit)
    use_env vllm
    start_master
    # TP=1 + DP + expert parallelism: the fork's Grug GPU path raises "GrugMoE expert-parallel GPU serving requires tensor_parallel_size=1" otherwise.
    # launch_vllm.py comes from $SPEC_DIR (the grug-dspark-gate branch): it falls back to config.json for the layer count (AutoConfig does not know grug_moe).
    # shellcheck disable=SC2086  # TARGET_LAYER_IDS is intentionally word-split
    start_server "$VLLM_GPUS" "$LOG_DIR/hs_vllm.log" \
        python "$SPEC_DIR/scripts/launch_vllm.py" train "$MODEL" \
        --hidden-states-backend mooncake \
        --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
        --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_GIB" --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_GIB" \
        --target-layer-ids $TARGET_LAYER_IDS \
        -- --tensor-parallel-size 1 --data-parallel-size "$NUM_VLLM_GPUS" --enable-expert-parallel --port "$VLLM_PORT" \
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL"
}

# W&B pre-flight: the trainer only imports and logs in at its first log call, so a bad login or entity would crash a
# train job that waited hours for a gb node. Check once, on the CPU, and fall back to offline logging instead of failing.
wandb_preflight() {
    [[ "$LOGGER" == *wandb* ]] || return 0
    use_env speculators
    if timeout 90 python - <<'PY'
import os, sys
try:
    import wandb
    viewer = wandb.Api(timeout=30).viewer
    allowed = {viewer.username, *(getattr(viewer, "teams", None) or [])}
    if os.environ["WANDB_ENTITY"] not in allowed:
        sys.exit(f"entity {os.environ['WANDB_ENTITY']!r} is not one of this login's entities {sorted(allowed)}")
except SystemExit:
    raise
except Exception as e:  # no key, no network, bad credentials ...
    sys.exit(f"W&B check failed: {type(e).__name__}: {str(e)[:150]}")
PY
    then echo "W&B ok: entity $WANDB_ENTITY, project $WANDB_PROJECT, run $RUN_NAME"
    else
        export WANDB_MODE=offline
        echo "WARNING: W&B is not usable from this node; logging OFFLINE under $LOG_DIR/tracker (upload later: wandb sync <run dir>)" >&2
    fi
}

stage_train() {
    [[ -d "$DATA_DIR" && -f "$DATA_DIR/state.json" && -f "$DATA_DIR/token_freq.pt" ]] || {
        echo "missing or unfinished dataset $DATA_DIR (needs state.json and token_freq.pt): build it with the Datakit command, see NOTES.md" >&2; exit 1; }
    use_env speculators
    # The checkout the env imports (its editable install points at $PROJECT_ROOT/speculators) must carry the Grug patch (branch grug-dspark-gate):
    # speculators.utils.hf_configs exists only there. Fail here, in seconds, rather than after the server has loaded 134 GB.
    python -c "import speculators, sys; p = speculators.__file__; print('speculators from', p); import speculators.utils.hf_configs" 2>/dev/null \
        && python -c "import speculators, sys; sys.exit(0 if speculators.__file__.startswith('$SPEC_DIR') else 1)" \
        || { echo "the env's speculators is not the Grug-patched checkout at $SPEC_DIR (branch grug-dspark-gate): git -C $SPEC_DIR switch grug-dspark-gate" >&2; exit 1; }
    echo "=== Step 1: Launching hidden-state vLLM server (Grug, TP=1 x DP=$NUM_VLLM_GPUS x EP) ==="
    start_hidden_state_server
    echo "=== Step 2: Training ==="
    use_env speculators
    # No --save-best: with it the trainer writes NO mid-epoch checkpoints (only at the end of an epoch). --checkpoint-freq 0.1 saves every 10% of
    # the epoch and a rerun of this stage resumes from the last one. --checkpoint-layout step keeps them as checkpoint-<step>/ (written
    # atomically, newest $SAVE_TOTAL_LIMIT kept) so a kill during a save cannot destroy the only copy.
    wandb_preflight
    echo "Metric logging: ${LOGGER:-none}${LOGGER:+ (entity $WANDB_ENTITY, project $WANDB_PROJECT, run $RUN_NAME)}"
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
        --draft-config "$DRAFT_CONFIG" \
        --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
        --mask-token-id "$MASK_TOKEN_ID" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --total-seq-len "$SEQ_LENGTH" \
        --block-size "$BLOCK_SIZE" \
        --max-anchors "$MAX_ANCHORS" \
        --target-layer-ids $TARGET_LAYER_IDS \
        --markov-rank "$MARKOV_RANK" \
        --markov-head-type "$MARKOV_HEAD_TYPE" \
        --enable-confidence-head \
        --confidence-head-with-markov \
        --loss-fn "$LOSS_FN" \
        --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
        --checkpoint-freq "$CKPT_FREQ" \
        --checkpoint-layout step --save-total-limit "$SAVE_TOTAL_LIMIT" \
        --on-missing generate \
        "${LOGGER_ARGS[@]}" \
        "${TRAIN_EXTRA[@]}"
    echo "Done. Checkpoints saved to $CKPT_DIR/"
}

for s in "${STAGES[@]}"; do "stage_$s"; done
upload_after_training "${STAGES[@]}"   # uploads the drafter when this job trained and HF_REPO is set (settings.sh)
echo "Stages finished: ${STAGES[*]}"
