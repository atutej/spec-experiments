#!/bin/bash
# Stages of the targetgen chain that are not training:
#   regen   2 gb nodes: each node starts its own Grug vLLM server (4 GPUs, DP4 x EP4, 512 sequences) and regenerates its half of the sampled rows (tools/regenerate_token_rows.py regen).
#           Resumable: parts already in $PARTS_DIR are skipped.
#   merge   CPU (runs at the start of the train job, on its first node): all parts -> the Arrow dataset + token_freq.pt + d2t/t2d (regenerate_token_rows.py merge) in $DATA_DIR.
# Training is vista/run_train.sbatch. Runs as-is on a one-node idev (regen then uses one shard): bash vista/run_regen.sh regen
set -euo pipefail
set -E; trap 'echo "run_regen.sh: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
use_env() { set +u; conda activate "$1"; set -u; }
source "$EXP_DIR/settings.sh"
TOOL="$REPO_DIR/pipelines/speculator_training/tools/regenerate_token_rows.py"
NNODES=${SLURM_NNODES:-1}

if [[ -z "${NODE_ROLE:-}" && "$NNODES" -gt 1 ]]; then      # the batch host: one copy of this script per node
    exec srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 --cpu-bind=none --mem=0 -c "${SLURM_CPUS_ON_NODE:-1}" --export=ALL \
        env NODE_ROLE=1 bash "${BASH_SOURCE[0]}" "$@"
fi
NODE_RANK=${SLURM_NODEID:-0}
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(regen)
mkdir -p "$LOG_DIR" "$WORK_DIR" "$PARTS_DIR"; cd "$WORK_DIR"

SERVER_PGID=""
stop_server() {
    [[ -n "$SERVER_PGID" ]] || return 0
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    for _ in $(seq 60); do kill -0 -- "-$SERVER_PGID" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    SERVER_PGID=""
}
trap stop_server EXIT

import_parts() {   # parts a regen job wrote under another work dir (settings.sh: REGEN_PARTS_SRC) -> PARTS_DIR
    [[ -d "${REGEN_PARTS_SRC:-}" && "$REGEN_PARTS_SRC" != "$PARTS_DIR" ]] || return 0
    mkdir -p "$PARTS_DIR"; cp -an "$REGEN_PARTS_SRC"/. "$PARTS_DIR"/ && echo "parts imported from $REGEN_PARTS_SRC: $(ls "$PARTS_DIR"/part-*.parquet | wc -l) part files in $PARTS_DIR"
}

stage_regen() {
    import_parts
    [[ "$NUM_GPUS" -ge 4 ]] || { echo "regen needs a 4-GPU gb node (NUM_GPUS=$NUM_GPUS)" >&2; exit 1; }
    [[ -f "$SOURCE_DATA/manifest.json" && -f "$SOURCE_DATA/state.json" ]] || { echo "source dataset $SOURCE_DATA is missing or unfinished (no manifest.json / state.json)" >&2; exit 1; }
    use_env vllm
    setsid vllm serve "$MODEL" --port "$VLLM_PORT" --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
        --max-model-len "$REGEN_MODEL_LEN" --max-num-seqs "$REGEN_CONCURRENCY" --gpu-memory-utilization "$GPU_MEM_UTIL" > "$LOG_DIR/regen_vllm_n$NODE_RANK.log" 2>&1 &
    SERVER_PGID=$!
    echo "node $NODE_RANK: waiting for vLLM (log $LOG_DIR/regen_vllm_n$NODE_RANK.log)"
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        kill -0 "$SERVER_PGID" 2>/dev/null || { echo "vLLM exited; see $LOG_DIR/regen_vllm_n$NODE_RANK.log" >&2; tail -n 30 "$LOG_DIR/regen_vllm_n$NODE_RANK.log" >&2; exit 1; }
        sleep 5
    done
    echo "node $NODE_RANK: vLLM ready after ${SECONDS}s; regenerating shard $NODE_RANK of $NNODES"
    use_env speculators
    python "$TOOL" regen --dataset "$SOURCE_DATA" --num "$REGEN_ROWS" --seed "$REGEN_SEED" --shard "$NODE_RANK" --num-shards "$NNODES" \
        --endpoint "http://127.0.0.1:${VLLM_PORT}" --model "$MODEL" --out-dir "$PARTS_DIR" --concurrency "$REGEN_CONCURRENCY" \
        --max-new-tokens "$REGEN_MAX_NEW_TOKENS" --max-total-len "$SEQ_LENGTH" \
        2>&1 | tee -a "$LOG_DIR/regen_n$NODE_RANK.log"
    stop_server
}

stage_merge() {
    import_parts
    # Skip when $DATA_DIR was already merged from exactly these parts (a train job that restarts after its time limit must not rebuild the data it resumes on).
    local parts; parts=$(ls "$PARTS_DIR"/part-*.parquet 2>/dev/null | wc -l)
    if [[ "${FORCE_MERGE:-0}" != 1 && -f "$DATA_DIR/regen_merge.json" ]] && python3 -c "import json,sys; sys.exit(0 if json.load(open('$DATA_DIR/regen_merge.json'))['parts']==$parts else 1)"; then
        echo "merge: $DATA_DIR already holds the merge of these $parts parts; skipping"; return 0
    fi
    use_env speculators
    python "$TOOL" merge --parts-dir "$PARTS_DIR" --out "$DATA_DIR" 2>&1 | tee -a "$LOG_DIR/merge.log"
}

for s in "${STAGES[@]}"; do case "$s" in regen|merge) "stage_$s" ;; *) echo "unknown stage '$s' (regen merge)" >&2; exit 2 ;; esac; done
echo "Stages finished: ${STAGES[*]}"
