#!/bin/bash
# Tune PERFORMANCE-ONLY knobs for a gb node (4x GB200, 189 GB each) on small data. Nothing here may change
# what is trained: the number of training GPUs stays 2 (it sets the effective batch size).
#   bash tune.sh gen      step 1: client concurrency, and vLLM --max-num-seqs/--max-num-batched-tokens
#   bash tune.sh train    steps 3-4: hidden-state server GPUs, Mooncake buffers, dataloader workers/prefetch
# Needs a 4-GPU gb node. Results are appended to $PROJECT_ROOT/logs/smoke/tune-results.txt as RESULT lines.
#   nohup bash tune.sh gen > "$PROJECT_ROOT/logs/smoke/tune-gen.log" 2>&1 &
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"
use_env() { set +u; conda activate "$1"; set -u; }
source "$EXP_DIR/settings.sh"
[[ "$NUM_GPUS" -ge 4 ]] || { echo "needs a 4-GPU gb node" >&2; exit 1; }
T=${TUNE_DIR:-$PROJECT_ROOT/runs/vista_tune}; R=$PROJECT_ROOT/logs/smoke/tune-results.txt
mkdir -p "$T/source" "$T/gen" "$T/logs"
SERVER_PGID="" MASTER_PGID=""
stop_all() {
    for g in $SERVER_PGID $MASTER_PGID; do kill -TERM -- "-$g" 2>/dev/null; done; sleep 4
    for g in $SERVER_PGID $MASTER_PGID; do kill -KILL -- "-$g" 2>/dev/null; done
    SERVER_PGID="" MASTER_PGID=""; sleep 10
}
trap stop_all EXIT
wait_health() {  # wait_health <pgid> <log>
    local t0=$SECONDS
    until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; do
        kill -0 "$1" 2>/dev/null || { echo "server exited; see $2" >&2; tail -n 20 "$2" >&2; return 1; }
        sleep 5
    done
    echo "server ready after $((SECONDS-t0))s"
}
result() { echo "RESULT $*" | tee -a "$R"; }

phase_gen() {
    local src=$T/source/3000.jsonl
    if [[ ! -f "$src" ]]; then
        use_env speculators
        python "$REPO_DIR/pipelines/speculator_training/tools/export_registry_dataset.py" --dataset "$DATASET" \
            "${EXPORT_ARGS[@]}" --limit 3000 --seed "$SAMPLE_SEED" --out "$src" || return 1
    fi
    [[ -f "$T/gen/slice_00.jsonl" ]] || split -l 400 -d --additional-suffix=.jsonl "$src" "$T/gen/slice_"
    use_env vllm
    run_server() {  # run_server <name> <extra vllm args...>
        local name=$1; shift
        CUDA_VISIBLE_DEVICES=0,1,2,3 setsid vllm serve "$MODEL" --port "$VLLM_PORT" --data-parallel-size 4 \
            --max-model-len "$REGEN_MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" "$@" > "$T/logs/gen_server_$name.log" 2>&1 &
        SERVER_PGID=$!
        wait_health "$SERVER_PGID" "$T/logs/gen_server_$name.log"
    }
    trial() {  # trial <server name> <slice> <concurrency>
        local out=$T/gen/out_$1_c$3_s$2.jsonl; rm -f "$out" "${out%.jsonl}.errors.jsonl"
        use_env speculators
        speculators regenerate-responses --dataset "$T/gen/slice_$2.jsonl" --limit 400 \
            --endpoint "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" --max-tokens "$MAX_GEN_TOKENS" \
            --concurrency "$3" --sampling-params "$SAMPLING_PARAMS" --seed 0 --outfile "$out" > "$T/logs/trial_$1_c$3_s$2.log" 2>&1
        local secs rows errs
        secs=$(tr '\r' '\n' < "$T/logs/trial_$1_c$3_s$2.log" | grep -oE "Pipeline complete in [0-9.]+" | grep -oE "[0-9.]+$")
        rows=$(wc -l < "$out" 2>/dev/null || echo 0); errs=$( [[ -f "${out%.jsonl}.errors.jsonl" ]] && wc -l < "${out%.jsonl}.errors.jsonl" || echo 0 )
        result "gen server=$1 slice=$2 concurrency=$3 rows=$rows errors=$errs secs=${secs:-NA} rows_per_s=$(awk -v r="$rows" -v s="${secs:-0}" 'BEGIN{printf "%.1f", (s>0? r/s : 0)}')"
        use_env vllm
    }
    run_server default || return 1
    trial default 00 256                     # warm-up (CUDA graphs, caches); not comparable
    for c in 512 1024 2048 4096; do
        n=$(printf "%02d" $(( $(echo "512 1024 2048 4096" | tr ' ' '\n' | grep -n "^$c$" | cut -d: -f1) )))
        trial default "$n" "$c"
    done
    stop_all
    run_server big --max-num-seqs 2048 --max-num-batched-tokens 32768 || return 1
    trial big 05 4096
    stop_all
}

phase_train() {
    local rows=$T/train_rows.jsonl
    cat "$T"/gen/out_*.jsonl > "$rows" || return 1
    use_env speculators
    [[ -d "$T/data" ]] || speculators prepare-data --model "$MODEL" --data "$rows" --output "$T/data" --seq-length "$SEQ_LENGTH" --overwrite || return 1
    variant() {  # variant <name> <server gpus> <server dp> <mooncake global GiB> <local GiB> <workers> <prefetch>
        local name=$1 sg=$2 dp=$3 mg=$4 ml=$5 w=$6 pf=$7 log=$T/logs/train_$1.log
        use_env vllm
        setsid mooncake_master --rpc_port "$MOONCAKE_PORT" > "$T/logs/master_$name.log" 2>&1 &
        MASTER_PGID=$!
        until (exec 3<>"/dev/tcp/127.0.0.1/$MOONCAKE_PORT") 2>/dev/null; do sleep 1; done
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$sg" setsid python "$SPEC_DIR/scripts/launch_vllm.py" "$MODEL" \
            --hidden-states-backend mooncake --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
            --mooncake-global-segment-gib "$mg" --mooncake-local-buffer-gib "$ml" --target-layer-ids $TARGET_LAYER_IDS \
            -- --data-parallel-size "$dp" --port "$VLLM_PORT" --gpu-memory-utilization "$GPU_MEM_UTIL" > "$T/logs/hs_server_$name.log" 2>&1 &
        SERVER_PGID=$!
        wait_health "$SERVER_PGID" "$T/logs/hs_server_$name.log" || { stop_all; return 1; }
        use_env speculators
        rm -rf "$T/ckpt_$name"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node 2 -m speculators.train \
            --verifier-name-or-path "$MODEL" --speculator-type "$SPECULATOR_TYPE" --data-path "$T/data" \
            --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" --hidden-states-backend mooncake \
            --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
            --mooncake-global-segment-gib "$mg" --mooncake-local-buffer-gib "$ml" \
            --save-path "$T/ckpt_$name" "${VOCAB_ARGS[@]}" --epochs 1 --max-steps 300 --lr "$LR" --total-seq-len "$SEQ_LENGTH" \
            --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" --num-layers "$NUM_LAYERS" --target-layer-ids $TARGET_LAYER_IDS \
            --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE" --enable-confidence-head --confidence-head-with-markov \
            --loss-fn "$LOSS_FN" --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" --on-missing generate \
            --num-workers "$w" --prefetch-factor "$pf" > "$log" 2>&1
        local rc=$?
        stop_all
        result "train $name rc=$rc server_gpus=$sg dp=$dp mooncake=${mg}/${ml}GiB workers=$w prefetch=$pf $(python3 -I - "$log" <<'PY'
import re, sys, statistics as st
t = open(sys.argv[1], errors="replace").read().replace("\r", "\n")
def vals(k): return [float(x) for x in re.findall(rf"profile/{k}=([0-9.]+(?:e[+-]?[0-9]+)?)", t)]
out = []
for k in ("tokens_per_s", "step_ms", "fetch_frac"):
    v = vals(k); v = v[int(len(v) * 0.4):]          # steady state: drop the first 40%
    out.append(f"{k}_median={st.median(v):.4g} (n={len(v)})" if v else f"{k}=NA")
print(" ".join(out))
PY
)"
    }
    variant base      0,1 2 4  2  12 4
    variant workers   0,1 2 4  2  24 8
    variant buffers   0,1 2 32 8  12 4
    variant server1   0   1 4  2  12 4
}

for p in "$@"; do "phase_$p" || { echo "phase $p failed" >&2; exit 1; }; done
echo "tune finished: $*"
