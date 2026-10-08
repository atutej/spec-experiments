#!/bin/bash
# Does a longer packed sequence (--total-seq-len) use more of a gb GPU, and is it faster per token?
# Same data (the rows from tune.sh gen), prepared once per length, trained for STEPS steps per length on GPUs 2,3 with
# the hidden-state server on GPUs 0,1 (one server for all lengths), sampling nvidia-smi. --max-anchors is scaled with
# the length (anchors per token constant). NOT a pure performance knob: it changes what is trained (clipping, tokens
# per step); use it to decide, not to configure. Needs a 4-GPU gb node and runs/vista_tune_b/train_rows.jsonl.
#   LENGTHS="8192 16384 32768" STEPS=100 bash tune_seqlen.sh
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"
use_env() { set +u; conda activate "$1"; set -u; }
source "$EXP_DIR/settings.sh"
[[ "$NUM_GPUS" -ge 4 ]] || { echo "needs a 4-GPU gb node" >&2; exit 1; }
LENGTHS=(${LENGTHS:-8192 16384 32768}); STEPS=${STEPS:-100}
T=${TUNE_DIR:-$PROJECT_ROOT/runs/vista_tune_b}; ROWS=$T/train_rows.jsonl; OUT=$PROJECT_ROOT/logs/smoke/tune-seqlen; R=$PROJECT_ROOT/logs/smoke/tune-results.txt
[[ -f "$ROWS" ]] || { echo "missing $ROWS (run tune.sh gen train first)" >&2; exit 1; }
mkdir -p "$OUT"; rm -f "$OUT"/*
SERVER="" MASTER="" SAMPLER=""
stop() { for g in $SERVER $MASTER; do kill -TERM -- "-$g" 2>/dev/null; done; [[ -n "$SAMPLER" ]] && kill "$SAMPLER" 2>/dev/null; sleep 5
         for g in $SERVER $MASTER; do kill -KILL -- "-$g" 2>/dev/null; done; }
trap stop EXIT

use_env speculators
for L in "${LENGTHS[@]}"; do
    [[ -d "$T/data_$L" ]] || speculators prepare-data --model "$MODEL" --data "$ROWS" --output "$T/data_$L" --seq-length "$L" --overwrite > "$OUT/prepare_$L.log" 2>&1 || { echo "prepare $L failed"; exit 1; }
done
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
echo "server ready"

for L in "${LENGTHS[@]}"; do
    A=$(( MAX_ANCHORS * L / 8192 ))
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i 2,3 -l 1 > "$OUT/gpu_$L.csv" & SAMPLER=$!
    use_env speculators
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node 2 -m speculators.train \
        --verifier-name-or-path "$MODEL" --speculator-type "$SPECULATOR_TYPE" --data-path "$T/data_$L" \
        --vllm-endpoint "http://localhost:$VLLM_PORT/v1" --hidden-states-backend mooncake \
        --mooncake-master "127.0.0.1:$MOONCAKE_PORT" --mooncake-protocol tcp \
        --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_GIB" --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_GIB" \
        --save-path "$OUT/ckpt_$L" "${VOCAB_ARGS[@]}" --epochs 1 --max-steps "$STEPS" --lr "$LR" --total-seq-len "$L" \
        --block-size "$BLOCK_SIZE" --max-anchors "$A" --num-layers "$NUM_LAYERS" --target-layer-ids $TARGET_LAYER_IDS \
        --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE" --enable-confidence-head --confidence-head-with-markov \
        --loss-fn "$LOSS_FN" --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" --on-missing generate > "$OUT/train_$L.log" 2>&1
    rc=$?; kill "$SAMPLER" 2>/dev/null; SAMPLER=""; use_env vllm
    echo "RESULT seqlen=$L anchors=$A rc=$rc $(python3 -I - "$OUT/train_$L.log" "$OUT/gpu_$L.csv" "$ROWS" "$L" <<'PY'
import csv, json, re, statistics as st, sys
log, gpu, rows, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
t = open(log, errors="replace").read().replace("\r", "\n")
def med(k):
    v = [float(x) for x in re.findall(rf"profile/{k}=([0-9.]+(?:e[+-]?[0-9]+)?)", t)]
    v = v[int(len(v) * 0.4):]
    return f"{st.median(v):.4g}" if v else "NA"
mem, util = [], []
for r in csv.reader(open(gpu)):
    if len(r) == 3: mem.append(float(r[1])); util.append(float(r[2]))
busy = [u for u in util if u > 5]
tot = clip = sup = supclip = 0
with open(rows, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line); n = len(d["input_ids"]); s = sum(d["loss_mask"]); m = sum(d["loss_mask"][:L])
        tot += 1; clip += n > L; sup += s; supclip += s - m
print(f"tokens_per_s_median={med('tokens_per_s')} step_ms_median={med('step_ms')} fetch_frac_median={med('fetch_frac')} "
      f"trainer_peak_mem_GiB={max(mem)/1024:.1f} util_when_busy={sum(busy)/len(busy) if busy else 0:.0f}% "
      f"rows_over_L={clip}/{tot} supervised_tokens_lost={100*supclip/sup:.2f}%")
PY
)" | tee -a "$R"
done
echo "tune_seqlen finished"
