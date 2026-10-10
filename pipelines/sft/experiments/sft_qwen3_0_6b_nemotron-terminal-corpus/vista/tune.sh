#!/bin/bash
# Throughput tuning of the SFT train stage: for each setting below, run a few steps on an existing dataset (no evaluation, no
# checkpoints worth keeping), then report the steady step time (steps 3 to MAX_STEPS) and the peak GPU memory. Same data and seed for
# every setting. Needs a GPU node with 4 GPUs (idev or a job); runs ~2 min per setting.
#   bash vista/tune.sh [DATASET_DIR]      default: $PROJECT_ROOT/runs/sft_smoke2/data
# Edit CONFIGS to change what is compared: "name VAR=VALUE ..." with the knobs of settings.sh (PER_DEVICE_BATCH, GRAD_CKPT, SAMPLING).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../../env.sh"
DATA=${1:-$PROJECT_ROOT/runs/sft_smoke2/data}
OUT=${TUNE_DIR:-$PROJECT_ROOT/runs/sft_tune}
STEPS=${MAX_STEPS:-8}
CONFIGS=(
    "pdb2_gc_on      PER_DEVICE_BATCH=2 GRAD_CKPT=true  SAMPLING=random"
    "pdb4_gc_on      PER_DEVICE_BATCH=4 GRAD_CKPT=true  SAMPLING=random"
    "pdb8_gc_on      PER_DEVICE_BATCH=8 GRAD_CKPT=true  SAMPLING=random"
    "pdb2_gc_off     PER_DEVICE_BATCH=2 GRAD_CKPT=false SAMPLING=random"
    "pdb4_gc_off     PER_DEVICE_BATCH=4 GRAD_CKPT=false SAMPLING=random"
    "pdb4_gc_on_grp  PER_DEVICE_BATCH=4 GRAD_CKPT=true  SAMPLING=group_by_length"
)
[[ $# -gt 1 ]] && CONFIGS=("${@:2}")
mkdir -p "$OUT"
printf '%-18s %10s %12s %12s  %s\n' config s/step samples/s peak_GiB status | tee "$OUT/summary.txt"
for c in "${CONFIGS[@]}"; do
    read -r name vars <<< "$c"
    wd="$OUT/$name"; rm -rf "$wd"; mkdir -p "$wd"
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1; sleep 2; done > "$wd/mem.txt" ) &
    mon=$!
    # shellcheck disable=SC2086
    env $vars MAX_STEPS=$STEPS GLOBAL_BATCH=64 LOGGING_STEPS=1 EVAL_STEPS=100000 SAVE_STEPS=100000 REPORT_TO=none \
        PREBUILT_DATASET="$DATA" WORK_DIR="$wd" bash "$HERE/run.sh" train > "$wd/run.log" 2>&1
    rc=$?; kill $mon 2>/dev/null; wait $mon 2>/dev/null
    peak=$(sort -n "$wd/mem.txt" | tail -1); peak=$(( ${peak:-0} / 1024 ))
    status=ok; [[ $rc -eq 0 ]] || status="FAILED($(grep -oE 'OutOfMemoryError|out of memory|Error[^ ]*' "$wd/run.log" | head -1))"
    python3 - "$wd/model/trainer_log.jsonl" "$STEPS" "$peak" "$name" "$status" <<'PY' | tee -a "$OUT/summary.txt"
import json, sys
path, steps, peak, name, status = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
def secs(t): h, m, s = t.split(":"); return int(h) * 3600 + int(m) * 60 + int(s)
try:
    rows = {r["current_steps"]: secs(r["elapsed_time"]) for r in map(json.loads, open(path)) if "loss" in r}
    a, b = 3, max(rows)
    sps = (rows[b] - rows[a]) / (b - a)
    print(f"{name:<18} {sps:10.2f} {64 / sps:12.2f} {peak:>12}  {status}")
except Exception as e:
    print(f"{name:<18} {'-':>10} {'-':>12} {peak:>12}  {status} ({type(e).__name__})")
PY
done
