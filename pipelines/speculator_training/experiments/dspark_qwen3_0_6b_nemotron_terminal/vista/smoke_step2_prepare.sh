#!/bin/bash
# Smoke test of step 2 (`speculators prepare-data`) on the step 1 smoke output. CPU only.
#   nohup bash <this file> > "$PROJECT_ROOT/logs/smoke/step2-prepare.log" 2>&1 &
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "$REPO_DIR/env.sh"
MODEL="Qwen/Qwen3-0.6B" SEQ_LENGTH=8192
REGEN=$PROJECT_ROOT/runs/smoke_step1/regen.jsonl
OUT=$PROJECT_ROOT/runs/smoke_step2/data
[[ -f "$REGEN" ]] || { echo "missing $REGEN (run smoke_step1_regen.sh first)" >&2; exit 1; }
set +u; conda activate speculators || exit 1; set -u
time speculators prepare-data --model "$MODEL" --data "$REGEN" --output "$OUT" \
    --seq-length "$SEQ_LENGTH" --overwrite
rc=$?
echo "prepare-data exit code $rc"; ls -la "$OUT" | head; du -sh "$OUT"
exit $rc
