#!/bin/bash
# bash submit.sh <TARGET_DIR> [MIX] [--dry-run]      (run from a LOGIN node)
#   Vista:   PARTITION=gg      ACCOUNT=CCR24067 TIME=2-00:00:00 bash submit.sh <TARGET_DIR>
#   Horizon: PARTITION=debug   ACCOUNT=CCR24067 TIME=12:00:00   bash submit.sh <TARGET_DIR>     (a node-hour is an SU from the shared pool)
# Knobs for prepare.sh (SPECULATORS_PY, CONCURRENT, WORKERS, STAGES, ...) are inherited from the submitting environment.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=$(realpath -m "${1:?usage: submit.sh <TARGET_DIR> [MIX] [--dry-run]}"); MIX=${2:-datakit_sft_1005}
PARTITION=${PARTITION:?set PARTITION (gg on Vista, debug on Horizon)}; ACCOUNT=${ACCOUNT:?set ACCOUNT}; TIME=${TIME:-12:00:00}
LOGS=${LOGS:-$TARGET/logs}; mkdir -p "$LOGS"
export DATA_CURATION_TOOLS="$(dirname "$HERE")"
cmd=(sbatch --parsable -p "$PARTITION" -A "$ACCOUNT" -t "$TIME" -J "data-curation-$MIX" -o "$LOGS/%x-%j.out" "$HERE/prepare.sbatch" "$TARGET" "$MIX")
if [[ "${3:-}" == "--dry-run" ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
