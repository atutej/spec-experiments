#!/bin/bash
# Build the mixed dataset $SOURCE_DATA (data/build_mix.py: symlinked itw shards + regenerated rows + validation shard). CPU only, runs on the batch host of the train job before training.
# Idempotent: nothing to do when $SOURCE_DATA/manifest.json exists. A leftover from a failed attempt (no manifest.json) inside $WORK_DIR is removed first.   bash vista/run_mix.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
use_env() { set +u; conda activate "$1"; set -u; }
source "$EXP_DIR/settings.sh"
[[ "$SOURCE_DATA" == "$WORK_DIR"/* ]] || { echo "refusing: SOURCE_DATA must be inside WORK_DIR" >&2; exit 1; }
if [[ -f "$SOURCE_DATA/manifest.json" ]]; then echo "mix: $SOURCE_DATA already built; skipping"; exit 0; fi
mkdir -p "$WORK_DIR" "$LOG_DIR"
if [[ -e "$SOURCE_DATA" ]]; then echo "mix: removing the unfinished $SOURCE_DATA"; rm -rf -- "$SOURCE_DATA"; fi   # symlinks only point at the (read only) Datakit shards: rm -rf does not follow them
use_env speculators
python "$EXP_DIR/data/build_mix.py" --out "$SOURCE_DATA" "${MIX_ARGS[@]}" 2>&1 | tee -a "$LOG_DIR/mix.log"
