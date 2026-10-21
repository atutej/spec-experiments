#!/bin/bash
# Make $DATA_DIR: a directory that looks like the Datakit dataset to the trainer, but whose files are symlinks (the 215 Arrow shards) or small copies (metadata, token_freq.pt).
# The source dataset is only read. Safe to rerun (rebuilds the view; refuses a non-empty directory this script did not make).   bash vista/make_data_view.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
source "$EXP_DIR/settings.sh"
echo "source dataset (read only): $SOURCE_DATA"; echo "view to build:              $DATA_DIR"
[[ -f "$SOURCE_DATA/manifest.json" && -f "$SOURCE_DATA/state.json" && -f "$SOURCE_DATA/token_freq.pt" ]] || { echo "the source dataset is missing or unfinished" >&2; exit 1; }
[[ "$DATA_DIR" != "$SOURCE_DATA" && "$DATA_DIR" == "$WORK_DIR"/* ]] || { echo "refusing: DATA_DIR must be a new directory inside WORK_DIR, not the source dataset" >&2; exit 1; }
if [[ -e "$DATA_DIR" ]] && [[ -n "$(ls -A "$DATA_DIR")" ]]; then
    [[ -f "$DATA_DIR/.data_view" ]] || { echo "refusing: $DATA_DIR is not empty and was not made by this script" >&2; exit 1; }
    # keep the trainer's cache files (d2t/t2d) if they exist; rebuild only the links and metadata
    find "$DATA_DIR" -maxdepth 1 \( -type l -o -name 'state.json' -o -name 'dataset_info.json' -o -name 'manifest.json' -o -name 'token_freq.pt' \) -delete
fi
mkdir -p "$DATA_DIR"; echo "view of $SOURCE_DATA, made $(date -Is)" > "$DATA_DIR/.data_view"
n=0; for f in "$SOURCE_DATA"/data-*.arrow; do ln -s "$f" "$DATA_DIR/$(basename "$f")"; n=$((n + 1)); done
for f in state.json dataset_info.json manifest.json token_freq.pt; do cp "$SOURCE_DATA/$f" "$DATA_DIR/$f"; done
echo "linked $n shards, copied state.json dataset_info.json manifest.json token_freq.pt"
