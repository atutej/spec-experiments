#!/bin/bash
# Build a SMALL dataset with the real pipeline, on THIS node (needs a GPU; run it in an allocation): the first 3,000 RL prompts, 2 Datakit shards x 3,000 rows, a 3.5e7-token budget, 4 output shards.
# Everything goes to a fresh directory (default runs/grug_rlsimilar_smoke; RLSIM_BUILD_DIR to change it) which this script refuses to reuse. Then run smoke_train.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_DIR="$(cd "$HERE/.." && pwd)"; REPO_DIR="$(cd "$HERE/../../../../.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
export RLSIM_BUILD_DIR="${RLSIM_BUILD_DIR:-$PROJECT_ROOT/runs/grug_rlsimilar_smoke}"
if [[ "$RLSIM_BUILD_DIR" == "$PROJECT_ROOT/runs/grug_rlsimilar_data" ]]; then echo "refusing: that is the full run's directory" >&2; exit 1; fi
if [[ -e "$RLSIM_BUILD_DIR" ]]; then echo "refusing: $RLSIM_BUILD_DIR exists" >&2; exit 1; fi
source "$EXP_DIR/settings.sh"
echo "smoke build dir: $BUILD_DIR (emb $EMB_DIR, sel $SEL_DIR, dataset $DATA_DIR)"
TOOLS="$REPO_DIR/pipelines/speculator_training/tools/select_similar_rows"
use_env() { set +u; conda activate "$1"; set -u; }
use_env embed
python -I "$TOOLS/embed_rows.py" rl --out "$EMB_DIR" --rl-dataset "$RL_DATASET" --limit 3000 || exit 1
python -I "$TOOLS/embed_rows.py" datakit --out "$EMB_DIR" --dataset "$SOURCE_DATA" --limit-shards 2 --limit-rows 3000 || exit 1
python -I "$TOOLS/select_rows.py" --emb "$EMB_DIR" --out "$SEL_DIR" --topk 8 --token-budget 3.5e7 --seed "$SELECT_SEED" || exit 1
use_env speculators
python -I "$TOOLS/write_subset.py" --dataset "$SOURCE_DATA" --sel "$SEL_DIR" --out "$DATA_DIR" --seed "$SELECT_SEED" --num-shards 4 || exit 1
echo "smoke dataset: $DATA_DIR"
