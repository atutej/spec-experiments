#!/bin/bash
# Build the Marin "datakit" environment the data_curation tools run in: a Marin checkout at a pinned commit plus a uv venv.
#   bash setup_env.sh <TARGET_DIR> <MIX>       MIX = datakit_sft_1005 | datakit_sft_0921
# Creates <TARGET_DIR>/{marin,venv}. Safe to rerun: a finished step is skipped. Needs git, uv, network access.
set -euo pipefail
TARGET=$(realpath -m "${1:?usage: setup_env.sh <TARGET_DIR> <MIX>}"); MIX=${2:?mix name}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=$(python3 -I - "$HERE/../mixes/pins.json" "$MIX" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]]["marin_commit"])
PY
)
mkdir -p "$TARGET"; cd "$TARGET"
if [[ ! -d marin/.git ]]; then git init -q marin && git -C marin remote add origin https://github.com/marin-community/marin.git; fi
if [[ "$(git -C marin rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]]; then
    git -C marin fetch -q --depth 1 origin "$COMMIT" && git -C marin checkout -q FETCH_HEAD
fi
echo "marin at $(git -C marin rev-parse HEAD)"
# Some clusters put a vendor compiler first on PATH (TACC aarch64: NVIDIA nvc), which cannot build the sdist-only wheels (pycosat); use gcc.
export CC=${CC_OVERRIDE:-gcc} CXX=${CXX_OVERRIDE:-g++}
if [[ ! -f venv/.verified-$COMMIT ]]; then
    # marin-core does not declare marin-dupekit, but marin.datakit.chat_normalize imports it, so sync both workspace packages.
    ( cd marin && UV_PROJECT_ENVIRONMENT="$TARGET/venv" UV_LINK_MODE=copy uv sync --frozen --package marin-core --package marin-dupekit --python 3.12 )
    "$TARGET/venv/bin/python" -I -c "import marin.datakit.sft_sources, marin.datakit.chat_normalize, levanter; print('env ok')"
    touch "venv/.verified-$COMMIT"    # only after the import check passed
fi
echo "env ready: $TARGET/venv"
