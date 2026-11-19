#!/bin/bash
# Build a Datakit SFT training dataset (Arrow shards of input_ids / loss_mask) from public Hugging Face sources, reproducibly, on any machine.
#
#   export HF_TOKEN=<token>                       # some sources are private/gated
#   bash prepare.sh <TARGET_DIR> [datakit_sft_1005]       # mix: datakit_sft_1005 (default) | datakit_sft_0921
#
# Everything lands under <TARGET_DIR>; nothing outside it is touched:
#   marin/ venv/        Marin checkout at the commit pinned in mixes/pins.json, and its uv environment      (stage env)
#   work/               plan.json, normalized_paths.json, stats/, selection.json                           (stages plan, stats, select)
#   out/  hf_home/      Marin's download/normalize outputs (MARIN_PREFIX) and the Hugging Face cache       (stage fetch)
#   dataset/            the result: data-*.arrow, manifest.json, token_freq.{json,pt}                      (stages write, finalize)
# Every stage skips work that is already finished, so rerunning the same command resumes after a crash or a time limit.
#
# Knobs (environment): STAGES="env plan fetch stats select write finalize" (run a subset), SEQ_LENGTH=32768 (row clip, the trainer's
# --total-seq-len), CONCURRENT=16 (Marin steps at once), WORKERS=<cpus-16> (tokenizing processes), ROWS=<n> (select a ~n-row sample, in proportion
# to the allocations, instead of the full mix; for smoke tests), SOURCES_FILE=<file> (restrict fetch/stats/select to these sources),
# SPECULATORS_PY=<python with torch> (finalize writes token_freq.pt; without it only token_freq.json is written).
set -euo pipefail
TARGET=$(realpath -m "${1:?usage: prepare.sh <TARGET_DIR> [MIX]}"); MIX=${2:-datakit_sft_1005}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGES=${STAGES:-"env plan fetch stats select write finalize"}
SEQ_LENGTH=${SEQ_LENGTH:-32768}; CONCURRENT=${CONCURRENT:-16}
CPUS=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')   # not nproc: OMP_NUM_THREADS=1 would make it report 1
WORKERS=${WORKERS:-$(( CPUS > 32 ? CPUS - 16 : (CPUS / 2 > 0 ? CPUS / 2 : 1) ))}
export HF_TOKEN=${HF_TOKEN:-${HUG_TOKEN:-}} HF_HOME=$TARGET/hf_home MARIN_PREFIX=$TARGET/out TOKENIZERS_PARALLELISM=false
export DATA_MIX=$MIX DATA_ROOT=$TARGET/work
PY=$TARGET/venv/bin/python; BUILD=("$PY" -I "$HERE/build.py")
SRC=(); [[ -n "${SOURCES_FILE:-}" ]] && SRC=(--sources-file "$SOURCES_FILE")
has() { [[ " $STAGES " == *" $1 "* ]]; }
mkdir -p "$TARGET/work"
echo "prepare: mix=$MIX target=$TARGET stages=[$STAGES] cpus=$CPUS workers=$WORKERS $(date)"

has env && bash "$HERE/setup_env.sh" "$TARGET" "$MIX"
if has plan; then
    [[ -s "$TARGET/work/plan.json" ]] || "$PY" -I "$HERE/plan.py" --mix "$MIX" --out "$TARGET/work/plan.json"
    [[ -s "$TARGET/work/normalized_paths.json" ]] || "${BUILD[@]}" paths
fi
has fetch && "${BUILD[@]}" fetch "${SRC[@]}" --max-concurrent "$CONCURRENT"
has stats && "${BUILD[@]}" stats "${SRC[@]}" --out "$TARGET/work/stats" --workers "$WORKERS" --verify-masks 200
if has select; then
    ROWS_FLAG=(); [[ -n "${ROWS:-}" ]] && ROWS_FLAG=(--rows "$ROWS")
    "${BUILD[@]}" select "${SRC[@]}" --stats "$TARGET/work/stats" --out "$TARGET/work/selection.json" "${ROWS_FLAG[@]}"
fi
if has write; then
    PARTIAL=(); [[ -n "${ROWS:-}${SOURCES_FILE:-}" ]] && PARTIAL=(--allow-partial)   # a sample or a source subset is a test, not the full mix
    "${BUILD[@]}" write --selection "$TARGET/work/selection.json" --out "$TARGET/dataset" --seq-length "$SEQ_LENGTH" --workers "$WORKERS" "${PARTIAL[@]}"
fi
if has finalize; then
    if [[ -n "${SPECULATORS_PY:-}" ]]; then "$SPECULATORS_PY" -I "$HERE/finalize.py" "$TARGET/dataset"
    else echo "finalize skipped: set SPECULATORS_PY=<python with torch+datasets> to write token_freq.pt and run the load check (token_freq.json is already there)"; fi
fi
echo "prepare: done $(date)"
