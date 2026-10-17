#!/bin/bash
# Rebuild this experiment's dataset: a ~100k-row sample (in proportion to Grug's SFT allocation) of the Datakit SFT sources that have stats, clipped to
# 32,768 tokens. Needs the output of the full-mix folder's Datakit command (fetch + stats for the sources in the list below); see ../NOTES.md.
#   bash data/make_dataset.sh            # about 5 minutes on a 144-core node once the stats exist
set -euo pipefail
ROOT=/scratch/09749/atutej/marin_datakit
TOOLS=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data
ROWS=${ROWS:-100000}; SEQ_LENGTH=${SEQ_LENGTH:-32768}; SOURCES=${SOURCES:-$ROOT/shards/shard-1-of-2.txt}   # the 87 sources that had stats on 2026-10-14
OUT=${OUT:-$ROOT/dataset_$((ROWS / 1000))k_seq$SEQ_LENGTH}
source ~/.bashrc 2>/dev/null || true
export HF_TOKEN=${HF_TOKEN:-${HUG_TOKEN:-}} HF_HOME=/scratch/09749/atutej/marin_speculator/cache/hf HF_HUB_OFFLINE=1 MARIN_PREFIX=$ROOT/out TOKENIZERS_PARALLELISM=false
PY=$ROOT/venv/bin/python
$PY -I $TOOLS/build_datakit_sft_mix.py select --stats $ROOT/stats --sources-file "$SOURCES" --rows "$ROWS" --out $ROOT/selection_$((ROWS / 1000))k.json
$PY -I $TOOLS/build_datakit_sft_mix.py write --selection $ROOT/selection_$((ROWS / 1000))k.json --out "$OUT" --seq-length "$SEQ_LENGTH" --shards 8 --workers 64 --shard-workers 8 --allow-partial
set +u; source /scratch/09749/atutej/marin_speculator/spec-experiments/env.sh; conda activate speculators; set -u
python $TOOLS/finalize_dataset.py "$OUT"
echo "dataset: $OUT (the work dir $OUT.work can be deleted)"
