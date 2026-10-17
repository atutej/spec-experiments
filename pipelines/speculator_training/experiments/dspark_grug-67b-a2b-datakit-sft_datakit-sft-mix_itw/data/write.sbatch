#!/bin/bash
# Datakit SFT mixture -> the trainer's dataset directory: `write` (datakit env) then `finalize_dataset.py` (speculators env), on one CPU node.
# Submit with submit_write.sh from a LOGIN node. Needs the output of submit_fetch.sh (selection.json). Resubmitting continues: tokenized files and
# finished shards are skipped.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A CCR24067
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
source ~/.bashrc 2>/dev/null || true   # before `set -u`: bashrc uses unset variables and would abort the job silently
set -uo pipefail
HERE=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data
ROOT=/scratch/09749/atutej/marin_datakit
SEQ_LENGTH=${SEQ_LENGTH:?set SEQ_LENGTH (the --total-seq-len of the training run), e.g. SEQ_LENGTH=32768}
OUT=${OUT:-$ROOT/dataset_seq$SEQ_LENGTH}
export HF_TOKEN=${HF_TOKEN:-${HUG_TOKEN:-}} HF_HOME=/scratch/09749/atutej/marin_speculator/cache/hf MARIN_PREFIX=${MARIN_PREFIX:-$ROOT/out} TOKENIZERS_PARALLELISM=false
echo "node $(hostname) | cpus $(python3 -c "import os; print(len(os.sched_getaffinity(0)))") | SEQ_LENGTH=$SEQ_LENGTH OUT=$OUT | $(date)"
$ROOT/venv/bin/python -I $HERE/build_datakit_sft_mix.py write --selection "$ROOT/selection.json" --out "$OUT" --seq-length "$SEQ_LENGTH" \
    --workers "${WORKERS:-32}" --shard-workers "${SHARD_WORKERS:-8}" || { echo "write failed or incomplete: resubmit to continue"; exit 1; }
set +u; source /scratch/09749/atutej/marin_speculator/spec-experiments/env.sh; conda activate speculators; set -u
python $HERE/finalize_dataset.py "$OUT" || exit 1
echo "done $(date); the work dir ${OUT}.work can be deleted"
