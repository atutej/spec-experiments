#!/bin/bash
# Build the mixed dataset (data/build_mix.py) on a compute node: it rewrites the 9 BigMix-TG shards (about 38 GB) into the Datakit layout. CPU/IO only; about 10-30 min. One node of the debug partition.
#   sbatch horizon/build_data.sbatch                                    the real dataset into $DATA_DIR (default $WORK_DIR/data)
#   SMOKE_DATA=1 sbatch horizon/build_data.sbatch                       a small one (2 Datakit shards + 1 BigMix-TG shard) into ${WORK_DIR}_smoke/data, for smoke tests
#SBATCH -J build-mix-data
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP=${EXP_DIR:?submit with: EXP_DIR=<experiment dir> sbatch horizon/build_data.sbatch (or submit.sh data)}
REPO_DIR=$(cd "$EXP/../../../.." && pwd)
set +u; source "$REPO_DIR/env.sh"; set -u
EXP_DIR=$EXP; source "$EXP/settings.sh"
export PATH="$PROJECT_ROOT/envs/speculators/bin:$PATH"
if [[ "${SMOKE_DATA:-0}" == 1 ]]; then OUT="${WORK_DIR}_smoke/data"; EXTRA=(--max-orig-shards 2 --max-tg-shards 1 --val-orig-rows 300 --val-tg-rows 30); else OUT="$DATA_DIR"; EXTRA=(); fi
echo "node=$(hostname) building $OUT"
python "$EXP/data/build_mix.py" --out "$OUT" "${EXTRA[@]}"
