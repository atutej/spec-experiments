#!/bin/bash
# Datakit SFT mixture: fetch (Marin's own chain for the selected sources) -> stats [-> select], on one node.
# Submit with submit_fetch.sh from a LOGIN node (sbatch is refused on compute nodes). Sharded runs (SHARD=k SHARDS=n) handle the k-th of n balanced source
# lists; every shard job ends by calling `select --only-if-complete`, so whichever shard finishes last writes selection.json (no extra job needed: the gb QOS
# allows only 3 submitted jobs). Every stage skips finished work, so resubmitting after a time limit continues where it stopped.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A CCR24067
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
source ~/.bashrc 2>/dev/null || true   # before `set -u`: bashrc uses unset variables and would abort the job silently
set -uo pipefail
HERE=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data
ROOT=/scratch/09749/atutej/marin_datakit
PY=$ROOT/venv/bin/python
export HF_TOKEN=${HF_TOKEN:-${HUG_TOKEN:-}} HF_HOME=$ROOT/hf_home MARIN_PREFIX=${MARIN_PREFIX:-$ROOT/out}
export TOKENIZERS_PARALLELISM=false
CPUS=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')   # not `nproc`: it honors OMP_NUM_THREADS=1 (set in this environment) and would report 1
CONCURRENT=${CONCURRENT:-16}                  # Marin steps run at once (Zephyr LocalClient threads inside each)
WORKERS=${WORKERS:-$(( CPUS > 32 ? CPUS - 16 : (CPUS / 2 > 0 ? CPUS / 2 : 1) ))}   # tokenizing processes for the stats stage
SHARD=${SHARD:-}; SHARDS=${SHARDS:-1}
SHARD_TAG=${SHARD_TAG:-}
SRC=(); [[ -n "$SHARD" ]] && SRC=(--sources-file "$ROOT/shards/shard-$SHARD-of-$SHARDS${SHARD_TAG:+-$SHARD_TAG}.txt")
echo "node $(hostname) | cpus $CPUS | MARIN_PREFIX=$MARIN_PREFIX | shard ${SHARD:-all}/${SHARDS} | $(date)"
$PY -I $HERE/build_datakit_sft_mix.py fetch "${SRC[@]}" --max-concurrent "$CONCURRENT" || { echo "fetch failed or incomplete: resubmit to continue"; exit 1; }
$PY -I $HERE/build_datakit_sft_mix.py stats "${SRC[@]}" --out "$ROOT/stats" --workers "$WORKERS" --verify-masks 200 || exit 1
if [[ -n "$SHARD" ]]; then SELECT_FLAGS=(--only-if-complete); else SELECT_FLAGS=(); fi
$PY -I $HERE/build_datakit_sft_mix.py select --stats "$ROOT/stats" --out "$ROOT/selection.json" "${SELECT_FLAGS[@]}" || exit 1
echo "done $(date)"
