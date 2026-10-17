#!/bin/bash
# Slurm wrapper around run.sh: ONE job per submit (train [upload]). Submit with submit_chain.sh (from a login node), not with sbatch directly.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run.sh "$@"
