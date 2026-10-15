#!/bin/bash
# Slurm wrapper around run.sh. Submit from a login node: sbatch -p gb vista/run.sbatch plain dspark
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A CCR24067
#SBATCH -t 02:00:00
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/rl_speculation/experiments/dspark_grug-67b-a2b-datakit-sft_r2egym_serving-smoke/vista/run.sh "$@"
