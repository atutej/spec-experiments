#!/bin/bash
# Slurm wrapper around run_2node_ddp.sh: ONE two-node job per submit. Submit with RUN_SBATCH=run_2node_ddp.sbatch bash submit_chain.sh (from a login node).
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh "$@"
