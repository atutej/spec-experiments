#!/bin/bash
# Slurm wrapper for the regen stage (TWO gb nodes, one vLLM server and one shard each). Submit with submit_chain.sh from a login node.
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen/vista/run_regen.sh "$@"
