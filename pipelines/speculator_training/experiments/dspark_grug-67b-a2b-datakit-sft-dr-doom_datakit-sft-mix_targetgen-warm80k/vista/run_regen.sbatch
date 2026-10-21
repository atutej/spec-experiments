#!/bin/bash
# Slurm wrapper for the pool + regen stages (TWO gb nodes, one vLLM server and one shard each; the row pool is built first, on the batch node). Submit with submit_chain.sh from a login node.
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k
NODE_ROLE=1 bash "$EXP/vista/run_regen.sh" pool || exit 1   # the row pool, on this node only (no srun fan-out); skipped when it exists
exec bash "$EXP/vista/run_regen.sh" "$@"
