#!/bin/bash
# Slurm wrapper around run.sh: ONE stage per job (export | regen | prepare | train). The partition, time limit
# and job name differ per stage, so submit with submit_chain.sh (from a login node), not with sbatch directly.
# All logic is in run.sh, which also runs as-is on an idev node.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A CCR24067
# Slurm cannot read variables here, and does not create the directory (env.sh does: logs/slurm/).
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_qwen3_0_6b_nemotron_terminal/vista/run.sh "$@"
