#!/bin/bash
# Slurm wrapper for serve_speedbench.sh (one gb node, arms run one after the other). Submit with submit_speedbench_figs12.sh from a login node; the settings travel in the environment.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/tools/eval_acceptance/serve_speedbench.sh
