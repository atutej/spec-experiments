#!/bin/bash
# Slurm wrapper around eval.sh (one gb GPU; the node is not shared). All logic is in eval.sh, which also runs on an idev node.
# Submit with submit_eval.sh from a login node.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A CCR24067
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy/vista/eval.sh "$@"
