#!/bin/bash
# Slurm wrapper for the training job (TWO gb nodes, 12 h): checks that the dataset exists, then runs the itw experiment's two-node DDP script with THIS experiment's settings (EXP_DIR_OVERRIDE).
# Submit with submit_chain.sh from a login node. A job cut by its time limit is continued by submitting again (training resumes from $CKPT_DIR).
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-rlsimilar_itw
DATA=${RLSIM_BUILD_DIR:-/scratch/09749/atutej/marin_speculator/runs/grug_rlsimilar_data}/dataset
[[ -f "$DATA/state.json" && -f "$DATA/token_freq.pt" ]] || { echo "the training dataset $DATA is missing or unfinished (submit_data.sh builds it)" >&2; exit 1; }
export EXP_DIR_OVERRIDE="$EXP"
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh train
