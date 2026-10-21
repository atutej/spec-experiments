#!/bin/bash
# Slurm wrapper for the training job (TWO gb nodes, 12 h): builds the data view, then runs the itw experiment's two-node DDP script with THIS experiment's settings (EXP_DIR_OVERRIDE).
# Submit with submit_chain.sh from a login node. A job cut by its time limit is continued by submitting again (training resumes from $CKPT_DIR).
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
bash "$EXP/vista/make_data_view.sh" || exit 1      # CPU, seconds
export EXP_DIR_OVERRIDE="$EXP"
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh train
