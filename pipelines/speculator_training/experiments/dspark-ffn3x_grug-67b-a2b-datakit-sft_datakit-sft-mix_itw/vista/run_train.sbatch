#!/bin/bash
# Slurm wrapper for the training job (TWO gb nodes, 12 h): builds the data view, then runs the itw experiment's two-node DDP script with THIS experiment's settings (EXP_DIR_OVERRIDE).
# Submit with submit_chain.sh from a login node. A job cut by its time limit is continued by submitting again (training resumes from the run's checkpoints).
# It does NOT start if a checkpoint at or past STOP_AFTER_STEP (80,813, the comparison point with the itw run) already exists, so a chained follow-up job after a job that got there exits at once.
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark-ffn3x_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw
RUN_WORK_DIR=${FFN3X_WORK_DIR:-/scratch/09749/atutej/marin_speculator/runs/grug_dspark_itw_ffn3x}   # NOT named WORK: that is TACC's $WORK, which env.sh builds the conda path from
STOP=80813
last=$(ls -d "$RUN_WORK_DIR"/checkpoints/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
if [[ -n "$last" && "$last" -ge "$STOP" ]]; then echo "a checkpoint at step $last (>= $STOP) exists: nothing to do"; exit 0; fi
echo "latest checkpoint step: ${last:-none}; training continues towards step $STOP"
bash "$EXP/vista/make_data_view.sh" || exit 1      # CPU, seconds
export EXP_DIR_OVERRIDE="$EXP"
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh train
