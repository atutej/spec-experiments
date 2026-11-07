#!/bin/bash
# Slurm wrapper for the training job (TWO gb nodes, 12 h): builds the data view, then runs the itw experiment's two-node DDP script with THIS experiment's settings (EXP_DIR_OVERRIDE).
# Submit with submit_chain.sh from a login node. A job cut by its time limit is continued by submitting again (training resumes from the run's checkpoints).
# It does NOT start if the final epoch checkpoint (the end of epoch 10) already exists, so a chained follow-up job after a job that finished exits at once.
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw
RUN_WORK_DIR=${BIG_POOL_WORK_DIR:-/scratch/09749/atutej/marin_speculator/runs/grug_dspark_itw_drdoom_big_pool}   # NOT named WORK: that is TACC's $WORK, which env.sh builds the conda path from
FINAL=epoch1_end   # EPOCHS=2 (settings.sh): the last end-of-epoch checkpoint
if [[ -e "$RUN_WORK_DIR/checkpoints/$FINAL" ]]; then echo "$FINAL exists: training is complete, nothing to do"; exit 0; fi
last=$(ls -d "$RUN_WORK_DIR"/checkpoints/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -1)
echo "latest step checkpoint: ${last:-none}; training continues towards $FINAL"
bash "$EXP/vista/make_data_view.sh" || exit 1      # CPU, seconds
export EXP_DIR_OVERRIDE="$EXP"
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh train
