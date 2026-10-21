#!/bin/bash
# Slurm wrapper for the merge + training stages (the merge of the regenerated parts runs first, on the batch node, ~0.5 h; the other node waits): the itw experiment's two-node DDP script, run with THIS experiment's settings (EXP_DIR_OVERRIDE).
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
export EXP_DIR_OVERRIDE=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k
NODE_ROLE=1 bash $EXP_DIR_OVERRIDE/vista/run_regen.sh merge || exit 1   # NODE_ROLE=1: run on this node only, no srun fan-out
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh train
