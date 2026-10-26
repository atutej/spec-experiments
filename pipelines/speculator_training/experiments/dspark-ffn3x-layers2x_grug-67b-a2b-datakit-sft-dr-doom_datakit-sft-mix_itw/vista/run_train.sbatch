#!/bin/bash
# Slurm wrapper for the training job (TWO gb nodes, 12 h): prints the target (Dr. Doom), builds the data view, keeps the checkpoints at the comparison steps (KEEP_STEPS) while training, then runs the itw experiment's
# two-node DDP script with THIS experiment's settings (EXP_DIR_OVERRIDE). Submit with submit_chain.sh from a login node. A job cut by its time limit is continued by submitting again (training resumes from the run's checkpoints);
# there is no stop step: the run goes on for as long as jobs are submitted.
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator
EXP=$ROOT/spec-experiments/pipelines/speculator_training/experiments/dspark-ffn3x-layers2x_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
( set +u; source "$ROOT/spec-experiments/env.sh" >/dev/null 2>&1; set -u; EXP_DIR="$EXP"; source "$EXP/settings.sh" || exit 1
  echo "target model: $MODEL"; echo "draft config: $DRAFT_CONFIG ($(python3 -c "import json;c=json.load(open('$DRAFT_CONFIG'));print('intermediate_size', c['intermediate_size'], ', layers', c['num_hidden_layers'])" 2>/dev/null || echo '?'))"; echo "work dir:     $WORK_DIR" ) || exit 1
bash "$EXP/vista/make_data_view.sh" || exit 1      # CPU, seconds
RUN_WORK_DIR=${DRDOOM_FFN3X_L6_WORK_DIR:-$ROOT/runs/grug_dspark_itw_drdoom_ffn3x_l6}   # NOT named WORK: that is TACC's $WORK, which env.sh builds the conda path from
KEEP_STEPS="80813 160860"                                                         # keep in step with settings.sh
keeper() {   # hard-link a finished checkpoint at a comparison step into $RUN_WORK_DIR/keep (the trainer writes checkpoint-N atomically; its rotation later unlinks only its own name)
    while true; do
        for s in $KEEP_STEPS; do
            src="$RUN_WORK_DIR/checkpoints/checkpoint-$s"; dst="$RUN_WORK_DIR/keep/checkpoint-$s"
            if [[ -d "$src" && ! -e "$dst" ]]; then mkdir -p "$RUN_WORK_DIR/keep" && cp -al "$src" "$dst.tmp" && mv "$dst.tmp" "$dst" && echo "kept $dst ($(date +%T))"; fi
        done
        sleep 60
    done
}
keeper & KEEPER=$!
export EXP_DIR_OVERRIDE="$EXP"
bash $ROOT/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh train; rc=$?
kill $KEEPER 2>/dev/null; exit $rc
