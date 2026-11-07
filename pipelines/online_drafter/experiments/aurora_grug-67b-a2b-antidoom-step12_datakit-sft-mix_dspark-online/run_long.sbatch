#!/bin/bash
# The long single-node job continuing a DSpark drafter (default: our Dr. Doom targetgen warm80k checkpoint) online on Datakit prompts, target Antidoom RLVR1 Step12, until Slurm ends the job.
# Sync/save intervals are in optimizer steps of 32 samples (draft_accumulation_steps 4): sync every 10 steps = 320 samples (the first run: 50 steps of 8 = 400), full checkpoint every 500.
# Same layout as ../aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_eagle3-online/run_long.sbatch (GPUs 0,1 trainer, GPU 2 server). Submit by hand: sbatch run_long.sbatch
# Drafts land in $RUN/drafts/step_N (HF-style state dict: pytorch_model.bin + config.json, loadable by Aurora's DSparkDraftModel and by vLLM's DSpark drafter).
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI-gb
#SBATCH -t 12:00:00
#SBATCH -J aurora-s12-dk-warm80k
#SBATCH -o /scratch/09749/atutej/marin_speculator/runs/aurora_antidoom_step12_datakit_warm80k_20261026/slurm-%j.out
set -uo pipefail
W=/scratch/09749/atutej/marin_speculator
RUN=$W/runs/aurora_antidoom_step12_datakit_warm80k_20261026/long_$SLURM_JOB_ID
[[ -e $RUN ]] && { echo "$RUN exists" >&2; exit 1; }
[[ -n ${WANDB_API_KEY:-} ]] || eval "$(grep '^export WANDB_API_KEY' ~/.bashrc)"
echo "node=$(hostname) run dir=$RUN"; nvidia-smi -L | head -4
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_aurora WANDB_GROUP=dspark-step12-warm80k-lr1e-5-datakit-online-$SLURM_JOB_ID SYNC_INTERVAL=10 KEEP_DRAFTS=10   # the trainer and the rollout client log to the same group
cd $W
E=$W/spec-experiments/pipelines/online_drafter/experiments
TARGET=open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02 FOREVER=1 KEEP_DRAFTS=$KEEP_DRAFTS METHOD=dspark SPEC_TOKENS=8 INIT_DRAFT=${INIT_DRAFT:-$W/runs/eval_ckpts/drdoom_targetgen_warm80k_final} CONFIG=$E/aurora_grug-67b-a2b-antidoom-step12_datakit-sft-mix_dspark-online/config.yaml \
    bash $E/aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_eagle3-online/run.sh \
    $RUN /scratch/09749/atutej/marin_datakit/dataset_seq32768/data-00000-of-00215.arrow \
    decode.weight_sync_interval=$SYNC_INTERVAL decode.weight_sync_checkpoint_path=$RUN/out/draft_sync training.save_interval=500 \
    logging.report_to=wandb logging.wandb_team=$WANDB_ENTITY logging.wandb_project=$WANDB_PROJECT logging.wandb_group=$WANDB_GROUP logging.wandb_random_suffix=false logging.wandb_dir=$RUN/wandb
