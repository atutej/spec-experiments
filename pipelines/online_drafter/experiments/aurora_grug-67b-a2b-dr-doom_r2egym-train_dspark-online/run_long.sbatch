#!/bin/bash
# 4 h job, one reservation node: Aurora online DSpark training for Snowball (the DR. DOOM target) on R2E-Gym TRAIN prompts (single-turn task instructions), warm-started from the Dr. Doom targetgen warm80k drafter. GPUs 0,1 trainer, GPU 2 vLLM server (Step12).
# Rollouts: up to 4,096 new tokens from prompts up to 4,096 tokens (server window 8,192, trainer / Mooncake length 8,320). Drafts land in $RUN/drafts/step_N, resumable checkpoints in $RUN/out/checkpoints.
# Needs the prompts file written by make_prompts.py (see NOTES.md). Submit by hand: sbatch run_long.sbatch
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI-gb
#SBATCH -t 04:00:00
#SBATCH -J aurora-dd-r2e-warm80k
#SBATCH -o /scratch/09749/atutej/marin_speculator/runs/aurora_drdoom_r2egym_warm80k_20261101/slurm-%j.out
set -uo pipefail
W=/scratch/09749/atutej/marin_speculator
RUN=$W/runs/aurora_drdoom_r2egym_warm80k_20261101/long_$SLURM_JOB_ID
PROMPTS=$W/runs/r2egym_train_prompts_20261101/prompts_token_ids.jsonl
[[ -e $RUN ]] && { echo "$RUN exists" >&2; exit 1; }
[[ -f $PROMPTS ]] || { echo "missing $PROMPTS (make_prompts.py)" >&2; exit 1; }
[[ -n ${WANDB_API_KEY:-} ]] || eval "$(grep '^export WANDB_API_KEY' ~/.bashrc)"
echo "node=$(hostname) run dir=$RUN target=Dr. Doom"; nvidia-smi -L | head -4
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_aurora WANDB_GROUP=dspark-drdoom-r2egym-warm80k-$SLURM_JOB_ID SYNC_INTERVAL=8 KEEP_DRAFTS=20   # the trainer and the rollout client log to the same group
cd $W
E=$W/spec-experiments/pipelines/online_drafter/experiments
TARGET=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom FOREVER=1 METHOD=dspark SPEC_TOKENS=8 INIT_DRAFT=$W/runs/eval_ckpts/drdoom_targetgen_warm80k_final CONFIG=$E/aurora_grug-67b-a2b-dr-doom_r2egym-train_dspark-online/config.yaml DRAFT_CONFIG=${DRAFT_CONFIG:-$E/aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_dspark-online/draft_config.json} \
    CLIENT_PROMPTS=$PROMPTS MAX_TOKENS=4096 MAX_PROMPT_TOKENS=4096 MAX_MODEL_LEN=8192 BATCH=32 MOONCAKE_SEGMENT_GB=96 MAX_POOL=64 KEEP_DRAFTS=$KEEP_DRAFTS \
    bash $E/aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_eagle3-online/run.sh \
    $RUN /scratch/09749/atutej/marin_datakit/dataset_seq32768/data-00000-of-00215.arrow \
    decode.weight_sync_interval=$SYNC_INTERVAL decode.weight_sync_checkpoint_path=$RUN/out/draft_sync training.save_interval=125 \
    logging.report_to=wandb logging.wandb_team=$WANDB_ENTITY logging.wandb_project=$WANDB_PROJECT logging.wandb_group=$WANDB_GROUP logging.wandb_random_suffix=false logging.wandb_dir=$RUN/wandb
