#!/bin/bash
# The long single-node Qwen3-8B / ShareGPT job: Aurora's qwen3-8b-external-with-draft online run (pretrained draft Tengyunw/qwen3_8b_eagle3) on all 44,012 ShareGPT
# conversations, cycled (temperature 1.0, so the replies differ between passes), until Slurm ends the job. GPUs 0,1 = trainer, 2 = vLLM server, 3 idle.
# Checkpoints: $RUN/drafts/step_N (HF-format draft, every sync), $RUN/out/checkpoints/iter_N (resumable). Run dir is fresh per job: runs/aurora_qwen3_8b_online_20261022/long_<jobid>
# Submit by hand: sbatch run_long.sbatch
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI-gb
#SBATCH -t 12:00:00
#SBATCH -J aurora-qwen3-8b-online
#SBATCH -o /scratch/09749/atutej/marin_speculator/runs/aurora_qwen3_8b_online_20261022/slurm-%j.out
set -uo pipefail
W=/scratch/09749/atutej/marin_speculator
RUN=$W/runs/aurora_qwen3_8b_online_20261022/long_$SLURM_JOB_ID
[[ -e $RUN ]] && { echo "$RUN exists" >&2; exit 1; }
[[ -n ${WANDB_API_KEY:-} ]] || eval "$(grep '^export WANDB_API_KEY' ~/.bashrc)"
echo "node=$(hostname) run dir=$RUN"; nvidia-smi -L | head -4
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_aurora WANDB_GROUP=qwen3-8b-sharegpt-online-$SLURM_JOB_ID SYNC_INTERVAL=50   # the trainer and the rollout client log to the same group
cd $W
FOREVER=1 TEMPERATURE=1.0 bash $W/spec-experiments/pipelines/online_drafter/experiments/aurora-repro_qwen3-8b-external-with-draft/run.sh \
    $RUN $W/runs/aurora_repro_qwen3_8b_20261021/data/all \
    training.num_train_steps=100000000 decode.weight_sync_interval=$SYNC_INTERVAL decode.weight_sync_checkpoint_path=$RUN/out/draft_sync training.save_interval=2000 \
    logging.report_to=wandb logging.wandb_team=$WANDB_ENTITY logging.wandb_project=$WANDB_PROJECT logging.wandb_group=$WANDB_GROUP logging.wandb_random_suffix=false logging.wandb_dir=$RUN/wandb
