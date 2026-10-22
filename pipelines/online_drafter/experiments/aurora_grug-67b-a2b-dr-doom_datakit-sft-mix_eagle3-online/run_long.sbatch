#!/bin/bash
# The long single-node job: Aurora online EAGLE-3 training for Snowball (Dr. Doom target) from a random draft on Datakit prompts, until Slurm ends the job.
# GPUs 0,1 = trainer, GPU 2 = vLLM capture server, GPU 3 idle. Checkpoints: $RUN/drafts/step_N (HF-format draft, every sync), $RUN/out/checkpoints/iter_N (resumable).
# Submit by hand (not submitted by anything else): sbatch run_long.sbatch      The run dir is fresh per job: runs/aurora_snowball_20261022/long_<jobid>
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI-gb
#SBATCH -t 12:00:00
#SBATCH -J aurora-snowball-online
#SBATCH -o /scratch/09749/atutej/marin_speculator/runs/aurora_snowball_20261022/slurm-%j.out
set -uo pipefail
W=/scratch/09749/atutej/marin_speculator
RUN=$W/runs/aurora_snowball_20261022/long_$SLURM_JOB_ID
[[ -e $RUN ]] && { echo "$RUN exists" >&2; exit 1; }
[[ -n ${WANDB_API_KEY:-} ]] || eval "$(grep '^export WANDB_API_KEY' ~/.bashrc)"
echo "node=$(hostname) run dir=$RUN"; nvidia-smi -L | head -4
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_aurora WANDB_GROUP=snowball-drdoom-datakit-online-$SLURM_JOB_ID SYNC_INTERVAL=50   # the trainer and the rollout client log to the same group
cd $W
FOREVER=1 bash $W/spec-experiments/pipelines/online_drafter/experiments/aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_eagle3-online/run.sh \
    $RUN /scratch/09749/atutej/marin_datakit/dataset_seq32768/data-00000-of-00215.arrow \
    decode.weight_sync_interval=$SYNC_INTERVAL decode.weight_sync_checkpoint_path=$RUN/out/draft_sync training.save_interval=2000 \
    logging.report_to=wandb logging.wandb_team=$WANDB_ENTITY logging.wandb_project=$WANDB_PROJECT logging.wandb_group=$WANDB_GROUP logging.wandb_random_suffix=false logging.wandb_dir=$RUN/wandb
