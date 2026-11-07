#!/bin/bash
# Smoke test (one reservation node, 1.5 h): the warm80k arm on the R2E-Gym train prompts for 4 optimizer steps (128 rollouts of up to 4,096 new tokens, a draft sync every 2 steps): checks the long-sequence path
# (rollouts, Mooncake sizing, 8.3k-token samples through the DSpark trainer, draft reload). No W&B. Submit by hand: sbatch run_smoke.sbatch
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI-gb
#SBATCH -t 01:30:00
#SBATCH -J aurora-r2e-smoke
#SBATCH -o /scratch/09749/atutej/marin_speculator/runs/aurora_drdoom_r2egym_warm80k_20261101/smoke-slurm-%j.out
set -uo pipefail
W=/scratch/09749/atutej/marin_speculator
RUN=$W/runs/aurora_drdoom_r2egym_warm80k_20261101/smoke_$SLURM_JOB_ID
PROMPTS=$W/runs/r2egym_train_prompts_20261101/prompts_token_ids.jsonl
[[ -e $RUN ]] && { echo "$RUN exists" >&2; exit 1; }
[[ -f $PROMPTS ]] || { echo "missing $PROMPTS (make_prompts.py)" >&2; exit 1; }
echo "node=$(hostname) run dir=$RUN target=Dr. Doom"; nvidia-smi -L | head -4
cd $W
E=$W/spec-experiments/pipelines/online_drafter/experiments
METHOD=dspark SPEC_TOKENS=8 INIT_DRAFT=$W/runs/eval_ckpts/drdoom_targetgen_warm80k_final CONFIG=$E/aurora_grug-67b-a2b-dr-doom_r2egym-train_dspark-online/config.yaml \
    CLIENT_PROMPTS=$PROMPTS REQUESTS=128 STEPS=4 MAX_TOKENS=4096 MAX_PROMPT_TOKENS=4096 MAX_MODEL_LEN=8192 BATCH=32 MOONCAKE_SEGMENT_GB=96 MAX_POOL=64 \
    bash $E/aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_eagle3-online/run.sh \
    $RUN /scratch/09749/atutej/marin_datakit/dataset_seq32768/data-00000-of-00215.arrow \
    decode.weight_sync_interval=2 decode.weight_sync_checkpoint_path=$RUN/out/draft_sync
