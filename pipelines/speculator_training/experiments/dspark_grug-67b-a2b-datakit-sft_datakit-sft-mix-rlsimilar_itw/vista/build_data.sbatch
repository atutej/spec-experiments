#!/bin/bash
# The whole data creation in ONE job on BOTH gb nodes (2 x 4 GPUs, up to 12 h): (1) embedding of the RL prompts and every Datakit shard on all 8 GPUs (embed_node.sh, one per node),
# (2) exact neighbours + round-robin selection and (3) the training dataset, on the first node (the second node is idle for these two short stages).
# Resumable: finished embedding shards, an existing selection and an existing dataset are skipped, so a job cut by the time limit (or an incomplete embedding) is continued by submitting again.
# A stage never writes into a non-empty directory: after a failure in stage 2 or 3 remove the half-written $SEL_DIR or $DATA_DIR yourself (after looking at it), then submit again.
# Submit with submit_data.sh from a login node. Needs the `embed` and `speculators` envs.
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
source ~/.bashrc 2>/dev/null || true   # before `set -u`: bashrc uses unset variables and would abort the job silently
set -uo pipefail
export EXP_DIR=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-rlsimilar_itw
export REPO_DIR=/scratch/09749/atutej/marin_speculator/spec-experiments
set +u; source "$REPO_DIR/env.sh"; set -u
source "$EXP_DIR/settings.sh"
TOOLS="$REPO_DIR/pipelines/speculator_training/tools/select_similar_rows"
mkdir -p "$EMB_DIR"
echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')| EMB_DIR=$EMB_DIR SEL_DIR=$SEL_DIR DATA_DIR=$DATA_DIR SOURCE_DATA=$SOURCE_DATA | $(date)"

# ---- 1. embed ----
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none --mem=0 --export=ALL bash "$EXP_DIR/vista/embed_node.sh"
n=$(ls "$EMB_DIR"/shard-*.done.json 2>/dev/null | wc -l); total=$(ls "$SOURCE_DATA"/data-*.arrow | wc -l); [[ -z "${EMBED_LIMIT_SHARDS:-}" ]] || total=$EMBED_LIMIT_SHARDS
echo "$n of $total shards embedded; RL: $(ls "$EMB_DIR"/rl.done.json 2>/dev/null) | $(date)"
[[ $n -eq $total && -f "$EMB_DIR/rl.done.json" ]] || { echo "embedding incomplete: see $EMB_DIR/worker*.log, then submit again to continue" >&2; exit 1; }

# ---- 2. neighbours and selection ----
DEDUP=(); [[ "$SELECT_DEDUP" != 1 ]] || DEDUP+=(--dedup)
if [[ -f "$SEL_DIR/summary.json" ]]; then echo "selection exists: $SEL_DIR"; else
    set +u; conda activate embed; set -u
    python -I "$TOOLS/select_rows.py" --emb "$EMB_DIR" --out "$SEL_DIR" --topk "$SELECT_TOPK" --token-budget "$SELECT_TOKEN_BUDGET" --seed "$SELECT_SEED" "${DEDUP[@]}" || exit 1
fi

# ---- 3. the training dataset ----
if [[ -f "$DATA_DIR/token_freq.pt" ]]; then echo "dataset exists: $DATA_DIR"; else
    set +u; conda activate speculators; set -u
    python -I "$TOOLS/write_subset.py" --dataset "$SOURCE_DATA" --sel "$SEL_DIR" --out "$DATA_DIR" --seed "$SELECT_SEED" || exit 1
fi
echo "done $(date): the training dataset is $DATA_DIR ; selection summary: $SEL_DIR/summary.json"
