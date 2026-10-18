#!/bin/bash
# Runs on ONE node (started once per node by build_data.sbatch with srun): 4 embedding workers, one per GPU. The workers of the two nodes share the Datakit shards: global worker = node * 4 + gpu, of 8.
# Worker 0 embeds the RL prompts first. Exits non-zero if any worker failed. Needs EXP_DIR and REPO_DIR (exported by the batch script) and the `embed` env.
set -uo pipefail
set +u; source "$REPO_DIR/env.sh"; set -u
source "$EXP_DIR/settings.sh"
set +u; conda activate embed; set -u
TOOLS="$REPO_DIR/pipelines/speculator_training/tools/select_similar_rows"
NODE=${SLURM_NODEID:-0}; WORKERS=${EMBED_WORKERS:-8}
EXTRA=(); [[ -z "${EMBED_LIMIT_SHARDS:-}" ]] || EXTRA+=(--limit-shards "$EMBED_LIMIT_SHARDS"); [[ -z "${EMBED_LIMIT_ROWS:-}" ]] || EXTRA+=(--limit-rows "$EMBED_LIMIT_ROWS")
RL_EXTRA=(); [[ -z "${EMBED_LIMIT_RL:-}" ]] || RL_EXTRA+=(--limit "$EMBED_LIMIT_RL")
echo "node $(hostname) (node rank $NODE): 4 workers of $WORKERS, EMB_DIR=$EMB_DIR"
pids=()
for g in 0 1 2 3; do
    w=$((NODE * 4 + g))
    (
        export CUDA_VISIBLE_DEVICES=$g
        if [[ $w -eq 0 ]]; then python -I "$TOOLS/embed_rows.py" rl --out "$EMB_DIR" --rl-dataset "$RL_DATASET" "${RL_EXTRA[@]}" || exit 1; fi
        python -I "$TOOLS/embed_rows.py" datakit --out "$EMB_DIR" --dataset "$SOURCE_DATA" --worker "$w" --num-workers "$WORKERS" "${EXTRA[@]}"
    ) > "$EMB_DIR/worker$w.log" 2>&1 &
    pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
