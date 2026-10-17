#!/bin/bash
# Rollout acceptance length of this experiment's drafter on held-out prompts (target Qwen3-0.6B generates, the draft proposes;
# see ../../../tools/eval_acceptance/). One GPU on a gb node; runs as-is on an idev node. eval.sbatch / submit_eval.sh wrap it.
#
#   bash vista/eval.sh [CHECKPOINT_NAME]        default 0 = $WORK_DIR/checkpoints/0 (the end-of-epoch checkpoint every run has)
#
# Results: $WORK_DIR/eval_rollout/<checkpoint name>/ (acceptance.csv, results.json, completions_*.jsonl, server.log) and a W&B
# run named <experiment>_rollout_eval_<checkpoint name> (entity atutej, project marin_speculator). The prompts are built once by
# tools/eval_acceptance/build_prompts.py into $PROJECT_ROOT/runs/eval_prompts_500 and must exist. MAX_PROMPTS=8 MAX_TOKENS=1024
# gives a quick test (without W&B: WANDB_NAME= ).
set -euo pipefail
set -E; trap 'echo "eval.sh: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_DIR/env.sh"
source "$EXP_DIR/settings.sh"

CKPT_NAME=${1:-0}
export WANDB_NAME=${WANDB_NAME-${RUN_NAME}_rollout_eval_${CKPT_NAME}}
exec bash "$REPO_DIR/pipelines/speculator_training/tools/eval_acceptance/run_eval.sh" \
    "$WORK_DIR/checkpoints/$CKPT_NAME" "$WORK_DIR/eval_rollout/$CKPT_NAME"
