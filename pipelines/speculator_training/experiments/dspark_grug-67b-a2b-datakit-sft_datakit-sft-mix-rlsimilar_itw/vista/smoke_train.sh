#!/bin/bash
# Ten training steps on the smoke dataset (smoke_data.sh), on THIS node (4 GPUs: 2 serve hidden states, 2 train): checks that the trainer accepts the filtered dataset. No W&B, no upload.
# Writes only under runs/grug_dspark_itw_rlsimilar_smoke (fresh; refused if it exists).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_DIR_OVERRIDE="$(cd "$HERE/.." && pwd)"
ROOT=/scratch/09749/atutej/marin_speculator
export RLSIM_BUILD_DIR="${RLSIM_BUILD_DIR:-$ROOT/runs/grug_rlsimilar_smoke}"
export RLSIM_WORK_DIR="${RLSIM_WORK_DIR:-$ROOT/runs/grug_dspark_itw_rlsimilar_smoke}"
for d in "$RLSIM_BUILD_DIR" "$RLSIM_WORK_DIR"; do case "$d" in "$ROOT/runs/grug_rlsimilar_data"|"$ROOT/runs/grug_dspark_itw_rlsimilar"|"$ROOT/runs/grug_dspark_full") echo "refusing: $d is a real run's directory" >&2; exit 1;; esac; done
[[ -f "$RLSIM_BUILD_DIR/dataset/token_freq.pt" ]] || { echo "no smoke dataset in $RLSIM_BUILD_DIR/dataset: run smoke_data.sh first" >&2; exit 1; }
[[ ! -e "$RLSIM_WORK_DIR" ]] || { echo "refusing: $RLSIM_WORK_DIR exists" >&2; exit 1; }
echo "smoke train: data $RLSIM_BUILD_DIR/dataset, work dir $RLSIM_WORK_DIR"
SMOKE=1 MAX_STEPS=${MAX_STEPS:-10} exec bash "$ROOT/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run.sh" train
