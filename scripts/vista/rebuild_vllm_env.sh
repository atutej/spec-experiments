#!/bin/bash
# Re-run the vllm recipe (reuses the cached wheel in $WORK/wheels, so no recompile).
#   nohup bash scripts/vista/rebuild_vllm_env.sh > rebuild-vllm.log 2>&1 &
set -uo pipefail
REPO_DIR=/scratch/09749/atutej/marin_speculator/spec-experiments
source "$REPO_DIR/env.sh"
bash "$REPO_DIR/setup/setup.sh" --rebuild vllm
