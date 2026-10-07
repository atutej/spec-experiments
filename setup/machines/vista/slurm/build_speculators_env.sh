#!/bin/bash
# Build the `speculators` env on Vista (pure pip, no compile; fine in an idev session on gg).
#   nohup bash setup/machines/vista/slurm/build_speculators_env.sh > "$PROJECT_ROOT/logs/setup/build-speculators.log" 2>&1 &
set -uo pipefail
REPO_DIR=/scratch/09749/atutej/marin_speculator/spec-experiments
source "$REPO_DIR/env.sh"
echo "node=$(hostname) kind=$NODE_KIND"
bash "$REPO_DIR/setup/setup.sh" --rebuild speculators
