#!/bin/bash
# Build the `speculators` env (trainer, data prep) on Horizon. Pure pip, no compile; about 15-30 min.
#   sbatch setup/machines/horizon/slurm/build_speculators_env.sbatch        (from a login node)
# Builds just this env (setup.sh would also clone every repo in repos.txt, some over ssh); clones `speculators` over https for hs_connectors and the trainer.
#SBATCH -J build-speculators
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out

set -uo pipefail
export MACHINE=horizon
REPO_DIR=${REPO_DIR:-/scratch/09749/atutej/marin_speculator/spec-experiments}
source "$REPO_DIR/env.sh"
echo "node=$(hostname) kind=$NODE_KIND gpus=$NUM_GPUS"
for h in repo.anaconda.com pypi.org download.pytorch.org github.com; do
    ok=0; for i in 1 2 3 4 5 6; do getent hosts $h >/dev/null && { ok=1; break; }; sleep 10; done
    [[ $ok -eq 1 ]] || { echo "no DNS for $h on $(hostname); aborting" >&2; exit 1; }
done
if [[ ! -d "$PROJECT_ROOT/speculators/.git" ]]; then
    git clone --branch grug-dspark-gate https://github.com/atutej/speculators.git "$PROJECT_ROOT/speculators" </dev/null || exit 1
fi
conda tos accept >/dev/null 2>&1 || true
conda_env_exists() { conda env list | awk '{print $1}' | grep -qx "$1" || [[ -d "${CONDA_ENVS_PATH:-/nonexistent}/$1" ]]; }
source "$REPO_DIR/setup/envs/speculators.sh"
source "$REPO_DIR/setup/machines/horizon/envs/speculators.sh"
env_build && env_check && echo "ok $ENV_NAME (built)" || { echo "FAILED $ENV_NAME"; exit 1; }
