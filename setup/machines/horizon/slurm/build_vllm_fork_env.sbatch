#!/bin/bash
# Build the `vllm_fork` env on Horizon from the Grug-DSpark fork of vLLM (atutej/vllm, branch grug-dspark-layers-main), sm_100 only. Horizon version of vista/slurm/build_vllm_fork_env.sbatch.
#   git clone -b grug-dspark-layers-main https://github.com/atutej/vllm $PROJECT_ROOT/vllm_fork_src_main
#   sbatch -p debug setup/machines/horizon/slurm/build_vllm_fork_env.sbatch        (from a login node; or `bash` it inside an interactive job)
# Re-running resumes the compile: the build dir in the source tree is kept. Unlike Vista it builds just this env (setup.sh would also clone every repo in repos.txt).
#SBATCH -J build-vllm-fork
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out

set -uo pipefail
export MACHINE=horizon
REPO_DIR=${REPO_DIR:-/scratch/09749/atutej/marin_speculator/spec-experiments}
source "$REPO_DIR/env.sh"
export CUDA_MODULE_HOME=${CUDA_MODULE_HOME:-${CUDA_HOME:-}}   # the cuda module root (libcuda stub); the env activate hook later overrides CUDA_HOME
export VLLM_ENV_NAME=vllm_fork VLLM_SRC=${VLLM_FORK_SRC:-$PROJECT_ROOT/vllm_fork_src_main}
export VLLM_SHA=$(git -C "$VLLM_SRC" rev-parse --short=10 HEAD)
export VLLM_WHEEL_DIR=${VLLM_WHEEL_DIR:-$PROJECT_ROOT/wheels_fork}
echo "node=$(hostname) kind=$NODE_KIND gpus=$NUM_GPUS cpus=$(nproc --all) CUDA_HOME=${CUDA_HOME:-}"
echo "building env $VLLM_ENV_NAME from $VLLM_SRC at $VLLM_SHA into $VLLM_WHEEL_DIR"
[[ -z "$(git -C "$VLLM_SRC" status --porcelain)" ]] || { echo "$VLLM_SRC has local changes: the build must be a clean checkout of the pushed branch" >&2; exit 1; }
for h in repo.anaconda.com pypi.org download.pytorch.org github.com; do
    ok=0; for i in 1 2 3 4 5 6; do getent hosts $h >/dev/null && { ok=1; break; }; sleep 10; done
    [[ $ok -eq 1 ]] || { echo "no DNS for $h on $(hostname); aborting" >&2; exit 1; }
done
# hs_connectors (pip -e in env_build) lives in the speculators repo; clone only that one (https, no ssh key needed)
if [[ ! -d "$PROJECT_ROOT/speculators/.git" ]]; then
    git clone --branch grug-dspark-gate https://github.com/atutej/speculators.git "$PROJECT_ROOT/speculators" </dev/null || exit 1
fi
conda tos accept >/dev/null 2>&1 || true   # recent conda refuses the default channels until the terms are accepted
export MAX_JOBS=${MAX_JOBS:-48} NVCC_THREADS=${NVCC_THREADS:-2}
conda_env_exists() { conda env list | awk '{print $1}' | grep -qx "$1" || [[ -d "${CONDA_ENVS_PATH:-/nonexistent}/$1" ]]; }
source "$REPO_DIR/setup/envs/vllm.sh"
source "$REPO_DIR/setup/machines/horizon/envs/vllm.sh"
env_build && env_check && echo "ok $ENV_NAME (built)" || { echo "FAILED $ENV_NAME"; exit 1; }
