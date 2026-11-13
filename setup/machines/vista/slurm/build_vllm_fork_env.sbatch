#!/bin/bash
# Build the `vllm_fork` env on Vista from the Grug-DSpark FORK of vLLM (atutej/vllm, branch grug-dspark-layers-main = ONE commit, the GrugDSparkModel change, on top of marin-community/vllm main), from source: the same recipe as the pinned `vllm` env
# (compiles for sm_90 + sm_100, hours, a whole gg node, no GPU), pointed at a fresh clone of the fork. It writes its own env (`vllm_fork`) and its own wheel directory,
# so the pinned `vllm` env and its wheel in $WORK/wheels are untouched. The result is a real install: no PYTHONPATH overlay.
#   git clone --depth 1 --branch grug-dspark-layers-main git@github.com:atutej/vllm.git $PROJECT_ROOT/vllm_fork_src_main      (done once; VLLM_FORK_SRC overrides the directory)
#   sbatch setup/machines/vista/slurm/build_vllm_fork_env.sbatch                                                   (from a login node)
#SBATCH -J build-vllm-fork
#SBATCH -p gg
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH -A CCR24067
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out

set -uo pipefail
REPO_DIR=/scratch/09749/atutej/marin_speculator/spec-experiments
source "$REPO_DIR/env.sh"
export VLLM_ENV_NAME=vllm_fork VLLM_SRC=${VLLM_FORK_SRC:-$PROJECT_ROOT/vllm_fork_src_main}
export VLLM_SHA=$(git -C "$VLLM_SRC" rev-parse --short=10 HEAD) VLLM_WHEEL_DIR=${VLLM_WHEEL_DIR:-$WORK/wheels_fork}
echo "node=$(hostname) kind=$NODE_KIND cpus=$SLURM_CPUS_ON_NODE $(free -g | awk '/Mem:/{print $2" GB"}')"
echo "building env $VLLM_ENV_NAME from $VLLM_SRC at $VLLM_SHA into $VLLM_WHEEL_DIR"
[[ -z "$(git -C "$VLLM_SRC" status --porcelain)" ]] || { echo "$VLLM_SRC has local changes: the build must be a clean checkout of the pushed branch" >&2; exit 1; }
for h in repo.anaconda.com pypi.org download.pytorch.org github.com; do
    ok=0
    for i in 1 2 3 4 5 6; do getent hosts $h >/dev/null && { ok=1; break; }; sleep 10; done
    [[ $ok -eq 1 ]] || { echo "no DNS for $h on $(hostname); aborting" >&2; exit 1; }
done
export MAX_JOBS=${MAX_JOBS:-24} NVCC_THREADS=${NVCC_THREADS:-2}
bash "$REPO_DIR/setup/setup.sh" --rebuild vllm_fork
