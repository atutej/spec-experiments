#!/bin/bash
# Build the `vllm` env on Vista: compiles the Marin vLLM fork for sm_90 + sm_100 (hours).
# Needs no GPU; a whole gg node. The wheel is cached in $WORK/wheels, so re-runs are quick.
#   sbatch scripts/vista/build_vllm_env.sbatch
#SBATCH -J build-vllm
#SBATCH -p gg
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 08:00:00
#SBATCH -A CCR24067
#SBATCH -o %x-%j.out

set -uo pipefail
REPO_DIR=/scratch/09749/atutej/marin_speculator/spec-experiments   # sbatch copies the script, so no $0-relative lookup
source "$REPO_DIR/env.sh"
echo "node=$(hostname) kind=$NODE_KIND cpus=$SLURM_CPUS_ON_NODE $(free -g | awk '/Mem:/{print $2" GB"}')"
# The build downloads from conda, PyPI, PyTorch and GitHub. Fail fast, and say which host, if
# this node cannot reach one (a first run died at `conda create` on a DNS failure).
for h in repo.anaconda.com pypi.org download.pytorch.org github.com; do
    ok=0
    for i in 1 2 3 4 5 6; do getent hosts $h >/dev/null && { ok=1; break; }; sleep 10; done
    [[ $ok -eq 1 ]] || { echo "no DNS for $h on $(hostname); aborting" >&2; exit 1; }
done
# 144 cores, ~237 GB: memory bounds the compile; each nvcc job peaks at several GB. Tune from the first run.
export MAX_JOBS=${MAX_JOBS:-24} NVCC_THREADS=${NVCC_THREADS:-2}
bash "$REPO_DIR/setup/setup.sh" --rebuild vllm
