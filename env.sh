#!/bin/bash
# Sourced before every step of every experiment, and by setup/setup.sh.
# Machine-specific settings live only here; scripts stay machine-independent.
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # workspace root: parent of this repo

# MACHINE selects the per-machine branches here and in setup/envs/*.sh.
if [[ "${TACC_SYSTEM:-}" == "vista" ]]; then
    export MACHINE=vista
    export CONDA_ROOT=$WORK/miniconda3
    # Envs live in the workspace ($SCRATCH, purgeable); setup/setup.sh rebuilds them.
    # With CONDA_ENVS_PATH, `conda activate <name>` and `conda create -n <name>` use it.
    export CONDA_ENVS_PATH=$PROJECT_ROOT/envs
    export PIP_CACHE_DIR=$PROJECT_ROOT/cache/pip
    # One env set serves gh (1x H100, sm_90) and gb (4 GPUs per node, Blackwell, sm_100);
    # env_check verifies torch was built for both.
    export REQUIRED_CUDA_ARCHS="sm_90 sm_100"
    # NODE_KIND: gg (Grace CPU only: code edits, commits, env builds), gh / gb (GPU nodes),
    # login. Decided by the Slurm partition, else by what nvidia-smi sees.
    case "${SLURM_JOB_PARTITION:-}" in
        gg*) NODE_KIND=gg ;; gh*) NODE_KIND=gh ;; gb*) NODE_KIND=gb ;; *) NODE_KIND="" ;;
    esac
    # TODO(vista): module loads (cuda, gcc, ...) once Phase 3 of docs/vista_setup.md settles them.
else
    export MACHINE=genai
    export CONDA_ROOT=/ssd1/an34232/miniconda3
    export PIP_CACHE_DIR=$CONDA_ROOT/pip_cache
    export REQUIRED_CUDA_ARCHS="sm_90"   # H100 NVL
    NODE_KIND=genai
fi
# NUM_GPUS: GPUs visible here (0 on CPU-only nodes, where nvidia-smi may be missing or fail).
NUM_GPUS=$( { command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null; } | grep -c '^GPU' )
if [[ -z "$NODE_KIND" ]]; then
    [[ "$NUM_GPUS" -gt 0 ]] && NODE_KIND=gpu || NODE_KIND=$([[ -n "${SLURM_JOB_ID:-}" ]] && echo cpu || echo login)
fi
export NODE_KIND NUM_GPUS
export CONDA_PKGS_DIRS=$CONDA_ROOT/pkgs

export HF_HOME=$PROJECT_ROOT/cache/hf
export HF_DATASETS_CACHE=$PROJECT_ROOT/cache/hf/datasets
export TRANSFORMERS_CACHE=$PROJECT_ROOT/cache/hf/transformers
export VLLM_CACHE_ROOT=$PROJECT_ROOT/cache/vllm
export TRITON_CACHE_DIR=$PROJECT_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$PROJECT_ROOT/cache/torch/inductor
export TORCH_HOME=$PROJECT_ROOT/cache/torch
export XDG_CACHE_HOME=$PROJECT_ROOT/cache/xdg
export TMPDIR=$PROJECT_ROOT/tmp

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
