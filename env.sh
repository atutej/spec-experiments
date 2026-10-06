#!/bin/bash
# Sourced before every step of the DSpark experiment.
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # workspace root: parent of this repo
export CONDA_ROOT=/ssd1/an34232/miniconda3

export HF_HOME=$PROJECT_ROOT/cache/hf
export HF_DATASETS_CACHE=$PROJECT_ROOT/cache/hf/datasets
export TRANSFORMERS_CACHE=$PROJECT_ROOT/cache/hf/transformers
export VLLM_CACHE_ROOT=$PROJECT_ROOT/cache/vllm
export TRITON_CACHE_DIR=$PROJECT_ROOT/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$PROJECT_ROOT/cache/torch/inductor
export TORCH_HOME=$PROJECT_ROOT/cache/torch
export XDG_CACHE_HOME=$PROJECT_ROOT/cache/xdg
export TMPDIR=$PROJECT_ROOT/tmp

export PIP_CACHE_DIR=$CONDA_ROOT/pip_cache
export CONDA_PKGS_DIRS=$CONDA_ROOT/pkgs

mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
