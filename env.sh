#!/bin/bash
# Sourced before every step of every experiment, and by setup/setup.sh.
# Machine-specific settings live in setup/machines/<machine>/env.sh; scripts stay machine-independent.
export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # workspace root: parent of this repo
_SPEC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pick the machine: an explicit MACHINE=<name> wins, else the first setup/machines/*/env.sh whose
# MACHINE_HOSTNAME_REGEX matches `hostname -f`, else the one marked MACHINE_FALLBACK=1 (genai).
_spec_host=$(hostname -f 2>/dev/null || hostname)
_spec_machine=${MACHINE:-}
if [[ -z "$_spec_machine" ]]; then
    for _spec_f in "$_SPEC_REPO"/setup/machines/*/env.sh; do
        _spec_m=$(basename "$(dirname "$_spec_f")")
        _spec_re=$(unset MACHINE_HOSTNAME_REGEX; source "$_spec_f" >/dev/null 2>&1; echo "${MACHINE_HOSTNAME_REGEX:-}")
        if [[ -n "$_spec_re" && "$_spec_host" =~ $_spec_re ]]; then _spec_machine=$_spec_m; break; fi
    done
fi
if [[ -z "$_spec_machine" ]]; then
    for _spec_f in "$_SPEC_REPO"/setup/machines/*/env.sh; do
        if (source "$_spec_f" >/dev/null 2>&1; [[ "${MACHINE_FALLBACK:-}" == 1 ]]); then _spec_machine=$(basename "$(dirname "$_spec_f")"); break; fi
    done
fi
if [[ ! -f "$_SPEC_REPO/setup/machines/$_spec_machine/env.sh" ]]; then
    _spec_known=$(ls "$_SPEC_REPO/setup/machines" | tr '\n' ' ')
    if [[ -n "${MACHINE:-}" ]]; then echo "env.sh: unknown MACHINE='$MACHINE' (known: $_spec_known)" >&2
    else echo "env.sh: no machine matches host '$_spec_host' (set MACHINE=<name>; known: $_spec_known)" >&2; fi
    unset _spec_host _spec_machine _spec_f _spec_m _spec_re _spec_known _SPEC_REPO; return 1 2>/dev/null || exit 1
fi
source "$_SPEC_REPO/setup/machines/$_spec_machine/env.sh"
export MACHINE
for _spec_v in MACHINE CONDA_ROOT REQUIRED_CUDA_ARCHS; do
    [[ -n "${!_spec_v:-}" ]] || { echo "env.sh: setup/machines/$_spec_machine/env.sh must set $_spec_v" >&2; return 1 2>/dev/null || exit 1; }
done
unset _spec_host _spec_machine _spec_f _spec_m _spec_re _spec_v _SPEC_REPO

# NUM_GPUS: GPUs visible here (0 on CPU-only nodes, where nvidia-smi may be missing or fail).
# `|| true`: grep -c exits 1 when it counts 0 (a CPU node), which would kill a caller running under `set -e`.
NUM_GPUS=$( { command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null; } | grep -c '^GPU' || true )
if [[ -z "${NODE_KIND:-}" ]]; then
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
export TMPDIR=${SPEC_TMPDIR:-$PROJECT_ROOT/tmp}   # a machine may set SPEC_TMPDIR (node-local, see its env.sh)
export FLASHINFER_WORKSPACE_BASE=$PROJECT_ROOT/cache/flashinfer   # JIT kernels; default is ~/.cache (small $HOME quota)

mkdir -p "$PROJECT_ROOT/logs/setup" "$PROJECT_ROOT/logs/slurm" "$PROJECT_ROOT/logs/smoke" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$PIP_CACHE_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
