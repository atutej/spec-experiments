# TACC Vista (aarch64 Grace; gg CPU, gh 1x H100, gb 4x GB200). Sourced by the root env.sh.
# Contract: see setup/machines/README.md.
MACHINE=vista
MACHINE_HOSTNAME_REGEX='\.vista\.tacc\.utexas\.edu$'
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
# The default `nvidia` module sets CC/CXX to nvc/nvc++, which rejects flags torch inductor
# (vLLM's torch.compile) passes (nvc-Error-Unknown switch: -Wno-psabi). Use GCC everywhere.
# CUDA comes from pip (the vllm env), not a module; recipes load gcc/14.2.0 for source builds.
export CC=gcc CXX=g++
# vLLM is built from source here for sm_90 + sm_100: env_check lists its kernel archs (setup/envs/vllm.sh).
export VLLM_CHECK_KERNEL_ARCHS=1
