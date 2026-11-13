# TACC Horizon (early access; Grace CPUs; GB nodes with 4x GB200 in the gpu queues, VV nodes without GPUs in the cpu queues). Sourced by the root env.sh.
# DRAFT written from docs.tacc.utexas.edu/hpc/horizon on 2026-11-13 without access to the machine: NOTES.md lists what is unverified.
# Contract: see setup/machines/README.md. `MACHINE=horizon` selects this profile even if the hostname regex below does not match.
MACHINE=horizon
MACHINE_HOSTNAME_REGEX='\.horizon\.tacc\.utexas\.edu$'
export CONDA_ROOT=${CONDA_ROOT:-$SCRATCH/miniconda3}   # $WORK is not available to early users; set CONDA_ROOT to an existing conda to reuse it
# Envs live in the workspace ($SCRATCH, purged after 10 days without access); setup/setup.sh rebuilds them.
export CONDA_ENVS_PATH=$PROJECT_ROOT/envs
export PIP_CACHE_DIR=$PROJECT_ROOT/cache/pip
export REQUIRED_CUDA_ARCHS="sm_100"   # GB200 only
case "${SLURM_JOB_PARTITION:-}" in
    gpu*) NODE_KIND=gb ;; cpu*) NODE_KIND=cpu ;; *) NODE_KIND="" ;;
esac
export CC=gcc CXX=g++   # as on Vista: an nvc default breaks torch inductor and FlashInfer builds
export SPEC_TMPDIR=/tmp
export VLLM_CHECK_KERNEL_ARCHS=1
export VLLM_WHEEL_DIR=${VLLM_WHEEL_DIR:-$PROJECT_ROOT/wheels}   # Vista keeps the built wheel on $WORK; there is none here yet
