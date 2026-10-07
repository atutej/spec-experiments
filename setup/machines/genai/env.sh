# UT ECE genai machines (x86_64, several H100 NVL, no scheduler). Sourced by the root env.sh.
# Contract: see setup/machines/README.md. No hostname regex: genai is the fallback machine.
MACHINE=genai
MACHINE_FALLBACK=1
export CONDA_ROOT=/ssd1/an34232/miniconda3
export PIP_CACHE_DIR=$CONDA_ROOT/pip_cache
export REQUIRED_CUDA_ARCHS="sm_90"   # H100 NVL
NODE_KIND=genai
