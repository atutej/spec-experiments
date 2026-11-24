#!/bin/bash
#SBATCH -J slurm-backend-tests
#SBATCH -p gpu-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 00:20:00
# SkyRL CPU test lane (docs/dev-guide/skyrl.md) for the Slurm backend and the launcher files it touches.
# Submit from a login node (login nodes cannot run the tests):  sbatch -o $PROJECT_ROOT/logs/slurm/%x-%j.log run_backend_tests.sbatch, PROJECT_ROOT exported.
# MARIN_ENV-style uv cache lives in $PROJECT_ROOT/marin_env/uv_cache.
P=${PROJECT_ROOT:?export PROJECT_ROOT (the workspace root) before sbatch}
cd "$P/marin/vendor/skyrl" || exit 1
echo "node $(hostname) $(uname -m); branch $(git -C "$P/marin" branch --show-current); $(git -C "$P/marin" status --short | wc -l) changed files"
env -i HOME=$HOME USER=$USER LANG=C.UTF-8 TMPDIR=/tmp UV_CACHE_DIR=$P/marin_env/uv_cache PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin \
  uv run --python /usr/bin/python3.12 --no-python-downloads --isolated --frozen --extra cpu --no-group dev \
    --with pytest==8.4.2 --with pytest-asyncio==1.4.0 \
    pytest -q -p no:cacheprovider cloud/iris/tests/test_slurm_backend.py cloud/iris/tests/test_launch.py cloud/iris/tests/test_launch_config.py cloud/iris/tests/test_iris_backend.py cloud/iris/tests/test_frozen_cuda_runtime_env.py
echo "pytest exit code: $?"
