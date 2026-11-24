#!/bin/bash
# Submit from a login node:  sbatch -o $PROJECT_ROOT/logs/slurm/%x-%j.log setup/machines/horizon/slurm/build_skyrl_env.sbatch   (PROJECT_ROOT exported)
#SBATCH -J skyrl-env-build
#SBATCH -p gpu-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 01:00:00
# Settings (environment variables, all optional):
#   MARIN_REPO    marin checkout to build from (default $PROJECT_ROOT/marin). The environment's editable installs record this path, so use a checkout that stays where it is.
#   RUNTIME_NAME  name of the runtime under $PROJECT_ROOT/marin_env (default skyrl-megatron); a new name builds beside an existing runtime instead of replacing it.
# Produces $PROJECT_ROOT/marin_env/<RUNTIME_NAME> (the environment), <RUNTIME_NAME>.runtime.env and <RUNTIME_NAME>-site-packages.tar (what slurm.runtime_archive names).
# Build Marin's frozen SkyRL runtime (megatron profile: --extra megatron --extra vllm --extra telemetry) on aarch64/GB200
# with the repo's own vendor/skyrl/cloud/iris/bootstrap_runtime.sh. Env and uv cache live on /scratch (home quota is small).
P=${PROJECT_ROOT:?export PROJECT_ROOT (the workspace root) before sbatch}
REPO=${MARIN_REPO:-$P/marin}; ROOT=$REPO/vendor/skyrl
NAME=${RUNTIME_NAME:-skyrl-megatron}
ENVDIR=$P/marin_env/$NAME; RUNTIME_FILE=$P/marin_env/$NAME.runtime.env; ARCHIVE=$P/marin_env/$NAME-site-packages.tar
export UV_CACHE_DIR=$P/marin_env/uv_cache
step(){ echo; echo "=== [$(date +%T)] $*"; }
step "0. node / tools (before cleaning env)"; hostname; uname -m; ls -l /usr/bin/python3.12; /usr/bin/python3.12 --version; git --version
(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1 || echo "nvidia-smi: not on PATH") | head -4
df -h /tmp $P | tail -2; echo "repo HEAD: $(git -C $REPO log -1 --format='%h %s' | cut -c1-80)"; cat $ROOT/.uv-version
step "1. bootstrap_runtime.sh (clean environment, like the Iris container: no CUDA modules on PATH)"
mkdir -p $UV_CACHE_DIR
t0=$(date +%s)
env -i HOME=$HOME USER=$USER LANG=C.UTF-8 TMPDIR=/tmp UV_CACHE_DIR=$UV_CACHE_DIR \
  PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin \
  bash $ROOT/cloud/iris/bootstrap_runtime.sh $ROOT $ENVDIR $RUNTIME_FILE megatron production
rc=$?; echo "bootstrap exit code: $rc ; elapsed $(( $(date +%s)-t0 )) s"
[ $rc -eq 0 ] || { echo BOOTSTRAP_FAILED; exit $rc; }
step "2. what got installed"
du -sh $ENVDIR $UV_CACHE_DIR 2>/dev/null; echo "runtime file:"; sed 's/^/  /' $RUNTIME_FILE | cut -c1-180 | head -20
step "3. torch / CUDA / GPUs on this node (sourcing the runtime file)"
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c "source $RUNTIME_FILE; $ENVDIR/bin/python - <<'PY'
import torch, platform
print('python', platform.python_version(), platform.machine())
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()): print(' gpu', i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    import torch.distributed as dist
    print('nccl', torch.cuda.nccl.version())
import vllm; print('vllm', vllm.__version__)
import transformer_engine; print('transformer_engine', getattr(transformer_engine,'__version__','?'))
import ray; print('ray', ray.__version__)
PY" 2>&1 | tail -20
step "4. archive for slurm.runtime_archive (dereferenced lib/, including the linker names bootstrap_runtime.sh adds beside site-packages)"
t1=$(date +%s); tar -chf $ARCHIVE.partial -C $ENVDIR lib && mv $ARCHIVE.partial $ARCHIVE
echo "archive: $(du -h $ARCHIVE | cut -f1), $(tar -tf $ARCHIVE | wc -l) entries, $(tar -tf $ARCHIVE | grep -c 'libcudart.so$') libcudart.so link(s), $(( $(date +%s)-t1 )) s"
echo; echo "=== DONE $(date +%T)"
