# Recipe for the `speculators` env: data prep, training, analysis notebooks.
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME and env_check (machine-independent).
# The build steps are per machine: setup/machines/<machine>/envs/speculators.sh defines env_build.
# Add a dependency: put the install step in env_build of every machine it applies to, make
# env_check (here) verify it, then run `setup/setup.sh --freeze`.

ENV_NAME=speculators

env_check() {
    conda_env_exists "$ENV_NAME" || { echo "env $ENV_NAME does not exist"; return 1; }
    conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import os, sys, shutil
from importlib.metadata import version
import torch, transformers, datasets, pyarrow, speculators, hs_connectors, wandb   # wandb: --logger wandb needs it
from speculators.data_generation.configs import DATASET_CONFIGS
print(f"speculators {version('speculators')} | torch {torch.__version__} | transformers "
      f"{transformers.__version__} | datasets {datasets.__version__} | pyarrow {pyarrow.__version__}")
problems = []
if "nemotron-terminal" not in DATASET_CONFIGS:
    problems.append("speculators lacks the nemotron-terminal preset (wrong branch?)")
if shutil.which("mooncake_master") is None:
    problems.append("mooncake_master not on PATH (needed by the mooncake hidden-states backend)")
import torch
need = os.environ.get("REQUIRED_CUDA_ARCHS", "").split()
have = torch.cuda.get_arch_list()   # compiled-in archs; works without a GPU
missing = [a for a in need if a not in have and a.replace("sm_", "compute_") not in have]
if missing:
    problems.append(f"torch lacks CUDA archs {missing} (has {have})")
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
PY
}
