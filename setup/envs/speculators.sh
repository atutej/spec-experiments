# Recipe for the `speculators` env: data prep, training, analysis notebooks.
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME, env_build, env_check.
# Add a dependency: put the install step in env_build (every machine it applies to),
# make env_check verify it, then run `setup/setup.sh --freeze`.

ENV_NAME=speculators

env_build() {
    case "$MACHINE" in
    genai)
        # Reconstructed from the working env; setup/locks/genai/speculators.txt is authoritative.
        conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
        conda activate "$ENV_NAME" || return 1
        pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
        pip install -e "$PROJECT_ROOT/speculators[mooncake-cuda13]" || return 1
        pip install jupyterlab ipywidgets matplotlib nbclient || return 1   # analysis notebooks
        ;;
    vista)
        # aarch64, one env for gh (sm_90) and gb (sm_100): the cu132 torch wheels are built for
        # both (env_check verifies REQUIRED_CUDA_ARCHS), the same torch as the vllm env. No
        # compile step, so this runs on any node including idev (scripts/vista/build_speculators_env.sh).
        # The rest resolves from speculators' own pins (transformers<5.17, datasets<=5.0.1, ...);
        # setup/locks/vista/speculators.txt records what that gave.
        local cu=https://download.pytorch.org/whl/cu132
        conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
        conda activate "$ENV_NAME" || return 1
        pip install --extra-index-url $cu "torch==2.13.0+cu132" "torchvision==0.28.0+cu132" || return 1
        # torchaudio (pulled in by transformers' audio_utils) has no cu132 build, and PyPI's cu130 one
        # refuses to import next to cu132 torch. Use the CPU build, as Marin's gpu-constraints.txt does.
        pip install --no-deps "torchaudio==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu || return 1
        pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
        pip install --extra-index-url $cu -e "$PROJECT_ROOT/speculators[mooncake-cuda13]" || return 1
        pip install jupyterlab ipywidgets matplotlib nbclient || return 1   # analysis notebooks
        ;;
    *) echo "no speculators recipe for MACHINE=$MACHINE" >&2; return 1 ;;
    esac
}

env_check() {
    conda_env_exists "$ENV_NAME" || { echo "env $ENV_NAME does not exist"; return 1; }
    conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import os, sys, shutil
from importlib.metadata import version
import torch, transformers, datasets, pyarrow, speculators, hs_connectors
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
