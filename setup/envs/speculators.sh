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
        # One env for gh (sm_90) and gb (sm_100): torch/vLLM wheels or builds must cover both
        # (e.g. TORCH_CUDA_ARCH_LIST="9.0;10.0" for source builds). Build on any Vista node,
        # gg included; only setup/setup.sh --gpu needs a GPU node.
        # TODO(vista): aarch64 recipe -- see docs/vista_setup.md, Phase 3. Likely the same pip
        # steps as genai once CUDA torch for aarch64 and an aarch64 Mooncake (or none) are settled.
        echo "speculators env recipe for vista is not written yet (docs/vista_setup.md, Phase 3)" >&2
        return 1
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
