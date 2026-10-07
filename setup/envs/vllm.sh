# Recipe for the `vllm` env: serving (generation, hidden-state server).
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME, env_build, env_check.
# Add a dependency: put the install step in env_build (every machine it applies to),
# make env_check verify it, then run `setup/setup.sh --freeze`.

ENV_NAME=vllm

env_build() {
    case "$MACHINE" in
    genai)
        # Reconstructed from the working env; setup/locks/genai/vllm.txt is authoritative.
        # The Marin vLLM fork (commit 39e62869693c) is a local wheel, not on PyPI.
        : "${VLLM_WHEEL:=$PROJECT_ROOT/tmp/vllm-0.0.0.dev20260929+marin.39e62869693c.cu132-cp38-abi3-manylinux_2_28_x86_64.whl}"
        [[ -f "$VLLM_WHEEL" ]] || { echo "Marin vLLM wheel not found: $VLLM_WHEEL (set VLLM_WHEEL)" >&2; return 1; }
        conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
        conda activate "$ENV_NAME" || return 1
        pip install "$VLLM_WHEEL" --extra-index-url https://download.pytorch.org/whl/cu132 || return 1
        pip uninstall -y torchaudio                      # no cu132 build; vLLM does not need it
        pip install "mooncake-transfer-engine-cuda13==0.3.13.post1" || return 1
        pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
        ;;
    vista)
        # One env for gh (sm_90) and gb (sm_100): torch/vLLM wheels or builds must cover both
        # (e.g. TORCH_CUDA_ARCH_LIST="9.0;10.0" for source builds). Build on any Vista node,
        # gg included; only setup/setup.sh --gpu needs a GPU node.
        # TODO(vista): aarch64 recipe -- see docs/vista_setup.md, Phase 3. Needs an aarch64
        # build of the Marin vLLM fork at 39e62869693c, CUDA torch for aarch64 (sbsa),
        # Mooncake (or the `file` hidden-states backend), and editable hs_connectors.
        echo "vllm env recipe for vista is not written yet (docs/vista_setup.md, Phase 3)" >&2
        return 1
        ;;
    *) echo "no vllm recipe for MACHINE=$MACHINE" >&2; return 1 ;;
    esac
}

env_check() {
    conda_env_exists "$ENV_NAME" || { echo "env $ENV_NAME does not exist"; return 1; }
    conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import os, sys, shutil
import torch, vllm, hs_connectors
print(f"vllm {vllm.__version__} | torch {torch.__version__} (cuda {torch.version.cuda})")
problems = []
if "+marin" not in vllm.__version__:
    problems.append("vllm is not the Marin fork build")
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
