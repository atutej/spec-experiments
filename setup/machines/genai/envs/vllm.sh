# vllm env on genai: build steps. Sourced by setup/setup.sh after setup/envs/vllm.sh (which has env_check).

env_build() {
    # Reconstructed from the working env; setup/machines/genai/locks/vllm.txt is authoritative.
    # The Marin vLLM fork (commit 39e62869693c) is a local wheel, not on PyPI.
    : "${VLLM_WHEEL:=$PROJECT_ROOT/tmp/vllm-0.0.0.dev20260929+marin.39e62869693c.cu132-cp38-abi3-manylinux_2_28_x86_64.whl}"
    [[ -f "$VLLM_WHEEL" ]] || { echo "Marin vLLM wheel not found: $VLLM_WHEEL (set VLLM_WHEEL)" >&2; return 1; }
    conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
    conda activate "$ENV_NAME" || return 1
    pip install "$VLLM_WHEEL" --extra-index-url https://download.pytorch.org/whl/cu132 || return 1
    pip uninstall -y torchaudio                      # no cu132 build; vLLM does not need it
    pip install "mooncake-transfer-engine-cuda13==0.3.13.post1" || return 1
    pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
}
