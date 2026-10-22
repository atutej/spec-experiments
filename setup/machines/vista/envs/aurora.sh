# aurora env on vista: build steps. Sourced by setup/setup.sh after setup/envs/aurora.sh (which has env_check).

env_build() {
    # aarch64, same cu132 torch as the speculators / vllm / embed envs. Pins from the 2026-10-21 resolve-only dry run
    # (uv pip compile, aarch64-manylinux_2_28, cp312, --no-build: every package resolved to a wheel).
    local cu=https://download.pytorch.org/whl/cu132 src="$PROJECT_ROOT/aurora"
    [[ -d "$src/.git" ]] || { echo "aurora checkout missing at $src (run setup/setup.sh to clone it)" >&2; return 1; }
    conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
    conda activate "$ENV_NAME" || return 1
    pip install --extra-index-url $cu "torch==2.13.0+cu132" || return 1
    # Aurora's declared dependencies minus sglang-router (+ fastapi/uvicorn for its callback server: Aurora gets them transitively from `sglang[all]`, which we do not install); the cuda13 Mooncake wheel (the plain mooncake-transfer-engine wheel resolves to the same version but is not the cu13 build we use elsewhere)
    pip install --extra-index-url $cu "transformers==4.57.1" "numpy<2.4" "mooncake-transfer-engine-cuda13==0.3.13.post1" \
        datasets pytest fastapi uvicorn setuptools tqdm wandb qwen-vl-utils psutil accelerate pydantic openai-harmony cmake ninja packaging pyzmq openai omegaconf ray numba || return 1
    pip install --no-deps -e "$src" || return 1
}
