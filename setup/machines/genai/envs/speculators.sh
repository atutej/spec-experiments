# speculators env on genai: build steps. Sourced by setup/setup.sh after setup/envs/speculators.sh (which has env_check).

env_build() {
    # Reconstructed from the working env; setup/machines/genai/locks/speculators.txt is authoritative.
    conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
    conda activate "$ENV_NAME" || return 1
    pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
    pip install -e "$PROJECT_ROOT/speculators[mooncake-cuda13]" || return 1
    pip install jupyterlab ipywidgets matplotlib nbclient || return 1   # analysis notebooks
    pip install wandb || return 1   # metric logging: the trainer's --logger wandb imports it at the first log call
}
