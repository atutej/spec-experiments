# speculators env on horizon: build steps, the same as vista/envs/speculators.sh (aarch64, cu132 torch wheels cover sm_100; pure pip, no compile). Sourced after setup/envs/speculators.sh (which has env_check).

env_build() {
    # aarch64, one env for gh (sm_90) and gb (sm_100): the cu132 torch wheels are built for
    # both (env_check verifies REQUIRED_CUDA_ARCHS), the same torch as the vllm env. No
    # compile step, so this runs on any node including idev (setup/machines/horizon/slurm/build_speculators_env.sbatch).
    # The rest resolves from speculators' own pins (transformers<5.17, datasets<=5.0.1, ...);
    # setup/machines/vista/locks/speculators.txt records what that gave.
    local cu=https://download.pytorch.org/whl/cu132
    conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
    conda activate "$ENV_NAME" || return 1
    pip install --extra-index-url $cu "torch==2.13.0+cu132" "torchvision==0.28.0+cu132" || return 1
    # torchaudio (pulled in by transformers' audio_utils) has no cu132 build, and PyPI's cu130 one
    # refuses to import next to cu132 torch. Use the CPU build, as Marin's gpu-constraints.txt does.
    pip install --no-deps "torchaudio==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu || return 1
    pip install -e "$PROJECT_ROOT/speculators/hs_connectors" || return 1
    pip install --extra-index-url $cu -e "$PROJECT_ROOT/speculators[mooncake-cuda13]" || return 1
    # The producer (vllm_fork env, Vista locks) is 0.3.13.post1; pip resolves 0.3.14 here, and the two do not talk (2026-11-17, job 97037: put_tensor status -800, "failed to read WRITE ack frame"). Keep both sides on one version.
    pip install "mooncake-transfer-engine-cuda13==0.3.13.post1" || return 1
    pip install jupyterlab ipywidgets matplotlib nbclient || return 1   # analysis notebooks
    pip install wandb || return 1   # metric logging: the trainer's --logger wandb imports it at the first log call
}
