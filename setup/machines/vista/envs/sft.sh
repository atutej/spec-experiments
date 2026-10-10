# sft env on vista: build steps. Sourced by setup/setup.sh after setup/envs/sft.sh (which has env_check).

env_build() {
    # aarch64, one env for gh (sm_90) and gb (sm_100): the same cu132 torch as the other envs. Pure pip, no compile,
    # so it builds on any node including idev. LLaMA-Factory is installed from source (the clone pinned in setup/repos.txt);
    # its own pins decide transformers, datasets, accelerate, peft and trl. deepspeed and liger-kernel are its optional
    # extras (requirements/*.txt); deepspeed builds its CUDA ops just in time, so training configs should use torch's fused AdamW.
    local cu=https://download.pytorch.org/whl/cu132
    conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
    conda activate "$ENV_NAME" || return 1
    pip install --extra-index-url $cu "torch==2.13.0+cu132" "torchvision==0.28.0+cu132" || return 1
    pip install --no-deps "torchaudio==2.11.0+cpu" --index-url https://download.pytorch.org/whl/cpu || return 1   # as in the speculators env
    pip install -e "$PROJECT_ROOT/llamafactory" || return 1
    pip install -r "$PROJECT_ROOT/llamafactory/requirements/deepspeed.txt" -r "$PROJECT_ROOT/llamafactory/requirements/liger-kernel.txt" || return 1
    pip install wandb || return 1   # report_to: wandb
}
