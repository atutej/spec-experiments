# embed env on vista: build steps. Sourced by setup/setup.sh after setup/envs/embed.sh (which has env_check).

env_build() {
    # aarch64, same cu132 torch as the speculators and vllm envs (one env serves gh and gb).
    # transformers and sentence-transformers are left unpinned: google/embeddinggemma-2 (model type
    # embedding_gemma2) needs a release newer than the 5.16.1 the other envs carry; env_check verifies it loads.
    local cu=https://download.pytorch.org/whl/cu132
    conda_env_exists "$ENV_NAME" || conda create -y -n "$ENV_NAME" python=3.12 || return 1
    conda activate "$ENV_NAME" || return 1
    pip install --extra-index-url $cu "torch==2.13.0+cu132" "torchvision==0.28.0+cu132" || return 1   # torchvision: the multimodal EmbeddingGemma2 processors import it
    pip install -U transformers sentence-transformers huggingface_hub pyarrow pandas numpy pillow || return 1   # pillow: sentence-transformers builds the multimodal EmbeddingGemma2Processor even for text
}
