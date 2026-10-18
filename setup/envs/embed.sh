# Recipe for the `embed` env: text embeddings (google/embeddinggemma-2) for selecting dataset rows by similarity.
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME and env_check (machine-independent).
# The build steps are per machine: setup/machines/<machine>/envs/embed.sh defines env_build.
# A separate env on purpose: embedding_gemma2 needs a newer transformers than the speculators/vllm envs pin.

ENV_NAME=embed

env_check() {
    conda_env_exists "$ENV_NAME" || { echo "env $ENV_NAME does not exist"; return 1; }
    conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import sys
from importlib.metadata import version
import torch, transformers, sentence_transformers, pyarrow, pandas
from transformers import AutoConfig
print(f"torch {torch.__version__} | transformers {transformers.__version__} | sentence-transformers "
      f"{sentence_transformers.__version__} | pyarrow {pyarrow.__version__}")
try:
    cfg = AutoConfig.from_pretrained("google/embeddinggemma-2")   # fails if transformers does not know embedding_gemma2
except Exception as e:
    print("PROBLEM: transformers cannot load google/embeddinggemma-2:", str(e)[:200]); sys.exit(1)
print("embeddinggemma-2 config loads:", type(cfg).__name__)
PY
}
