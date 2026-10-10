# Recipe for the `sft` env: supervised fine-tuning with LLaMA-Factory (pipelines/sft). Separate from `speculators` because the
# pins conflict (LLaMA-Factory: datasets<=4.0.0, transformers<=5.8, accelerate<=1.11, peft, trl; speculators: datasets<=5.0.1).
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME and env_check (machine-independent).
# The build steps are per machine: setup/machines/<machine>/envs/sft.sh defines env_build.

ENV_NAME=sft

env_check() {
    conda_env_exists "$ENV_NAME" || { echo "env $ENV_NAME does not exist"; return 1; }
    conda run --no-capture-output -n "$ENV_NAME" python - <<'PY'
import inspect, os, shutil, sys
from importlib.metadata import version
import torch, transformers, datasets, accelerate, deepspeed, liger_kernel, wandb, llamafactory
from llamafactory.data.template import ReasoningTemplate
print(f"llamafactory {version('llamafactory')} | torch {torch.__version__} | transformers {transformers.__version__} | "
      f"datasets {datasets.__version__} | accelerate {accelerate.__version__} | deepspeed {deepspeed.__version__}")
problems = []
# the reason for this pin: with `mask_history`, earlier turns' thinking is dropped (not kept, no empty think tags added)
if "discarding_history_cot" not in inspect.signature(ReasoningTemplate.encode_multiturn).parameters:
    problems.append("llamafactory lacks discarding_history_cot (wrong commit? OpenThoughts-Agent's fork has no such option)")
if shutil.which("llamafactory-cli") is None:
    problems.append("llamafactory-cli not on PATH")
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
