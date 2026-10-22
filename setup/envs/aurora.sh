# Recipe for the `aurora` env: Aurora's online drafter trainer (Ray controller, FSDP2 trainer, Mooncake hidden-state store).
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME and env_check (machine-independent).
# A separate env on purpose: Aurora pins transformers==4.57.1 (the speculators env has 5.x). SGLang is NOT installed: serving is the Marin vLLM fork
# (the `vllm` env or a copy of it); our Aurora fork (repos.txt) makes the SGLang engine import optional.

ENV_NAME=aurora

env_check() {
    conda_env_exists "$ENV_NAME" || { echo "env $ENV_NAME does not exist"; return 1; }
    # python -I: the clone directory $PROJECT_ROOT/aurora would otherwise shadow the package (namespace package) when the cwd is the workspace root
    conda run --no-capture-output -n "$ENV_NAME" python -I - <<'PY'
import shutil, sys
import torch, transformers, ray
print(f"torch {torch.__version__} | transformers {transformers.__version__} | ray {ray.__version__}")
problems = []
if transformers.__version__ != "4.57.1":
    problems.append("transformers is not the 4.57.1 Aurora pins")
if "+cu132" not in torch.__version__:
    problems.append("torch is not the cu132 build")
try:
    import mooncake  # noqa: F401
except Exception as e:  # noqa: BLE001
    problems.append(f"mooncake does not import: {e}")
if shutil.which("mooncake_master") is None:
    problems.append("mooncake_master not on PATH")
import importlib.util, os
spec = importlib.util.find_spec("aurora")   # (find_spec on a submodule would import aurora.inference, which loads mooncake.store)
factory_file = os.path.join(os.path.dirname(spec.origin), "inference", "factory.py") if spec and spec.origin else None
if factory_file is None or not os.path.exists(factory_file):
    problems.append("aurora is not installed")
elif "SGLang is only needed" not in open(factory_file).read():
    problems.append("aurora is not our fork build (aurora/inference/factory.py imports SGLang unconditionally): check out the pin in setup/repos.txt")
else:
    try:
        from aurora.inference.factory import create_inference_engines  # noqa: F401
    except ImportError as e:
        if "libcuda" in str(e):
            # mooncake.store links the GPU driver library, which CPU-only nodes (login, gg) do not have
            print("note: aurora's engine factory was not import-checked here (needs libcuda.so.1: run the check on a GPU node)")
        else:
            problems.append(f"aurora does not import: {e}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"aurora does not import: {e}")
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
PY
}
