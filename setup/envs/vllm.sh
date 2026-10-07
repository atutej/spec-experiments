# Recipe for the `vllm` env: serving (generation, hidden-state server).
# Sourced by setup/setup.sh with env.sh loaded; defines ENV_NAME and env_check (machine-independent).
# The build steps are per machine: setup/machines/<machine>/envs/vllm.sh defines env_build.
# Add a dependency: put the install step in env_build of every machine it applies to, make
# env_check (here) verify it, then run `setup/setup.sh --freeze`.

ENV_NAME=vllm

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
# vLLM's compiled kernels, for machines that build it from source (their env.sh sets
# VLLM_CHECK_KERNEL_ARCHS=1): list the SASS targets and require REQUIRED_CUDA_ARCHS.
if os.environ.get("VLLM_CHECK_KERNEL_ARCHS") == "1":
    import glob, re, subprocess
    tool = shutil.which("cuobjdump") or next(iter(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/*/bin/cuobjdump"))), None)
    sos = glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_C*.so"))
    if not tool or not sos:
        problems.append("cannot inspect vLLM kernels (cuobjdump or vllm/_C*.so missing)")
    else:
        out = subprocess.run([tool, "--list-elf", sos[0]], capture_output=True, text=True).stdout
        built = sorted(set("sm_" + m for m in re.findall(r"sm_(\d+)", out)))
        print("vllm kernel SASS:", built)
        problems += [f"vllm kernels lack {a}" for a in need if a not in built]
for p in problems:
    print("PROBLEM:", p)
sys.exit(1 if problems else 0)
PY
}
