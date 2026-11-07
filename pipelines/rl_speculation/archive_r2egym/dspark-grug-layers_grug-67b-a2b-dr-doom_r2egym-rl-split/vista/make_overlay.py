#!/usr/bin/env python
"""Build a vLLM overlay for the RL env: a symlink forest of the env's installed `vllm` package with the grug-dspark-layers files copied over.
  python make_overlay.py <installed vllm package dir> <vllm checkout> <overlay dir>      (put <overlay dir> first on PYTHONPATH)
Refuses to write into an existing overlay dir and checks that each overridden file is identical to the pinned commit in the installed wheel."""
import os, shutil, subprocess, sys
site, repo, out = sys.argv[1:4]
PIN = "39e6286969"
FILES = ["vllm/config/speculative.py", "vllm/model_executor/models/qwen3_dspark.py", "vllm/model_executor/models/registry.py",
         "vllm/transformers_utils/configs/speculators/algos.py", "vllm/model_executor/models/grug_dspark.py"]
for p in (site, repo):
    if not os.path.isdir(p):
        sys.exit(f"no such dir: {p}")
if os.path.exists(out):
    sys.exit(f"{out} exists; pick a fresh dir")
for f in FILES:
    inst = os.path.join(site, f[len("vllm/"):])
    if os.path.exists(inst):
        pin = subprocess.run(["git", "-C", repo, "show", f"{PIN}:{f}"], capture_output=True, check=True).stdout
        if pin != open(inst, "rb").read():
            sys.exit(f"installed {f} differs from pin {PIN}: overlay would drop other changes")
def link(src, dst):
    """Real directories along the path of an overridden file, symlinks for everything else."""
    rel = os.path.relpath(src, site)
    prefix = "vllm/" if rel == "." else "vllm/" + rel + "/"
    os.makedirs(dst)
    for e in os.listdir(src):
        s, d = os.path.join(src, e), os.path.join(dst, e)
        if e == "__pycache__":
            continue
        if os.path.isdir(s) and any(f.startswith(prefix + e + "/") for f in FILES):
            link(s, d)
        else:
            os.symlink(s, d)
link(site, os.path.join(out, "vllm"))
for f in FILES:
    dst = os.path.join(out, f)
    if os.path.islink(dst):
        os.remove(dst)
    shutil.copy(os.path.join(repo, f), dst)
print("overlay at", out)
