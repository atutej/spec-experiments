#!/usr/bin/env python
"""Patch an OpenThoughts-Agent RL sbatch for Vista after `hpc.launch --dry_run`: point CUDA_HOME at a small toolkit shim.

The module's CUDA_HOME is a huge tree on a shared filesystem; transformer_engine globs it recursively for every CUDA library at import
(minutes of stat() calls). FlashInfer's JIT links -lcudart, so the shim needs lib64/ and an unversioned libcudart.so (envs/skyrl_cuda_home:
the env's pip toolkit nvidia/cu13 with those symlinks added; see docs/rl_dspark_infra.md).

  python patch_vista_sbatch.py <sbatch> <cuda_home_shim_dir>

Also (so that an out-of-memory run explains itself): replaces PYTORCH_ALLOC_CONF=...max_split_size_mb:128 (it stops the allocator splitting cached blocks above 128 MB,
a fragmentation trap for multi-GB tensors) by garbage_collection_threshold only, exports SKYRL_VLLM_SLEEP_LEVEL only if set while generating (needs the sleep-level patch), and starts a per-GPU memory logger on every node of the job
(`gpu_mem_<node>.log` next to the Slurm log: per-process used memory and per-GPU used memory every 20 s).
"""
import os
import sys

path, shim = sys.argv[1], sys.argv[2]
s = open(path).read()
anchor = '    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"\nfi\n'
if anchor not in s:
    sys.exit("anchor not found: the launcher's CUDA block changed")
if "skyrl_cuda_home" in s:
    sys.exit("already patched")
block = f'# Vista CUDA_HOME shim (spec-experiments patch_vista_sbatch.py)\nexport CUDA_HOME={shim}\nexport CUDA_PATH="$CUDA_HOME"\n'
s = s.replace(anchor, anchor + block, 1)
old_alloc = "export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128"
if old_alloc in s:
    s = s.replace(old_alloc, "export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6   # max_split_size_mb:128 removed (spec-experiments patch_vista_sbatch.py)", 1)
# vLLM engines sleep during training in the colocated layout. Level 2 (MarinSkyRL's default) discards weights and the weight sync refills only the target model, so a
# speculative-decoding draft comes back uninitialised (accept length 1.0 in the 2026-10-14 colocated run). Level 1 keeps every weight in host memory.
# Needs the MarinSkyRL patch setup/machines/vista/patches/marinskyrl-sleep-level.patch (SKYRL_VLLM_SLEEP_LEVEL).
# Opt-in: only exported when SKYRL_VLLM_SLEEP_LEVEL is set while generating (the MarinSkyRL clone is clean at the moment; without the patch the variable has no effect).
sleep_level = os.environ.get("SKYRL_VLLM_SLEEP_LEVEL", "")
if sleep_level:
    s = s.replace(block, block + f"export SKYRL_VLLM_SLEEP_LEVEL={sleep_level}\n", 1)
import os
logdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "logs")
mem = f"""# GPU memory logger on every node (spec-experiments patch_vista_sbatch.py): who holds how much memory when a run goes out of memory
GPU_MEM_LOGDIR={logdir}
mkdir -p "$GPU_MEM_LOGDIR"
for _node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  srun --overlap --nodes=1 --ntasks=1 -w "$_node" bash -c 'while true; do date +%T; nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; sleep 20; done' >> "$GPU_MEM_LOGDIR/gpu_mem_$_node.log" 2>&1 &
done
"""
s = s.replace(block, block + mem, 1)
open(path, "w").write(s)
print("patched", path)
