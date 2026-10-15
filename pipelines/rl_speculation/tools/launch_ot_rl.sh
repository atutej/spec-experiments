#!/bin/bash
# Generate an OpenThoughts-Agent RL job for Vista, patch the sbatch, and (optionally) run it in the CURRENT allocation (idev) instead of sbatch.
#   launch_ot_rl.sh <rl_config.yaml> <job_name> [--run]      (extra launcher flags: LAUNCH_ARGS="--num_nodes 2 --policy_num_nodes 1 --time_limit 03:00:00")
# Needs the workspace clones (MarinSkyRL, OpenThoughts-Agent with setup/machines/vista/patches applied) and envs/skyrl-megatron. See docs/rl_dspark_infra.md.
set -euo pipefail
YAML=${1:?usage: launch_ot_rl.sh <rl_config.yaml> <job_name> [--run]}; JOB=${2:?job name}; RUN=${3:-}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
OT="$PROJECT_ROOT/OpenThoughts-Agent"
export RL_ENV_DIR="$PROJECT_ROOT/envs/skyrl-megatron" DCFT_RL_ENV="$PROJECT_ROOT/envs/skyrl-megatron" RL_REPO_DIR="$PROJECT_ROOT/MarinSkyRL"
# OpenThoughts-Agent tells the GH200 (gh, 1 GPU/node) and GB200 (gb, 4 GPUs/node) Vista profiles apart only by $SLURM_JOB_PARTITION, which is unset on a login node:
# without this the sbatch is generated with the gh profile (gpus_per_node 1, cpus 72, cuda/12.8 module). Always generate for gb.
export SLURM_JOB_PARTITION=gb
export OT_AGENT_NO_GPU_GRES=1 DCFT="$OT" LD_LIBRARY_PATH="$PROJECT_ROOT/envs/skyrl_extra_lib:${LD_LIBRARY_PATH:-}"   # libstdc++ for the flash-attn wheel
MODEL=${MODEL:-$(ls -d "$PROJECT_ROOT"/cache/hf/hub/models--open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21/snapshots/* | head -1)}
cd "$OT"; set +u; source hpc/dotenv/tacc.env >/dev/null 2>&1; set -u
E="experiments/${JOB}__dryrun"; rm -rf "$E/sbatch" "$E/configs" "$E/ray_logs" "$E/$JOB/ray_logs"
/work/09749/atutej/vista/miniconda3/envs/vllm_sandboxes/bin/python -m hpc.launch --job_type rl --rl_config "$YAML" --model_path "$MODEL" --job_name "$JOB" ${LAUNCH_ARGS:-} --dry_run > "$PROJECT_ROOT/logs/smoke/${JOB}_dryrun.log" 2>&1
python3 "$REPO_DIR/pipelines/rl_speculation/tools/patch_vista_sbatch.py" "$E/sbatch/${JOB}_rl.sbatch" "$PROJECT_ROOT/envs/skyrl_cuda_home"
mkdir -p "$E/logs"
if [[ "$RUN" == "--run" ]]; then
    nohup bash -l "$E/sbatch/${JOB}_rl.sbatch" > "$E/logs/run_direct.log" 2>&1 &
    echo "running: $OT/$E/logs/run_direct.log"
else
    echo "sbatch ready: $OT/$E/sbatch/${JOB}_rl.sbatch"
fi
