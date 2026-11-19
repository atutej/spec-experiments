#!/bin/bash
# 5-minute probe of what the two-node training launcher assumes (read-only; no model, no training): GPUs, network interfaces, whether the launcher's srun flags are accepted, and that
# the envs' key binaries run on a compute node.   sbatch setup/machines/horizon/slurm/probe_two_nodes.sbatch   (from a login node)
#SBATCH -J probe-2node
#SBATCH -p debug
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
export MACHINE=horizon
source /scratch/09749/atutej/marin_speculator/spec-experiments/env.sh
echo "nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')  cpus_on_node=${SLURM_CPUS_ON_NODE:-?}"
# the same srun flags run_2node_ddp.sh uses
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpu-bind=none --mem=0 -c "${SLURM_CPUS_ON_NODE:-1}" --export=ALL bash -c '
echo "=== $(hostname) node_id=$SLURM_NODEID"
echo "-- gpus"; nvidia-smi -L; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tr "\n" ";"; echo
echo "-- ip"; ip -br addr | grep -v "^lo"
echo "-- infiniband"; ls /sys/class/infiniband 2>&1; for d in /sys/class/infiniband/*; do for p in $d/ports/*; do echo "$(basename $d) port $(basename $p): $(cat $p/state) $(cat $p/rate)"; done; done 2>&1
echo "-- net ifaces with ib in name"; ls /sys/class/net | tr "\n" " "; echo
echo "-- mem"; free -g | head -2; df -h /tmp | tail -1
echo "-- aarch64?"; uname -m
echo "-- mooncake_master"; PATH=/scratch/09749/atutej/marin_speculator/envs/vllm_fork/bin:$PATH; command -v mooncake_master
echo "-- hf reachable"; curl -sI --max-time 10 https://huggingface.co | head -1
echo "-- wandb reachable"; curl -sI --max-time 10 https://api.wandb.ai | head -1
echo "-- ulimit"; ulimit -l; ulimit -n
'
echo "-- gcc"; gcc --version | head -1
echo probe done
