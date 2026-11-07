#!/bin/bash
# One gb node: the sleep/wake probe, three variants in parallel on GPUs 0, 1, 2 (see tools/sleep_wake_probe.py). Submit: sbatch vista/sleep_wake_probe.sbatch
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -p gb
#SBATCH -A ASC26088
#SBATCH --reservation=CGAI-gb
#SBATCH -t 01:00:00
#SBATCH -J sleep-wake-probe
#SBATCH -o /scratch/09749/atutej/marin_speculator/runs/rl_speculation/sleep_wake_probe/slurm-%j.out
set -uo pipefail
W=/scratch/09749/atutej/marin_speculator
OUT=$W/runs/rl_speculation/sleep_wake_probe
export HF_HOME=$W/cache/hf CUDA_HOME=$W/envs/skyrl_cuda_home CUDA_PATH=$W/envs/skyrl_cuda_home LD_LIBRARY_PATH=$W/envs/skyrl_extra_lib:${LD_LIBRARY_PATH:-}
export PATH=$W/envs/skyrl-megatron/bin:$W/envs/skyrl_cuda_home/bin:$PATH   # FlashInfer JIT needs ninja and nvcc on PATH
export CC=gcc CXX=g++ VLLM_CACHE_ROOT=$W/cache/vllm TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_$SLURM_JOB_ID TRITON_CACHE_DIR=/tmp/triton_$SLURM_JOB_ID
MODEL=$(ls -d $W/cache/hf/hub/models--open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21/snapshots/* | head -1)
DRAFT=$W/runs/grug_dspark_100k/checkpoints/0
PROMPTS=$W/runs/r2egym_prompts_128/first_turn.jsonl
PY=$W/envs/skyrl-megatron/bin/python
echo "node=$(hostname) model=$MODEL"; nvidia-smi -L | head -4
i=0
for v in none sleep1 sleep2; do
    TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}_$v TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}_$v CUDA_VISIBLE_DEVICES=$i $PY $W/spec-experiments/pipelines/rl_speculation/tools/sleep_wake_probe.py --variant $v --model "$MODEL" --draft $DRAFT --prompts $PROMPTS \
        --phys-gpu $i --out $OUT/$v.json > $OUT/$v.log 2>&1 &
    i=$((i+1))
done
wait
echo "done:"; ls -la $OUT
