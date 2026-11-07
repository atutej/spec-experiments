#!/bin/bash
# Slurm wrapper for the training job (TWO gb nodes, 12 h): keeps a milestone copy of the newest checkpoint, builds the data view, then runs the itw experiment's two-node DDP script with THIS experiment's settings.
# Submit with submit_chain.sh from a login node. A job cut by its time limit is continued by the next job (training resumes from the run's checkpoints). It does NOT start once the epoch is finished (epoch0_end exists).
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP=/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark-grug-layers10-vocab32k_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
RUN_WORK_DIR=${GRUG_L10V32_WORK_DIR:-/scratch/09749/atutej/marin_speculator/runs/grug_dspark_itw_drdoom_grug-layers10-vocab32k}   # NOT named WORK: that is TACC's $WORK
if [[ -e "$RUN_WORK_DIR/checkpoints/epoch0_end" ]]; then echo "epoch0_end exists: training is complete, nothing to do"; exit 0; fi
# An interrupted save (the previous job was killed mid-save at its time limit) leaves checkpoint-<step>.tmp: move it aside (never delete) so neither the milestone copy nor the resume can pick it up.
for t in "$RUN_WORK_DIR"/checkpoints/checkpoint-*.tmp; do [[ -d "$t" ]] || continue; mkdir -p "$RUN_WORK_DIR/incomplete" && mv "$t" "$RUN_WORK_DIR/incomplete/$(basename "$t").job${SLURM_JOB_ID:-x}" && echo "moved the interrupted save aside: $(basename "$t")"; done
last=$(ls -d "$RUN_WORK_DIR"/checkpoints/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sed 's#.*/checkpoint-##' | sort -n | tail -1)
echo "latest step checkpoint: ${last:-none}; training continues towards epoch0_end"
if [[ -n "$last" && ! -e "$RUN_WORK_DIR/milestones/checkpoint-$last" ]]; then   # one copy per job (about every 11% of the epoch), kept: the rolling checkpoints are deleted
    mkdir -p "$RUN_WORK_DIR/milestones" && cp -rL "$RUN_WORK_DIR/checkpoints/checkpoint-$last" "$RUN_WORK_DIR/milestones/checkpoint-$last.tmp" && mv "$RUN_WORK_DIR/milestones/checkpoint-$last.tmp" "$RUN_WORK_DIR/milestones/checkpoint-$last" && echo "milestone copy: checkpoint-$last"
fi
# 2026-11-04 (job 1123392 hit a CUDA driver OOM in its first step on c672-004, the node of the eval job before it): log the GPU memory in use on each node at job start and kill stray GPU processes (the nodes are exclusive: anything on a GPU now is left over).
srun --nodes="${SLURM_NNODES:-2}" --ntasks-per-node=1 bash -c 'echo "$(hostname) GPU memory used at job start (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " ")"; pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr "\n" " "); if [[ -n "${pids// /}" ]]; then echo "$(hostname): killing stray GPU processes: $pids"; kill -9 $pids 2>/dev/null; sleep 20; echo "$(hostname) GPU memory used after the kill (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " ")"; fi' || true
bash "$EXP/vista/make_data_view.sh" || exit 1      # CPU, seconds
export EXP_DIR_OVERRIDE="$EXP"
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run_2node_ddp.sh train
