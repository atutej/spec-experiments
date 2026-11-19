#!/bin/bash
# Slurm wrapper of the training job on Horizon (TWO GB nodes). Submit with submit.sh from a LOGIN node (sbatch is refused on compute nodes).
# One epoch (about 409k steps) is far longer than the 24 h-effective stop point: the job is meant to be STOPPED BY HAND (scancel) where tools/effective_time.py says so; the rolling checkpoints
# (every 0.1% of the epoch) lose at most about 8 min, and copies of every MILESTONE_STEPS-th checkpoint are kept in $WORK_DIR/milestones. A job that hits its time limit is continued by the next
# (training resumes from $CKPT_DIR). It does NOT start once epoch0_end exists.
#SBATCH -N 2
#SBATCH --exclusive
#SBATCH -p debug
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
[[ -f "$EXP/settings.sh" ]] || EXP=${EXP_DIR:?batch script copied by sbatch: submit with submit.sh (sets EXP_DIR)}
REPO_DIR=$(cd "$EXP/../../../.." && pwd)
set +u; source "$REPO_DIR/env.sh"; set -u
EXP_DIR=$EXP; source "$EXP/settings.sh"
if [[ -e "$CKPT_DIR/epoch0_end" ]]; then echo "epoch0_end exists: training is complete, nothing to do"; exit 0; fi
# An interrupted save (killed mid-save) leaves checkpoint-<step>.tmp: move it aside (never delete) so neither the milestone copy nor the resume can pick it up.
for t in "$CKPT_DIR"/checkpoint-*.tmp; do [[ -d "$t" ]] || continue; mkdir -p "$WORK_DIR/incomplete" && mv "$t" "$WORK_DIR/incomplete/$(basename "$t").job${SLURM_JOB_ID:-x}" && echo "moved the interrupted save aside: $(basename "$t")"; done
# GPU memory in use on each node at job start; kill stray GPU processes (the nodes are exclusive: anything on a GPU now is left over).
srun --nodes="${SLURM_NNODES:-2}" --ntasks-per-node=1 bash -c 'echo "$(hostname) GPU memory used at job start (MiB): $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr "\n" " ")"; pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr "\n" " "); if [[ -n "${pids// /}" ]]; then echo "$(hostname): killing stray GPU processes: $pids"; kill -9 $pids 2>/dev/null; sleep 20; fi' || true
[[ -f "$DATA_DIR/state.json" ]] || { echo "no dataset at $DATA_DIR: run horizon/build_data.sbatch first" >&2; exit 1; }
# Milestone copier: while training runs, copy the newest COMPLETE checkpoint whenever its step passes the next multiple of MILESTONE_STEPS (the trainer keeps only the newest $SAVE_TOTAL_LIMIT).
MILESTONE_STEPS=${MILESTONE_STEPS:-8180}   # 20 checkpoints (409-step grid)
milestones() {
    local next last
    mkdir -p "$WORK_DIR/milestones"
    while sleep 120; do
        last=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | grep -E 'checkpoint-[0-9]+$' | sed 's#.*/checkpoint-##' | sort -n | tail -1)
        [[ -n "$last" ]] || continue
        next=$(( last / MILESTONE_STEPS * MILESTONE_STEPS ))
        [[ "$next" -gt 0 ]] || continue
        [[ -e "$WORK_DIR/milestones/checkpoint-$last" || -e "$WORK_DIR/milestones/.m$next" ]] && continue
        cp -rL "$CKPT_DIR/checkpoint-$last" "$WORK_DIR/milestones/checkpoint-$last.tmp" 2>/dev/null \
            && mv "$WORK_DIR/milestones/checkpoint-$last.tmp" "$WORK_DIR/milestones/checkpoint-$last" && touch "$WORK_DIR/milestones/.m$next" && echo "milestone copy: checkpoint-$last"
    done
}
mkdir -p "$WORK_DIR" "$LOG_DIR"
if [[ "$SAVE_TOTAL_LIMIT" -gt 0 ]]; then   # with SAVE_TOTAL_LIMIT=0 every checkpoint is kept, so milestone copies would only duplicate them
    milestones > "$LOG_DIR/milestones.log" 2>&1 &
    MILESTONE_PID=$!
    trap 'kill $MILESTONE_PID 2>/dev/null' EXIT
fi
export EXP_DIR_OVERRIDE="$EXP"
bash "$EXP/horizon/run_2node_ddp.sh" train
