#!/bin/bash
# Drive the full-epoch drafter run as a series of resumable 12 h `train` jobs, ONE AT A TIME, until the epoch is finished (the trainer's `epoch0_end` checkpoint link exists).
# Why a driver: the epoch is about 11 jobs, but the gb QOS allows only 3 submitted jobs per user (a submission over the limit is rejected, not queued), and sbatch is refused on compute nodes,
# so the next job is submitted from a LOGIN node after the previous one has left the queue. Run it inside tmux/screen (or nohup) on a login node; it polls squeue lightly (every INTERVAL s).
#
#   bash vista/drive_chain.sh [--dry-run]
#   env: AFTER=<job id> (first job waits for it with afterok, e.g. the write job), MAX_JOBS=14, INTERVAL=120, MIN_RUNTIME=600 (a job shorter than this counts as a fast failure; two in a row stop the driver)
# It never deletes anything and never uploads. To stop it: Ctrl-C (a running job continues; scancel it separately). Rerunning it continues the run from the last checkpoint.
# Untested at full scale: written 2026-10-14 night; only its --dry-run path has been exercised.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$HERE/.." && pwd)"
REPO_DIR="$(cd "$HERE/../../../../.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
source "$EXP_DIR/settings.sh"        # CKPT_DIR, WORK_DIR
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
MAX_JOBS=${MAX_JOBS:-14}; INTERVAL=${INTERVAL:-120}; MIN_RUNTIME=${MIN_RUNTIME:-600}
finished() { [[ -e "$CKPT_DIR/epoch0_end" ]]; }
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found: run this on a login node" >&2; exit 1; fi
if finished; then echo "already finished: $CKPT_DIR/epoch0_end exists"; exit 0; fi
if [[ $DRY -eq 1 ]]; then
    echo "[dry run] would submit, one at a time, up to $MAX_JOBS of:"
    AFTER="${AFTER:-}" EXPLICIT_STAGES=1 bash "$HERE/submit_chain.sh" --dry-run train
    echo "[dry run] after each job leaves the queue: finished if $CKPT_DIR/epoch0_end exists; else submit the next; poll every ${INTERVAL}s; stop after two jobs shorter than ${MIN_RUNTIME}s"
    exit 0
fi
prev="${AFTER:-}"; fast=0; n=0
while (( n < MAX_JOBS )); do
    out=$(AFTER="$prev" EXPLICIT_STAGES=1 bash "$HERE/submit_chain.sh" train 2>&1)
    id=$(printf '%s\n' "$out" | sed -n 's/^train: job \([0-9][0-9]*\).*/\1/p' | tail -1)
    if [[ -z "$id" ]]; then
        echo "$(date +%T) submit failed (a QOS slot may be taken); retrying in 5 min:"; printf '%s\n' "$out" | tail -3; sleep 300; continue
    fi
    n=$((n+1)); prev=""
    echo "$(date +%T) job $n: $id submitted"
    while squeue -j "$id" -h 2>/dev/null | grep -q .; do sleep "$INTERVAL"; done
    read -r state elapsed < <(sacct -j "$id" -X -n -o State,ElapsedRaw 2>/dev/null | awk 'NR==1{print $1, $2}')
    echo "$(date +%T) job $id ended: state=${state:-?} elapsed=${elapsed:-?}s"
    if finished; then echo "finished: $CKPT_DIR/epoch0_end exists after $n jobs"; exit 0; fi
    if [[ "${elapsed:-0}" -lt "$MIN_RUNTIME" ]]; then fast=$((fast+1)); else fast=0; fi
    if (( fast >= 2 )); then echo "two jobs in a row ended in under ${MIN_RUNTIME}s: stopping (look at logs/slurm/${JOB_PREFIX:-dspark-grugfull}-train-<id>.out)"; exit 1; fi
done
echo "reached MAX_JOBS=$MAX_JOBS without finishing; rerun to continue"; exit 1
