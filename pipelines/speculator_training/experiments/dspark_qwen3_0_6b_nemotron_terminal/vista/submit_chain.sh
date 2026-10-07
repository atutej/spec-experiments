#!/bin/bash
# Submit the pipeline stages as a chain of Slurm jobs (run from a LOGIN node: sbatch is refused on compute nodes).
#   bash submit_chain.sh [--dry-run] [stage ...]     stages: export regen prepare train   (default: all)
# Each submitted stage waits for the previous submitted one (afterok). Submit a subset to rerun or resume one
# stage (e.g. `bash submit_chain.sh train` resumes training from its checkpoints). Overrides pass through the
# environment of this command, e.g. REGEN_LIMIT=1000 WORK_DIR=... bash submit_chain.sh --dry-run
#
# Time limits are the maximum the partition's QOS allows (rule of thumb for all jobs: qgb 12:00:00,
# qgg / qgh 2-00:00:00). The qgb QOS also allows only 3 submitted jobs per user: this chain uses 2 gb jobs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(export regen prepare train)
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi

partition() { case "$1" in export|prepare) echo gg ;; regen|train) echo gb ;; *) return 1 ;; esac; }
maxtime()   { case "$1" in gb) echo 12:00:00 ;; *) echo 2-00:00:00 ;; esac; }

prev=""
for s in "${STAGES[@]}"; do
    p=$(partition "$s") || { echo "unknown stage '$s' (export regen prepare train)" >&2; exit 2; }
    cmd=(sbatch --parsable -p "$p" -t "$(maxtime "$p")" -J "dspark-nemotron-$s" --export=ALL)
    [[ -n "$prev" ]] && cmd+=(--dependency="afterok:$prev")
    cmd+=("$HERE/run.sbatch" "$s")
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$s>"; continue; fi
    prev=$("${cmd[@]}") || { echo "sbatch failed for stage $s" >&2; exit 1; }
    echo "$s: job $prev (partition $p)"
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs in the workspace logs/slurm/, run logs in \$WORK_DIR/logs/"
