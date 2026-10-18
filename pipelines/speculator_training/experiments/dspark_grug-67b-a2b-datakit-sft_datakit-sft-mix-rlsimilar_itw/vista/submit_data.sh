#!/bin/bash
# Submit the data creation: ONE job on both gb nodes of the CGAI-gb reservation (build_data.sbatch: embedding on all 8 GPUs, then selection and the dataset). Run from a LOGIN node.   bash submit_data.sh [--dry-run]
# DATA_JOBS=n chains n resumable jobs (each afterany the previous): a job cut by the time limit is continued, a job with nothing left to do skips its stages. AFTER=<jobid> makes the first job wait.
# The gb QOS allows 3 submitted jobs per user and 12 h per job; this uses 1 per chained job.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../../common/slurm.sh"   # gb_reservation_args
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi
ACCOUNT=${ACCOUNT:-ASC26088}; DATA_JOBS=${DATA_JOBS:-1}; prev="${AFTER:-}"
submit() {
    local out id
    out=$("$@") || { printf '%s\n' "$out" >&2; return 1; }
    id=$(printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1)
    [[ -n "$id" ]] || { echo "no job id in the sbatch output:" >&2; printf '%s\n' "$out" >&2; return 1; }
    echo "$id"
}
res=$(gb_reservation_args)
if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then echo "ERROR: no reservation would be applied (CGAI-gb missing); ALLOW_NO_RESERVATION=1 overrides." >&2; exit 1; fi
for i in $(seq "$DATA_JOBS"); do
    cmd=(sbatch --parsable -p gb -A "$ACCOUNT" -t 12:00:00 -J dspark-rlsim-data --export=ALL)
    [[ -z "$res" ]] || cmd+=("$res")
    [[ -z "$prev" ]] || cmd+=(--dependency="afterany:$prev")
    cmd+=("$HERE/build_data.sbatch")
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$i>"; continue; fi
    prev=$(submit "${cmd[@]}") || { echo "sbatch failed for job $i." >&2; exit 1; }
    echo "data job $i: $prev"
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs/slurm/dspark-rlsim-data-<jobid>.out ; per-worker logs in runs/grug_rlsimilar_data/emb/worker*.log ; then check runs/grug_rlsimilar_data/sel/summary.json (duplicate_rows) before training"
