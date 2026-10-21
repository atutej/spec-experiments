#!/bin/bash
# Submit the training job(s) (run from a LOGIN node: sbatch is refused on compute nodes).   bash submit_chain.sh [--dry-run]
#   JOBS=1 (default): ONE 12 h job on 2 gb nodes of the CGAI-gb reservation. JOBS=n: n resumable jobs, each afterany the previous (a job cut by its time limit still hands over).
#   AFTER=<jobid> makes the first job wait for a queued job. The gb QOS allows 3 submitted jobs per user (an idev counts as one) and 12 h per job.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../../common/slurm.sh"   # gb_reservation_args
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi
ACCOUNT=${ACCOUNT:-ASC26088}; JOBS=${JOBS:-1}; prev="${AFTER:-}"
submit() {
    local out id
    out=$("$@") || { printf '%s\n' "$out" >&2; return 1; }
    id=$(printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1)
    [[ -n "$id" ]] || { echo "no job id in the sbatch output:" >&2; printf '%s\n' "$out" >&2; return 1; }
    echo "$id"
}
for i in $(seq "$JOBS"); do
    cmd=(sbatch --parsable -p gb -A "$ACCOUNT" -t 12:00:00 -J dspark-grugdrdoom-train --export=ALL)
    res=$(gb_reservation_args)
    if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then echo "ERROR: no reservation would be applied (CGAI-gb missing); ALLOW_NO_RESERVATION=1 overrides." >&2; exit 1; fi
    [[ -z "$res" ]] || cmd+=("$res")
    [[ -z "$prev" ]] || cmd+=(--dependency="afterany:$prev")
    cmd+=("$HERE/run_train.sbatch")
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$i>"; continue; fi
    prev=$(submit "${cmd[@]}") || { echo "sbatch failed for job $i." >&2; exit 1; }
    echo "train job $i: $prev"
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs/slurm/dspark-grugdrdoom-train-<jobid>.out ; run logs in runs/grug_dspark_itw_drdoom/logs"
