#!/bin/bash
# Submit the Grug 100k-sample drafter job (run from a LOGIN node: sbatch is refused on compute nodes).
#   bash submit_chain.sh [--dry-run] [job ...]     stages: train upload   (default: one job, `train`)
# One job on a gb node of the CGAI-gb reservation (common/slurm.sh), 12 h (the gb QOS maximum; at most 3 submitted jobs per user, an idev counts as one).
# `train` resumes from $CKPT_DIR, so a job that hits its time limit is continued by submitting again: `AFTER=<jobid> bash submit_chain.sh train`.
# The dataset is NOT built here: it comes from the Datakit command (NOTES.md). Overrides pass through the environment, e.g. MAX_STEPS=10 WORK_DIR=... bash submit_chain.sh --dry-run
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../../common/slurm.sh"   # gb_reservation_args
DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(train)
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi
submit() {
    local out id
    out=$("$@") || { printf '%s\n' "$out" >&2; return 1; }
    id=$(printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1)
    [[ -n "$id" ]] || { echo "no job id in the sbatch output:" >&2; printf '%s\n' "$out" >&2; return 1; }
    echo "$id"
}
JOB_PREFIX=dspark-grug100k
prev="${AFTER:-}"
for s in "${STAGES[@]}"; do
    cmd=(sbatch --parsable -p gb -t 12:00:00 -J "$JOB_PREFIX-${s//,/-}" --export=ALL)
    res=$(gb_reservation_args)
    # common/slurm.sh only warns when the reservation is missing and then submits WITHOUT it; here that is an error (unless allowed).
    if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then
        echo "ERROR: no Slurm reservation would be applied (RESERVATION='${RESERVATION-}'; it is spelled CGAI-gb). ALLOW_NO_RESERVATION=1 submits without one." >&2; exit 1
    fi
    [[ -z "$res" ]] || cmd+=("$res")
    [[ -n "$prev" ]] && cmd+=(--dependency="afterok:$prev")
    # shellcheck disable=SC2206  # comma-joined stages become separate arguments of run.sh on purpose
    cmd+=("$HERE/run.sbatch" ${s//,/ })
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$s>"; continue; fi
    prev=$(submit "${cmd[@]}") || { echo "sbatch failed for stage $s." >&2; exit 1; }
    echo "$s: job $prev (partition gb, reservation ${RESERVATION-})"
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs/slurm/${JOB_PREFIX}-<stage>-<jobid>.out ; run logs in the experiment's WORK_DIR (settings.sh)"
