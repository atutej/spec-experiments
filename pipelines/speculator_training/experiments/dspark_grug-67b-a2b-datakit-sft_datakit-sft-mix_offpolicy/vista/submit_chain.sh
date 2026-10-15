#!/bin/bash
# Submit the Grug full-mixture drafter chain (run from a LOGIN node: sbatch is refused on compute nodes).
#   JOBS=3 AFTER=<write job id> bash submit_chain.sh [--dry-run]     JOBS resumable `train` jobs, each after the previous (afterany: a job cut by its 12 h limit still hands over)
#   bash submit_chain.sh [--dry-run] upload                          the upload stage alone (HF_REPO=auto), by hand once the run is finished
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
JOB_PREFIX=dspark-grugfull
ACCOUNT=${ACCOUNT:-ASC26088}
# JOBS train jobs by default (the 12.5B-token budget is about 33 h); an explicit stage list still works. QOS qgb allows only 3 submitted jobs per user (running or pending, an idev
# counts): a chain of JOBS=3 therefore blocks every other gb submission (such as an RL run) until it drains.
JOBS=${JOBS:-3}
[[ ${#STAGES[@]} -eq 1 && "${STAGES[0]}" == train ]] && [[ -z "${EXPLICIT_STAGES:-}" ]] && { STAGES=(); for _ in $(seq "$JOBS"); do STAGES+=(train); done; }
prev="${AFTER:-}"
for s in "${STAGES[@]}"; do
    cmd=(sbatch --parsable -p gb -A "$ACCOUNT" -t 12:00:00 -J "$JOB_PREFIX-${s//,/-}" --export=ALL)
    res=$(gb_reservation_args)
    # common/slurm.sh only warns when the reservation is missing and then submits WITHOUT it; here that is an error (unless allowed).
    if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then
        echo "ERROR: no Slurm reservation would be applied (RESERVATION='${RESERVATION-}'; it is spelled CGAI-gb). ALLOW_NO_RESERVATION=1 submits without one." >&2; exit 1
    fi
    [[ -z "$res" ]] || cmd+=("$res")
    if [[ -n "$prev" ]]; then
        # The first job waits for AFTER (the write job) with afterok: a failed write must not start training. Later jobs use afterany: a job that hits its time limit (TIMEOUT) still hands over.
        if [[ "$prev" == "${AFTER:-}" ]]; then cmd+=(--dependency="afterok:$prev"); else cmd+=(--dependency="afterany:$prev"); fi
    fi
    # shellcheck disable=SC2206  # comma-joined stages become separate arguments of run.sh on purpose
    cmd+=("$HERE/run.sbatch" ${s//,/ })
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$s>"; continue; fi
    prev=$(submit "${cmd[@]}") || { echo "sbatch failed for stage $s." >&2; exit 1; }
    echo "$s: job $prev (partition gb, reservation ${RESERVATION-})"
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs/slurm/${JOB_PREFIX}-<stage>-<jobid>.out ; run logs in the experiment's WORK_DIR (settings.sh)"
