#!/bin/bash
# Submit the chain (run from a LOGIN node: sbatch is refused on compute nodes):  bash submit_chain.sh [--dry-run]
#   regen (gb, 2 nodes, 12 h) -> [regen again, afterany: resumes from the parts already written; it exits quickly when nothing is left] -> train (gb, 2 nodes, 12 h, afterok the last regen;
#   its first step is the merge of the regenerated parts)
# REGEN_JOBS=2 (default) covers a regeneration that does not fit in one 12 h job (about 17 h of regeneration is expected for 400k rows); REGEN_JOBS=1 submits only one. The gb QOS allows 3 submitted jobs per user (an idev counts as one).
# A stage can be redone alone: STAGES="train" AFTER=<jobid> bash submit_chain.sh (regen resumes from the parts already written; train resumes from the last checkpoint).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../../common/slurm.sh"   # gb_reservation_args
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi
ACCOUNT=${ACCOUNT:-ASC26088}; REGEN_JOBS=${REGEN_JOBS:-2}
if [[ -z "${STAGES:-}" ]]; then STAGES="regen train"; fi
submit() {
    local out id
    out=$("$@") || { printf '%s\n' "$out" >&2; return 1; }
    id=$(printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1)
    [[ -n "$id" ]] || { echo "no job id in the sbatch output:" >&2; printf '%s\n' "$out" >&2; return 1; }
    echo "$id"
}
res=$(gb_reservation_args)
if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then echo "ERROR: no reservation would be applied (CGAI-gb missing); ALLOW_NO_RESERVATION=1 overrides." >&2; exit 1; fi
prev="${AFTER:-}"; prev_dep=afterok
run() {   # stage script dependency-type
    local cmd=(sbatch --parsable -N 2 -p gb -A "$ACCOUNT" -t 12:00:00 -J "dspark-drdoomtgfull-$1" --export=ALL)
    [[ -z "$res" ]] || cmd+=("$res")
    [[ -z "$prev" ]] || cmd+=(--dependency="$3:$prev")
    cmd+=("$HERE/$2")
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$1>"; return 0; fi
    prev=$(submit "${cmd[@]}") || { echo "sbatch failed for stage $1." >&2; exit 1; }
    echo "$1: job $prev"
}
for s in $STAGES; do
    case "$s" in
        regen) run regen run_regen.sbatch afterok
               for i in $(seq 2 "$REGEN_JOBS"); do run regen run_regen.sbatch afterany; done ;;
        train) run train run_train.sbatch afterok ;;
        *) echo "unknown stage '$s' (regen train)" >&2; exit 2 ;;
    esac
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs/slurm/dspark-drdoomtgfull-<stage>-<jobid>.out ; run logs in runs/grug_dspark_drdoom_targetgen/logs"
