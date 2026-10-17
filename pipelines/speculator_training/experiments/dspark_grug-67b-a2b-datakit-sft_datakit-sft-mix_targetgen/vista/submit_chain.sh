#!/bin/bash
# Submit the targetgen chain (run from a LOGIN node: sbatch is refused on compute nodes):  bash submit_chain.sh [--dry-run]
#   regen (gb, 2 nodes, 12 h) -> train (gb, 2 nodes, 12 h, afterok regen; its first step is the merge of the regenerated parts)
# Two gb jobs (three when the source dataset has to be rebuilt first: a `write` job comes first); the gb QOS allows 3 submitted jobs per user, and an idev counts as one. A stage can be redone alone: STAGES="train" AFTER=<jobid> bash submit_chain.sh
# (regen resumes from the parts already written; train resumes from the last checkpoint). Overrides pass through the environment (REGEN_ROWS, WORK_DIR, ...).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../../common/slurm.sh"   # gb_reservation_args
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi
ACCOUNT=${ACCOUNT:-ASC26088}
SOURCE_DATA=${SOURCE_DATA:-/scratch/09749/atutej/marin_datakit/dataset_seq32768}
WRITE_SBATCH="$HERE/../../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data/write.sbatch"
# The source dataset must exist before regen. If it is not complete (no manifest.json), the chain starts with the Datakit `write` job (resumes from the .work dir, 1 node).
if [[ -z "${STAGES:-}" ]]; then
    if [[ -f "$SOURCE_DATA/manifest.json" && -f "$SOURCE_DATA/token_freq.pt" ]]; then STAGES="regen train"; else STAGES="write regen train"; fi
fi
export SEQ_LENGTH=${SEQ_LENGTH:-32768}
submit() {
    local out id
    out=$("$@") || { printf '%s\n' "$out" >&2; return 1; }
    id=$(printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1)
    [[ -n "$id" ]] || { echo "no job id in the sbatch output:" >&2; printf '%s\n' "$out" >&2; return 1; }
    echo "$id"
}
prev="${AFTER:-}"
for s in $STAGES; do
    case "$s" in
        write) part=gb; limit=04:00:00; script=$WRITE_SBATCH ;;
        regen) part=gb; limit=12:00:00; script=run_regen.sbatch ;;
        train) part=gb; limit=12:00:00; script=run_train.sbatch ;;
        *) echo "unknown stage '$s' (write regen train)" >&2; exit 2 ;;
    esac
    cmd=(sbatch --parsable -p "$part" -A "$ACCOUNT" -t "$limit" -J "dspark-grugon-$s" --export=ALL)
    if [[ "$part" == gb ]]; then
        res=$(gb_reservation_args)
        if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then echo "ERROR: no reservation would be applied (CGAI-gb missing); ALLOW_NO_RESERVATION=1 overrides." >&2; exit 1; fi
        [[ -z "$res" ]] || cmd+=("$res")
    fi
    [[ -z "$prev" ]] || cmd+=(--dependency="afterok:$prev")
    [[ "$script" == /* ]] || script="$HERE/$script"
    cmd+=("$script")
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$s>"; continue; fi
    prev=$(submit "${cmd[@]}") || { echo "sbatch failed for stage $s." >&2; exit 1; }
    echo "$s: job $prev (partition $part)"
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs/slurm/dspark-grugon-<stage>-<jobid>.out ; run logs in the experiment's WORK_DIR"
