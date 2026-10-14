#!/bin/bash
# Submit the Datakit SFT mixture job(s): fetch -> stats -> select. Run from a LOGIN node.
#   bash submit_fetch.sh [--dry-run]                          one job on a gg node (2-day limit, no reservation)
#   PARTITION=gb SHARDS=2 bash submit_fetch.sh [--dry-run]    two balanced shards, one per reservation node (12-hour limit)
# PARTITION=gb uses the CGAI-gb reservation (common/slurm.sh) and its QOS limits: 12:00:00 per job, 3 submitted jobs per user (an idev counts as one).
#   RESUME=1 TAG=resume1 PARTITION=gb SHARDS=2 CONCURRENT=24 bash submit_fetch.sh   re-splits what is left (after a time limit) across the nodes
#   RESUME=1 TAG=resume1 PARTITION=gb SHARDS=1 CONCURRENT=24 bash submit_fetch.sh   the same on ONE node (e.g. the other node is training)
# ONLY="1" resubmits just shard 1 (e.g. after a time limit; finished work is skipped). CONCURRENT=<n> WORKERS=<n> pass through to the job.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/scratch/09749/atutej/marin_datakit
source "$HERE/../../../../../common/slurm.sh" 2>/dev/null || true       # gb_reservation_args
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
PARTITION=${PARTITION:-gg}; SHARDS=${SHARDS:-1}
if [[ "$PARTITION" == gb ]]; then TIME=${TIME:-12:00:00}; else TIME=${TIME:-2-00:00:00}; fi
if [[ $DRY -eq 0 ]]; then command -v sbatch >/dev/null || { echo "sbatch not found (login node?)" >&2; exit 1; }; fi
mkdir -p /scratch/09749/atutej/marin_speculator/logs/slurm
submit() { local out; out=$("$@") || { printf '%s\n' "$out" >&2; return 1; }; printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1; }
base=(sbatch --parsable -p "$PARTITION" -t "$TIME")
if [[ "$PARTITION" == gb ]]; then
    res=$(gb_reservation_args)
    # common/slurm.sh only warns when the reservation is missing and then submits WITHOUT it (a mistyped name, e.g. CGAI_gb instead of CGAI-gb, or an expired
    # reservation would silently queue these jobs in the general gb partition). Here that is an error, unless explicitly allowed.
    if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then
        echo "ERROR: PARTITION=gb but no Slurm reservation would be applied (RESERVATION='${RESERVATION-}'). The reservation is spelled CGAI-gb; list them with: scontrol show reservation" >&2
        echo "       Set ALLOW_NO_RESERVATION=1 to submit to the gb partition without one." >&2
        exit 1
    fi
    [[ -z "$res" ]] || base+=("$res")
fi
if [[ "$SHARDS" -le 1 && "${RESUME:-0}" != 1 ]]; then   # one job over every source; a resume goes through the shard logic below even with one shard (it skips finished work per source)
    cmd=("${base[@]}" -J datakit-sft-mix --export=ALL "$HERE/fetch.sbatch")
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; else submit "${cmd[@]}"; fi
    exit 0
fi
[[ "$PARTITION" == gb && "$SHARDS" -gt 2 ]] && echo "note: the gb QOS allows 3 submitted jobs per user in total (an idev counts)" >&2
# RESUME=1 TAG=<name>: re-split only the work that is left (what is on disk decides), into lists tagged <name>; the first split's lists stay untouched.
export MARIN_PREFIX=${MARIN_PREFIX:-$ROOT/out} HF_HOME=${HF_HOME:-$ROOT/hf_home}
shard_flags=(); [[ "${RESUME:-0}" == 1 ]] && shard_flags+=(--only-todo --tag "${TAG:?set TAG=<name> with RESUME=1}")
# REUSE_LISTS=1: keep the list files that already exist (generated on a compute node), e.g. if the login node cannot run the split.
if [[ "${REUSE_LISTS:-0}" == 1 && -f "$ROOT/shards/shard-0-of-$SHARDS${TAG:+-$TAG}.txt" ]]; then
    echo "reusing the existing lists in $ROOT/shards/ (tag '${TAG:-}')"
else
    $ROOT/venv/bin/python -I "$HERE/build_datakit_sft_mix.py" shards --n "$SHARDS" --out-dir "$ROOT/shards" "${shard_flags[@]}" || exit 1
fi
for ((k = 0; k < SHARDS; k++)); do
    if [[ -n "${ONLY:-}" && " $ONLY " != *" $k "* ]]; then continue; fi
    cmd=("${base[@]}" -J "datakit-shard-$k-of-$SHARDS" --export="ALL,SHARD=$k,SHARDS=$SHARDS,SHARD_TAG=${TAG:-}" "$HERE/fetch.sbatch")
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; else echo "shard $k: job $(submit "${cmd[@]}")"; fi
done
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; logs/slurm/datakit-shard-*.out ; the last shard to finish writes $ROOT/selection.json"
