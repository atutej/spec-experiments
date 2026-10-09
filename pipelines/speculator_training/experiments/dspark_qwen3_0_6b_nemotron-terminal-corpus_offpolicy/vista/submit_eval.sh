#!/bin/bash
# Submit the rollout acceptance eval (eval.sh) as one gb job. Run from a LOGIN node (sbatch is refused on compute nodes).
#   bash submit_eval.sh [--dry-run] [CHECKPOINT_NAME]      default checkpoint 0
# AFTER=<jobid> waits for a queued job first (e.g. the training job). Time limit = qgb maximum (12:00:00); qgb allows 3 submitted jobs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
CKPT_NAME=${1:-0}
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi
cmd=(sbatch --parsable -p gb -t 12:00:00 -J offpolicy-eval-$CKPT_NAME --export=ALL)
[[ -n "${AFTER:-}" ]] && cmd+=(--dependency="afterok:$AFTER")
cmd+=("$HERE/eval.sbatch" "$CKPT_NAME")
if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; exit 0; fi
out=$("${cmd[@]}") || { printf '%s\n' "$out" >&2; exit 1; }   # Vista prints a banner before the id: keep the bare-number line
id=$(printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1)
[[ -n "$id" ]] || { echo "no job id in the sbatch output:" >&2; printf '%s\n' "$out" >&2; exit 1; }
echo "eval: job $id (partition gb). Slurm log: logs/slurm/offpolicy-eval-$CKPT_NAME-$id.out"
