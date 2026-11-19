#!/bin/bash
# Submit from a LOGIN node (sbatch is refused on compute nodes).
#   bash horizon/submit.sh data                      build the mixed dataset (CPU node)           SMOKE_DATA=1 for the small test dataset
#   AFTER=<data job id> bash horizon/submit.sh train [--dry-run]         JOBS (default 1) training jobs, each afterany the previous; TIME per job (default 48:00:00; the debug partition has no limit)
#   smoke test:  SMOKE_DATA=1 bash horizon/submit.sh data   then   SMOKE=1 MAX_STEPS=10 WORK_DIR=$PROJECT_ROOT/runs/mix_smoke DATA_DIR=$PROJECT_ROOT/runs/grug_dspark_drdoom_layers6-taps5_datakit-orig-bigmix-tg-mix_smoke/data NCCL_DEBUG=INFO TIME=01:00:00 bash horizon/submit.sh train
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_DIR="$(cd "$HERE/.." && pwd)"
DRY=0; [[ "${2:-}" == "--dry-run" ]] && DRY=1
what=${1:-train}
run() { if [[ $DRY -eq 1 ]]; then echo "DRY: $*" >&2; echo "<job>"; else "$@"; fi; }
case "$what" in
  data) run sbatch --parsable --export=ALL "$HERE/build_data.sbatch" ;;
  train)
    prev="${AFTER:-}"   # AFTER=<job id>: the first job waits for it (afterok: a failed data build must not start training)
    for _ in $(seq "${JOBS:-1}"); do
        cmd=(sbatch --parsable --export=ALL -t "${TIME:-48:00:00}" -J "mix-train")
        if [[ -n "$prev" ]]; then cmd+=(--dependency="$([[ "$prev" == "${AFTER:-}" ]] && echo "${DEP_TYPE:-afterok}" || echo afterany):$prev"); fi
        cmd+=("$HERE/run_train.sbatch")
        prev=$(run "${cmd[@]}" | tail -1); echo "job $prev"
    done ;;
  *) echo "usage: submit.sh data|train [--dry-run]" >&2; exit 2 ;;
esac
