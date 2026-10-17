#!/bin/bash
# Submit serve_eval.sh as a one-node gb job in the CGAI-gb reservation (run from a LOGIN node: sbatch is refused on compute nodes).
#   MODEL=<target repo> OUT=<dir> ARMS="label=<drafter dir> ..." [PLAIN=1] PROMPTS_DIR=... MAX_PROMPTS=... MAX_TOKENS=... MAX_MODEL_LEN=... SPEC_EXTRA=... bash submit_eval.sh [--dry-run] [job name]
# The job first makes sure the target's weights are in the Hugging Face cache (HF_HOME of env.sh); download them beforehand when you can: 134 GB eats about 1-2 h of the job.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../common/slurm.sh"   # gb_reservation_args
DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
: "${OUT:?set OUT (a NEW directory)}"
cmd=(sbatch --parsable -p gb -A "${ACCOUNT:-ASC26088}" -t "${TIME:-12:00:00}" -J "${1:-serve-eval}" --export=ALL)
res=$(gb_reservation_args)
if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then echo "ERROR: no reservation would be applied (CGAI-gb missing); ALLOW_NO_RESERVATION=1 overrides." >&2; exit 1; fi
[[ -z "$res" ]] || cmd+=("$res")
cmd+=("$HERE/eval_arms.sbatch")
if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; exit 0; fi
command -v sbatch >/dev/null || { echo "sbatch not found (login node?)" >&2; exit 1; }
mkdir -p /scratch/09749/atutej/marin_speculator/logs/slurm "$OUT"
"${cmd[@]}"
