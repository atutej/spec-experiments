#!/bin/bash
# Submit one drafter's Evalchemy acceptance-length run (run_evalchemy_al.sbatch). Run from a LOGIN node.
#   ARM=A DRAFTER=atutej/dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-20261108 TIME=08:00:00 bash submit_evalchemy_al.sh
#   AFTER=<jobid>  start only after that job has ended (run drafters serially);  DRY=1 prints the sbatch command only.
#   TASKS, SEEDS, TEMPERATURE, PARTITION (debug), ACCOUNT, GPUS, VLLM_ENV, ROOT, RESULTS (default $ROOT/runs/evalchemy_al_20261114/results) as in run_evalchemy_al.sbatch.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../../.." && pwd)}
die() { echo "ERROR: $*" >&2; exit 1; }
: "${ARM:?set ARM}"
[[ "$ARM" =~ ^[A-Za-z0-9_.-]+$ ]] || die "ARM must be letters, digits, _ . -"
DRAFTER=${DRAFTER:-}
RESULTS=${RESULTS:-$ROOT/runs/evalchemy_al_20261114/results}
OUT=$RESULTS/$ARM
[[ ! -e "$OUT" ]] || die "$OUT exists: results are never overwritten"
logdir=$ROOT/logs/slurm
cmd=(sbatch --parsable -p "${PARTITION:-debug}" -t "${TIME:-08:00:00}" -J "evalchemy-al-$ARM" -o "$logdir/%x-%j.out")
[[ -z "${ACCOUNT:-}" ]] || cmd+=(-A "$ACCOUNT")
[[ -z "${AFTER:-}" ]] || cmd+=(--dependency="afterany:$AFTER")
cmd+=("$HERE/run_evalchemy_al.sbatch")   # no --export: Slurm exports the submit environment (Horizon rejects --export)
echo "arm $ARM | drafter ${DRAFTER:-none} | tasks ${TASKS:-default} | seeds ${SEEDS:-policy 43} | out $OUT"
if [[ "${DRY:-0}" == 1 ]]; then echo "${cmd[*]}"; exit 0; fi
mkdir -p "$logdir" "$RESULTS"
id=$( export ROOT ARM OUT DRAFTER TASKS SEEDS TEMPERATURE LIMIT KEEP_OUTPUT GPUS VLLM_ENV; "${cmd[@]}" 2>&1 | tail -1 ) || die "sbatch failed"
echo "job $id"
