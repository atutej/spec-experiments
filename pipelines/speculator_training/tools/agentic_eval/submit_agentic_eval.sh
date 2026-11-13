#!/bin/bash
# Submit one agentic eval (run_agentic_eval.sbatch). Run from a LOGIN node.
#
#   ARM=nofin24h DRAFTER=atutej/dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-filtered_itw-20261112 BENCH=ot-tblite \
#       HARBOR_EXTRA="--n-tasks 3 --n-attempts 1" TIME=02:00:00 bash submit_agentic_eval.sh
#   ARM=plain BENCH=ot-tblite bash submit_agentic_eval.sh          # no DRAFTER: plain decoding
#   DRY=1 ...                                                       # print the sbatch command and stop
#
#   ARM          label of this run (letters, digits, _ . -); the result dir is $RESULTS/$BENCH/$ARM and must not exist yet
#   BENCH        ot-tblite | tb2 (Marin eval policy Harbor configs, Terminus-2 on Daytona)
#   DRAFTER      DSpark checkpoint: a directory with config.json, or a Hugging Face repo id; empty = plain decoding
#   HARBOR_EXTRA extra flags for `harbor run` (the policy config has 16 concurrent trials and 3 attempts per task; --n-tasks / --n-attempts override)
#   Machine settings (defaults are Vista gb):
#     PARTITION (gb), ACCOUNT (none: the login's only project), TIME (08:00:00), GPUS (GPUs per node, 4), RESERVATION (unset: the Vista CGAI-gb one if it exists;
#     empty: none; a name: that one), VLLM_ENV (conda env with vLLM, default vllm), VLLM_OVERLAY (only for a vLLM without Grug DSpark support),
#     KEYS_ENV (file with `export DAYTONA_API_KEY=...`, default ~/keys.env), ROOT (workspace root, default: the parent of this repo),
#     RESULTS (default $ROOT/runs/agentic_eval_20261113/results)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${ROOT:-$(cd "$HERE/../../../../.." && pwd)}   # <ROOT>/spec-experiments/pipelines/speculator_training/tools/agentic_eval
die() { echo "ERROR: $*" >&2; exit 1; }
: "${ARM:?set ARM}" "${BENCH:?set BENCH (ot-tblite | tb2)}"
[[ "$ARM" =~ ^[A-Za-z0-9_.-]+$ ]] || die "ARM must be letters, digits, _ . -"
[[ "$BENCH" == ot-tblite || "$BENCH" == tb2 ]] || die "BENCH must be ot-tblite or tb2"
DRAFTER=${DRAFTER:-}
if [[ -n "$DRAFTER" && ! -f "$DRAFTER/config.json" && ! "$DRAFTER" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then die "DRAFTER $DRAFTER is neither a directory with config.json nor a Hugging Face repo id"; fi
RESULTS=${RESULTS:-$ROOT/runs/agentic_eval_20261113/results}
OUT=$RESULTS/$BENCH/$ARM
[[ ! -e "$OUT" ]] || die "$OUT exists: results are never overwritten"
PARTITION=${PARTITION:-gb}
# Reservation: RESERVATION=<name> uses it, RESERVATION= (empty) none; unset: the Vista gb helper (warns and returns nothing when its reservation has expired).
if [[ "${RESERVATION+set}" == set ]]; then res=${RESERVATION:+--reservation=$RESERVATION}
elif [[ "$PARTITION" == gb && -f "$ROOT/spec-experiments/common/slurm.sh" ]]; then source "$ROOT/spec-experiments/common/slurm.sh"; res=$(gb_reservation_args)
else res=""; fi
logdir=$ROOT/logs/slurm
cmd=(sbatch --parsable -p "$PARTITION" -t "${TIME:-08:00:00}" -J "agentic-$BENCH-$ARM" -o "$logdir/%x-%j.out")
[[ -z "${ACCOUNT:-}" ]] || cmd+=(-A "$ACCOUNT")
[[ -z "$res" ]] || cmd+=($res)
cmd+=("$HERE/run_agentic_eval.sbatch")   # no --export: Slurm exports the submit environment by default (Horizon rejects --export)
echo "arm $ARM | bench $BENCH | drafter ${DRAFTER:-none (plain)} | partition $PARTITION ${res:+| $res }| extra '${HARBOR_EXTRA:-}' | out $OUT"
if [[ "${DRY:-0}" == 1 ]]; then echo "${cmd[*]}"; exit 0; fi
command -v sbatch >/dev/null || die "sbatch not found (run from a login node)"
mkdir -p "$logdir" "$RESULTS/$BENCH"
id=$( export ROOT ARM BENCH OUT DRAFTER HARBOR_EXTRA="${HARBOR_EXTRA:-}" SERVER_EXTRA="${SERVER_EXTRA:-}" GPUS="${GPUS:-4}" VLLM_ENV="${VLLM_ENV:-vllm}" VLLM_OVERLAY="${VLLM_OVERLAY:-}" \
     KEYS_ENV="${KEYS_ENV:-$HOME/keys.env}"; "${cmd[@]}" 2>&1 ) || die "sbatch failed: $id"
echo "job $id"
