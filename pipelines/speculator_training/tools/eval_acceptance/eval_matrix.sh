#!/bin/bash
# Evaluate drafters on benchmarks: pick the drafters and the benchmarks, and choose how the (benchmark x drafter) pairs are spread over one-node gb jobs.
# Run from a LOGIN node (sbatch is refused on compute nodes).
#
#   DRAFTERS="a=<ckpt dir> b=<ckpt dir>@dspark@5+overlay" BENCHES="speedbench datakit" bash eval_matrix.sh [--serial | --per-drafter | --per-pair] [--dry-run]
#   bash eval_matrix.sh --drafter a=<ckpt dir> --drafter b=<ckpt dir>@dspark@5+overlay --bench speedbench --bench bigmix --per-drafter
#
# Both lists default to EMPTY: nothing is evaluated until you give them (flags, env, or a local eval_local.sh next to this file, which is sourced when present; keep that file out of git).
#   drafter   label=<ckpt dir>[@<method>[@<speculative tokens>]][+overlay]    method dspark, 8 tokens by default; "+overlay" puts the vLLM overlay with the Grug-layer drafter in front of the
#             wheel (needed for Grug-layer drafters; VLLM_OVERLAY overrides its location). Labels: letters, digits, _ . -
#   bench     one of: speedbench (SPEED-Bench qualitative, AIPerf), datakit (Datakit-SFT held-out, first + later turns), bigmix (big-mix held-out)   (settings: eval_benchmarks.sh)
# How the pairs are spread:
#   --serial (default)  ONE job runs every pair one after the other (one node; the time limit grows with the number of pairs)
#   --per-drafter       one job per drafter, which runs all its benchmarks (jobs run side by side as nodes free up)
#   --per-pair          one job per (benchmark, drafter)
# Options: --dry-run (print, submit nothing)  --stamp S  --out-base DIR (default runs/eval_matrix_<stamp>; rerun with the same dir to fill in missing pairs: finished pairs are skipped)
#          --time HH:MM:SS (per job; default 1 h per pair + 15 min, at most 12 h)  --model <target repo> (default Dr. Doom)
# Results: <out base>/<benchmark>/<drafter label>/...   Settings travel in the environment: T=1, probabilistic draft sampling with standard rejection.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../common/slurm.sh"   # gb_reservation_args
source "$HERE/eval_benchmarks.sh"
[[ -f "$HERE/eval_local.sh" ]] && source "$HERE/eval_local.sh"
DRAFTERS=${DRAFTERS:-}; BENCHES=${BENCHES:-}; JOBS=${JOBS:-1}; DRY=0; STAMP=${STAMP:-$(date +%Y%m%d-%H%M)}; OUT_BASE=${OUT_BASE:-}; TIME=${TIME:-}
MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom}
VLLM_OVERLAY=${VLLM_OVERLAY:-$ROOT/tmp/grug_dspark_serving_smoke_2026-10-27/vllm_overlay}
die() { echo "ERROR: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
    case $1 in
        --drafter) DRAFTERS+=" $2"; shift 2 ;;
        --bench) BENCHES+=" $2"; shift 2 ;;
        --serial) JOBS=1; shift ;;
        --per-drafter) JOBS=drafter; shift ;;
        --per-pair) JOBS=pair; shift ;;
        --dry-run) DRY=1; shift ;;
        --stamp) STAMP=$2; shift 2 ;;
        --out-base) OUT_BASE=$2; shift 2 ;;
        --time) TIME=$2; shift 2 ;;
        --model) MODEL=$2; shift 2 ;;
        -h|--help) sed -n 2,22p "$0"; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done
[[ -n "${DRAFTERS// }" ]] || die "no drafters: set DRAFTERS or pass --drafter label=<ckpt dir>  (see --help)"
[[ -n "${BENCHES// }" ]] || die "no benchmarks: set BENCHES or pass --bench NAME  (known: $BENCHES_KNOWN)"
BENCHES=$(echo $BENCHES); DRAFTERS=$(echo $DRAFTERS)
OUT_BASE=${OUT_BASE:-$RUNS/eval_matrix_$STAMP}

# ---- normalise and check the drafters: label=dir@method@N[+overlay]
declare -a LABELS=() SPECS=(); declare -A SEEN=(); NEED_OVERLAY=0
for d in $DRAFTERS; do
    [[ "$d" == *=* ]] || die "drafter '$d': expected label=<ckpt dir>"
    label=${d%%=*}; rest=${d#*=}; ov=""
    [[ "$label" =~ ^[A-Za-z0-9_.-]+$ ]] || die "drafter label '$label': use letters, digits, _ . -"
    [[ -z "${SEEN[$label]:-}" ]] || die "drafter label '$label' given twice"; SEEN[$label]=1
    if [[ "$rest" == *+overlay ]]; then ov="+overlay"; rest=${rest%+overlay}; NEED_OVERLAY=1; fi
    IFS=@ read -r dir method ntok <<< "$rest"; method=${method:-dspark}; ntok=${ntok:-8}
    [[ "$ntok" =~ ^[0-9]+$ ]] || die "drafter '$label': speculative tokens '$ntok' is not a number"
    [[ -f "$dir/config.json" && -f "$dir/model.safetensors" ]] || die "drafter '$label': no config.json / model.safetensors in $dir"
    LABELS+=("$label"); SPECS+=("$label=$dir@$method@$ntok$ov")
done
if [[ $NEED_OVERLAY -eq 1 ]]; then
    [[ -f "$VLLM_OVERLAY/vllm/model_executor/models/grug_dspark.py" ]] || die "+overlay drafters need the vLLM overlay, but $VLLM_OVERLAY has no vllm/model_executor/models/grug_dspark.py (set VLLM_OVERLAY)"
fi
# ---- check the benchmarks
for b in $BENCHES; do
    bench_known=0; for k in $BENCHES_KNOWN; do [[ "$k" == "$b" ]] && bench_known=1; done
    [[ $bench_known -eq 1 ]] || die "unknown benchmark '$b' (known: $BENCHES_KNOWN)"
    while read -r f; do [[ -f "$f" ]] || die "benchmark $b: missing $f"; done < <(bench_files "$b")
done
case $JOBS in 1|drafter|pair) ;; *) die "JOBS must be 1, drafter or pair (got '$JOBS')" ;; esac

# ---- group the pairs into jobs. A pair is "<bench>|<label>=<dir>@<method>@<N>[+overlay]"
declare -a GROUP_PAIRS=() GROUP_NAMES=()
if [[ $JOBS == 1 ]]; then
    p=""; for b in $BENCHES; do for s in "${SPECS[@]}"; do p+="${p:+;}$b|$s"; done; done
    GROUP_PAIRS+=("$p"); GROUP_NAMES+=("evm-all")
elif [[ $JOBS == drafter ]]; then
    for i in "${!SPECS[@]}"; do p=""; for b in $BENCHES; do p+="${p:+;}$b|${SPECS[$i]}"; done; GROUP_PAIRS+=("$p"); GROUP_NAMES+=("evm-${LABELS[$i]:0:24}"); done
else
    for b in $BENCHES; do for i in "${!SPECS[@]}"; do GROUP_PAIRS+=("$b|${SPECS[$i]}"); GROUP_NAMES+=("evm-$b-${LABELS[$i]:0:20}"); done; done
fi
auto_time() {  # one hour per pair plus 15 minutes, at most 12 h
    local n=$1 m=$(( $1 * 60 + 15 )); (( m > 720 )) && { echo "WARNING: $n pairs need over 12 h in one job; use --per-drafter or --per-pair" >&2; m=720; }
    printf '%02d:%02d:00' $((m / 60)) $((m % 60))
}
res=$(gb_reservation_args)
if [[ -z "$res" && "${ALLOW_NO_RESERVATION:-0}" != 1 ]]; then die "no reservation would be applied (CGAI-gb missing); ALLOW_NO_RESERVATION=1 overrides"; fi
[[ $DRY -eq 1 ]] || { command -v sbatch >/dev/null || die "sbatch not found (login node?)"; mkdir -p "$ROOT/logs/slurm" "$OUT_BASE"; }
echo "target $MODEL | benchmarks: $BENCHES | drafters: ${LABELS[*]} | jobs: ${#GROUP_PAIRS[@]} ($( [[ $JOBS == 1 ]] && echo serial || echo "per-$JOBS" )) | out $OUT_BASE"
for g in "${!GROUP_PAIRS[@]}"; do
    pairs=${GROUP_PAIRS[$g]}; n=$(awk -F';' '{print NF}' <<< "$pairs"); t=${TIME:-$(auto_time "$n")}
    cmd=(sbatch --parsable -p gb -A "${ACCOUNT:-ASC26088}" -t "$t" -J "${GROUP_NAMES[$g]}" --export=ALL); [[ -z "$res" ]] || cmd+=("$res"); cmd+=("$HERE/eval_matrix.sbatch")
    echo "== ${GROUP_NAMES[$g]}: $n pair(s), time $t"; tr ';' '\n' <<< "$pairs" | sed 's/^/     /'
    if [[ $DRY -eq 1 ]]; then echo "   ${cmd[*]}"; continue; fi
    out=$( export EVAL_PAIRS="$pairs" EVAL_OUT_BASE="$OUT_BASE" MODEL VLLM_OVERLAY; "${cmd[@]}" 2>&1 ); id=$(grep -E '^[0-9]+$' <<< "$out" | tail -1)
    [[ "$id" =~ ^[0-9]+$ ]] || die "sbatch failed for ${GROUP_NAMES[$g]}: $(tail -3 <<< "$out")"; echo "   job $id"
done
