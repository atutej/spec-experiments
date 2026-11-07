#!/bin/bash
# Worker of eval_matrix.sh: one gb node runs the (benchmark x drafter) pairs of EVAL_PAIRS one after the other. Submit through eval_matrix.sh, not by hand; the settings travel in the environment
# (EVAL_PAIRS "<bench>|<label>=<dir>@<method>@<N>[+overlay];...", EVAL_OUT_BASE, MODEL, VLLM_OVERLAY).
# A failed pair does not stop the rest; finished pairs (result files present) are skipped, so a rerun with the same out base only fills in what is missing. A summary table closes the log.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator
: "${EVAL_PAIRS:?set EVAL_PAIRS (use eval_matrix.sh)}" "${EVAL_OUT_BASE:?set EVAL_OUT_BASE}" "${MODEL:?set MODEL}"
set +u; source ~/.bashrc 2>/dev/null || true   # ~/.bashrc uses unset variables: under `set -u` it aborts the job silently (empty log)
source $ROOT/spec-experiments/env.sh; set -u
source $ROOT/spec-experiments/pipelines/speculator_training/tools/eval_acceptance/eval_benchmarks.sh
# The target's weights must be in the HF cache (a no-op when they are; resumes a partial download otherwise).
set +u; conda activate speculators; set -u
python - "$MODEL" <<'PY' || exit 1
import os, sys
from huggingface_hub import snapshot_download
print("target weights:", snapshot_download(sys.argv[1], ignore_patterns=["configs/*", "evidence/*", "source/*", "*.png", "*.svg"], max_workers=8, token=os.environ.get("HUG_TOKEN") or None), flush=True)
PY
export TEMPERATURE=1.0 SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"'
declare -a SUMMARY=(); FAILED=0
IFS=';' read -ra PAIRS <<< "$EVAL_PAIRS"
echo "== ${#PAIRS[@]} pair(s) -> $EVAL_OUT_BASE (target $MODEL)"
for pair in "${PAIRS[@]}"; do
    bench=${pair%%|*}; spec=${pair#*|}; label=${spec%%=*}; rest=${spec#*=}; overlay=0
    [[ "$rest" == *+overlay ]] && { overlay=1; rest=${rest%+overlay}; }
    result="$EVAL_OUT_BASE/$bench/$label"
    echo; echo "== $(date +%T) $bench | $label | $rest$([[ $overlay == 1 ]] && echo ' | overlay')"
    if [[ -e "$result" ]]; then
        if bench_done "$bench" "$result"; then echo "   already done ($result); skipped"; SUMMARY+=("skipped  $bench  $label"); continue; fi
        echo "   $result exists but has no result files; NOT overwritten: remove or rename it to rerun this pair"; SUMMARY+=("blocked  $bench  $label  (incomplete $result)"); FAILED=1; continue
    fi
    (
        bench_env "$bench" || exit 2
        export OUT="$EVAL_OUT_BASE/$bench" ARMS="$label=$rest"
        [[ $overlay == 1 ]] && export PYTHONPATH="${VLLM_OVERLAY:?set VLLM_OVERLAY}${PYTHONPATH:+:$PYTHONPATH}"
        bash "$RUNNER"
    ); rc=$?
    if [[ $rc -eq 0 ]] && bench_done "$bench" "$result"; then SUMMARY+=("ok       $bench  $label")
    else echo "   FAILED (exit $rc, results $([[ -e $result ]] && echo present || echo missing)): $bench $label"; SUMMARY+=("FAILED   $bench  $label  (exit $rc; see $result)"); FAILED=1
        pkill -f "vllm serve" 2>/dev/null; sleep 20   # a server left behind by the failed pair would hold the port and the GPUs
    fi
done
echo; echo "== summary ($EVAL_OUT_BASE)"; printf '   %s\n' "${SUMMARY[@]}"
exit $FAILED
