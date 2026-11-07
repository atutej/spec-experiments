#!/bin/bash
# SPEED-Bench (qualitative split) with NVIDIA's AIPerf against a vLLM server of Grug-67B-A2B + a DSpark drafter. One server per arm, started and stopped in turn (same launch
# as serve_eval.sh: DP4 x EP4 on a 4-GPU gb node); the client is `aiperf profile` once per category, the acceptance report is `aiperf speed-bench-report`.
#
#   OUT=<NEW dir> ARMS="a=<drafter dir>@dspark@8 b=..." SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_speedbench.sh
#
# env: MODEL (the target; default Dr. Doom), OUT (its arm directories must not exist), ARMS (label=<drafter dir>[@method@tokens]), SPEC_EXTRA (as in serve_eval.sh), SERVER_EXTRA (extra `vllm serve` flags),
#      CONC (client concurrency, 32: the SPEED-Bench paper's batch size for the qualitative split), OSL (max new tokens, 4096), TEMPERATURE (1.0), MAX_MODEL_LEN (32768),
#      DATA (SPEED-Bench JSONL, one row per sample: question_id, category, messages; built from the specdec_bench `prepare_data.py` output), CATEGORIES (the 11 qualitative categories).
# DISPATCH=sessions (default, from 2026-11-03): `--num-conversations <samples in the category>`, one conversation per sample, so every turn of every sample is sent exactly once (the follow-up turns carry the
# live reply); verified against a fake server (reasoning: 80/80/28/2 requests of turn depth 1/2/3/4) and, after each category, against the per-request export (coverage_<category>.json).
# DISPATCH=turns is the first version (`--request-count` = the category's number of TURNS): it repeated about 4% of the requests and never sent the same number of follow-up turns (results of 2026-11-02). Results: $OUT/<arm>/speed_bench_<category>/ (AIPerf artifacts), $OUT/<arm>/speed_bench_report.csv (acceptance length per category and overall), $OUT/<arm>/server.log
source ~/.bashrc 2>/dev/null || true
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator; SB=/scratch/09749/atutej/marin_speed_bench
set +u; source $ROOT/spec-experiments/env.sh; set -u
use_env() { set +u; conda activate "$1"; set -u; }
MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom}
OUT=${OUT:?set OUT}   # arm directories under it must not exist (checked per arm in run_arm), so several drafters can share one OUT
mkdir -p "$OUT"
PORT=${PORT:-8000}; CONC=${CONC:-32}; OSL=${OSL:-4096}; TEMPERATURE=${TEMPERATURE:-1.0}; NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-8}
DATA=${DATA:-$SB/aiperf_data/qualitative.jsonl}; AIPERF=$SB/aiperf_venv/bin/aiperf
DISPATCH=${DISPATCH:-sessions}
CATEGORIES=${CATEGORIES:-"coding humanities math multilingual qa rag reasoning roleplay stem summarization writing"}
[[ -f "$DATA" && -x "$AIPERF" ]] || { echo "ERROR: missing $DATA or $AIPERF" >&2; exit 1; }
turns_of() { python3 - "$DATA" "$1" "$2" <<'PY'
import json, sys
rows = [r for r in map(json.loads, open(sys.argv[1])) if r["category"] == sys.argv[2]]
print(len(rows) if sys.argv[3] == "sessions" else sum(len(r["messages"]) for r in rows))
PY
}
coverage() {  # coverage <data file> <category> <artifact dir> -> prints and writes coverage.json: every (sample, turn) exactly once?
    python3 - "$1" "$2" "$3" <<'PY'
import json, sys, collections
data, cat, art = sys.argv[1:4]
want = {(r["question_id"], t) for r in map(json.loads, open(data)) if r["category"] == cat for t in range(len(r["messages"]))}
try:
    recs = [json.loads(l)["metadata"] for l in open(art + "/profile_export.jsonl")]
except OSError:
    print(f"coverage {cat}: NO per-request export"); sys.exit(0)
got = collections.Counter((m["conversation_id"], m["turn_index"]) for m in recs)
out = {"category": cat, "expected_requests": len(want), "sent": len(recs), "repeated": sum(v - 1 for v in got.values() if v > 1), "missing": len(want - set(got)), "unexpected": len(set(got) - want)}
json.dump(out, open(art + "/coverage.json", "w"))
print("coverage", cat, out, "OK" if out["repeated"] == out["missing"] == out["unexpected"] == 0 and out["sent"] == out["expected_requests"] else "MISMATCH")
PY
}
run_arm() {  # run_arm <label> <drafter dir@method@tokens>
    local name=$1 draft=$2 d="$OUT/$1"
    [[ -e "$d" ]] && { echo "ERROR: $d exists; old results are never overwritten" >&2; return 1; }
    mkdir -p "$d"
    local method=dspark ntok=$NUM_SPEC_TOKENS extra=${SPEC_EXTRA:-}
    if [[ "$draft" == *@* ]]; then IFS=@ read -r draft method ntok <<< "$draft"; [[ "$method" == dspark ]] || extra=""; fi
    use_env vllm
    local spec_args=(--speculative-config "$(printf '{"model": "%s", "num_speculative_tokens": %s, "method": "%s"%s}' "$draft" "$ntok" "$method" "$extra")")
    setsid vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --max-model-len ${MAX_MODEL_LEN:-32768} \
        --gpu-memory-utilization 0.9 ${SERVER_EXTRA:-} "${spec_args[@]}" > "$d/server.log" 2>&1 &
    local PGID=$!
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 "$PGID" 2>/dev/null || { echo "[$name] server exited; tail of $d/server.log:" >&2; tail -30 "$d/server.log" >&2; return 1; }; sleep 10; done
    echo "[$name] ready after ${SECONDS}s"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1       # the client only needs the tokenizer, which is in the cache
    for cat in $CATEGORIES; do
        local n; n=$(turns_of "$cat" "$DISPATCH")
        local disp=(--request-count "$n"); [[ "$DISPATCH" == sessions ]] && disp=(--num-conversations "$n")
        echo "[$name] $cat: $n $DISPATCH (${disp[*]}) at concurrency $CONC"
        "$AIPERF" profile --model "$MODEL" --tokenizer "$MODEL" --endpoint-type chat --streaming --url "localhost:$PORT" \
            --custom-dataset-type "speed_bench_$cat" --input-file "$DATA" --server-metrics "http://localhost:$PORT/metrics" \
            --osl "$OSL" --extra-inputs "temperature:$TEMPERATURE" top_p:1.0 top_k:-1 --concurrency "$CONC" "${disp[@]}" \
            --no-gpu-telemetry --ui-type none --output-artifact-dir "$d/speed_bench_$cat" > "$d/aiperf_$cat.log" 2>&1 \
            || echo "[$name] WARNING: aiperf failed for $cat (see $d/aiperf_$cat.log)" >&2
        coverage "$DATA" "$cat" "$d/speed_bench_$cat" | sed "s/^/[$name] /"
    done
    "$AIPERF" speed-bench-report "$d" --format both --output "$d/speed_bench_report.csv" 2>&1 | tail -12
    kill -TERM -- "-$PGID" 2>/dev/null; sleep 20; kill -KILL -- "-$PGID" 2>/dev/null; sleep 10
}
for arm in ${ARMS:?set ARMS}; do run_arm "${arm%%=*}" "${arm#*=}"; done
echo "== all arms done; results in $OUT"
