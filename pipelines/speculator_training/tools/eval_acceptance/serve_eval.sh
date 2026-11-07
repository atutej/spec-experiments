#!/bin/bash
# Serve Grug-67B-A2B on this node (DP4 x EP4, 4 GPUs) with and/or without a DSpark drafter, send a set of prompts, and measure acceptance length and output
# tokens per second (run_eval.py). One server per arm, started and stopped in turn. Run on a 4-GPU gb node.
#
#   PLAIN=1 ARMS="a=<drafter dir> b=<drafter dir>" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"' bash serve_eval.sh
#
# env: ARMS       space-separated label=<drafter dir> (a drafter arm each); PLAIN=1 adds a no-draft arm first (the throughput baseline)
#      SPEC_EXTRA extra fields for the speculative config, as JSON text starting with a comma. Not stored in the drafter checkpoint: vLLM reads only the method,
#                 the token count, the tapped layers and the vocabulary map from it, so the sampling modes must be set here (or in the RL engine's speculative_config).
#                 Recommended for sampled decoding (T=1): ', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"'
#      PROMPTS_DIR (runs/val_prompts_128, from build_val_prompts.py; or runs/r2egym_prompts_128) MAX_PROMPTS (all) MAX_TOKENS (4096) TEMPERATURE (1.0)
#      SUBSETS (first_turn; "first_turn later_turn" for prompt dirs from build_val_prompts.py --turns random)
#      MAX_MODEL_LEN (32768; the RL runs use a 102,400-token window with up to 49,152 new tokens per turn)
#      OUT (runs/rl_speculation/serve_eval) NUM_SPEC_TOKENS (8, the drafter's block size)
#      SERVER_EXTRA extra `vllm serve` flags for every server, e.g. "--max-cudagraph-capture-size 2048"; PROMPT_SETS, GPU_LOG: see run_arm
#      An arm may carry its own method and token count: ARMS="eagle3=<dir>@eagle3@3 n4=<dir>@dspark@4" (default: dspark with NUM_SPEC_TOKENS and SPEC_EXTRA).
#      CONCURRENCIES="4 16 64 256" (default 16) sweeps the client concurrency on each server; PROMPTS_MULT / PROMPTS_MIN size the prompt set per level (see run_arm).
# Results: $OUT/<label>/acceptance.csv and results.json (acceptance_length, output_tokens_per_s, per-position acceptance), completions_first_turn.jsonl
source ~/.bashrc 2>/dev/null || true
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator
set +u; source $ROOT/spec-experiments/env.sh; set -u
use_env() { set +u; conda activate "$1"; set -u; }
HERE=$ROOT/spec-experiments/pipelines/speculator_training/tools/eval_acceptance
MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21}
PROMPTS_DIR=${PROMPTS_DIR:-$ROOT/runs/val_prompts_128}; MAX_TOKENS=${MAX_TOKENS:-4096}; TEMPERATURE=${TEMPERATURE:-1.0}; NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-8}
OUT=${OUT:-$ROOT/runs/rl_speculation/serve_eval}; PORT=${PORT:-8000}; mkdir -p "$OUT"
EXTRA=(); [[ -z "${MAX_PROMPTS:-}" ]] || EXTRA+=(--max-prompts "$MAX_PROMPTS")
run_arm() {  # run_arm <label> <drafter dir, empty for the no-draft arm>
    local name=$1 draft=$2 d="$OUT/$1"; mkdir -p "$d"
    local method=dspark ntok=$NUM_SPEC_TOKENS extra=${SPEC_EXTRA:-}
    if [[ "$draft" == *@* ]]; then   # <drafter dir>@<method>@<speculative tokens>, e.g. /path/to/eagle3-draft@eagle3@3 (SPEC_EXTRA is DSpark-only: kept for dspark arms, dropped for the others)
        IFS=@ read -r draft method ntok <<< "$draft"; [[ "$method" == dspark ]] || extra=""
    fi
    use_env vllm
    local spec_args=() client_args=()
    if [[ -n "$draft" ]]; then
        spec_args=(--speculative-config "$(printf '{"model": "%s", "num_speculative_tokens": %s, "method": "%s"%s}' "$draft" "$ntok" "$method" "$extra")")
        client_args=(--checkpoint "$draft")
    else client_args=(--no-spec); fi
    setsid vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --max-model-len ${MAX_MODEL_LEN:-32768} \
        --gpu-memory-utilization 0.9 ${SERVER_EXTRA:-} "${spec_args[@]}" > "$d/server.log" 2>&1 &
    local PGID=$!
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 "$PGID" 2>/dev/null || { echo "[$name] server exited; tail of $d/server.log:" >&2; tail -30 "$d/server.log" >&2; return 1; }; sleep 10; done
    echo "[$name] ready after ${SECONDS}s"
    use_env speculators
    # one pass at concurrency 16 by default; CONCURRENCIES="4 16 64" runs each level against this one server, results in $d/c<level>/
    # (PROMPTS_MULT x level prompts, at least PROMPTS_MIN, so a higher level still gets enough requests; without them MAX_PROMPTS applies)
    # PROMPT_SETS="name=<prompts dir>:<max tokens>:<subsets, comma separated>[:<max prompts>] ..." serves the arm ONCE and runs every set at every level (results in $d/<name>/c<level>/); unset: PROMPTS_DIR as before
    local sets=("${PROMPT_SETS:-}")
    [[ -n "${PROMPT_SETS:-}" ]] && read -ra sets <<< "$PROMPT_SETS"
    local GPU_PID=""
    if [[ -n "${GPU_LOG:-}" ]]; then nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l "${GPU_LOG_SECS:-5}" > "$d/gpu_util.csv" 2>/dev/null & GPU_PID=$!; fi
    for set in "${sets[@]}"; do
    local sdir="$PROMPTS_DIR" smax="$MAX_TOKENS" ssub="${SUBSETS:-first_turn}" sname="" smaxp=""
    if [[ -n "$set" ]]; then sname=${set%%=*}; IFS=: read -r sdir smax ssub smaxp <<< "${set#*=}"; ssub=${ssub//,/ }; fi   # optional 4th field: only the first N prompts of each subset
    for c in ${CONCURRENCIES:-16}; do
        local od="$d" pe=("${EXTRA[@]}")
        if [[ -n "${CONCURRENCIES:-}" ]]; then od="$d/c$c"; fi
        if [[ -n "$sname" ]]; then od="$d/$sname/c$c"; fi
        mkdir -p "$od"
        if [[ -n "${PROMPTS_MULT:-}" ]]; then local n=$(( c * PROMPTS_MULT )); (( n < ${PROMPTS_MIN:-0} )) && n=${PROMPTS_MIN}; pe=(--max-prompts "$n"); fi
        [[ -z "$smaxp" ]] || pe=(--max-prompts "$smaxp")
        python "$HERE/run_eval.py" --target "http://127.0.0.1:$PORT" --model "$MODEL" --prompts-dir "$sdir" --subsets $ssub --output-dir "$od" \
            --max-tokens "$smax" --temperature "$TEMPERATURE" --top-p 1.0 --top-k -1 --concurrency "$c" "${client_args[@]}" "${pe[@]}"
    done
    done
    [[ -z "$GPU_PID" ]] || kill "$GPU_PID" 2>/dev/null
    kill -TERM -- "-$PGID" 2>/dev/null; sleep 20; kill -KILL -- "-$PGID" 2>/dev/null; sleep 10
}
[[ -z "${PLAIN:-}" ]] || run_arm plain ""
for arm in ${ARMS:-}; do run_arm "${arm%%=*}" "${arm#*=}"; done
echo "== all arms done; results in $OUT"
