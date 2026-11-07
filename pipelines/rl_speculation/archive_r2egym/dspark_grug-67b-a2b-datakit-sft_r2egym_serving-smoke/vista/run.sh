#!/bin/bash
# Serving smoke test of the DSpark drafter on the Grug Datakit-SFT target, one 4-GPU gb node, no RL.
#
#   bash vista/run.sh [stage ...]       stages: prompts plain dspark   (default: prompts plain dspark)
#
#   prompts  CPU   R2E-Gym instructions -> $PROMPTS_DIR (skipped if present)
#   plain    4 GPU serve the target DP x EP, send the prompts, record tok/s            -> $WORK_DIR/plain/
#   dspark   4 GPU the same server with --speculative-config {method: dspark}; accept length from vLLM's counters, tok/s -> $WORK_DIR/dspark/
# Same prompts, sampling and per-request seeds in both arms; compare completions_*.jsonl and results.json (see NOTES.md for what to read).
set -euo pipefail
set -E; trap 'echo "run.sh: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
use_env() { set +u; conda activate "$1"; set -u; }
source "$EXP_DIR/settings.sh"
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(prompts plain dspark)
for s in "${STAGES[@]}"; do case "$s" in
    prompts) ;;
    plain|dspark) [[ "$NUM_GPUS" -ge "$SERVE_DP" ]] || { echo "stage '$s' needs $SERVE_DP GPUs (NODE_KIND=$NODE_KIND, NUM_GPUS=$NUM_GPUS)" >&2; exit 1; } ;;
    *) echo "unknown stage '$s' (prompts plain dspark)" >&2; exit 2 ;;
esac; done
mkdir -p "$WORK_DIR"; cd "$WORK_DIR"
echo "node=$(hostname) kind=$NODE_KIND gpus=$NUM_GPUS stages=${STAGES[*]} DRAFT=$DRAFT PROMPTS_DIR=$PROMPTS_DIR"

stage_prompts() {
    [[ -f "$PROMPTS_DIR/first_turn.jsonl" ]] && { echo "prompts present: $PROMPTS_DIR"; return; }
    use_env speculators
    python "$REPO_DIR/pipelines/rl_speculation/tools/build_r2egym_prompts.py" --tasks-dir "$TASKS_DIR" --model "$MODEL" \
        --output-dir "$PROMPTS_DIR" --num "$MAX_PROMPTS"
}

SERVER_PGID=""
stop_server() {  # the server runs in its own process group; stop all of it, also on failure (EXIT trap)
    [[ -n "$SERVER_PGID" ]] || return 0
    kill -TERM -- "-$SERVER_PGID" 2>/dev/null || true
    for _ in $(seq 30); do kill -0 -- "-$SERVER_PGID" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-$SERVER_PGID" 2>/dev/null || true
    SERVER_PGID=""; sleep 10
}
trap stop_server EXIT

serve_and_measure() {  # serve_and_measure <arm> [speculative-config json]
    local arm="$1" spec="${2:-}" out="$WORK_DIR/$1"
    mkdir -p "$out"
    [[ -f "$PROMPTS_DIR/first_turn.jsonl" ]] || { echo "no prompts at $PROMPTS_DIR: run the 'prompts' stage" >&2; exit 1; }
    use_env vllm
    local spec_args=(); [[ -z "$spec" ]] || spec_args=(--speculative-config "$spec")
    setsid vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 --data-parallel-size "$SERVE_DP" --enable-expert-parallel \
        --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" "${spec_args[@]}" > "$out/server.log" 2>&1 &
    SERVER_PGID=$!
    echo "[$arm] serving (log $out/server.log)"
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 "$SERVER_PGID" 2>/dev/null || { echo "[$arm] server exited; tail of $out/server.log:" >&2; tail -30 "$out/server.log" >&2; exit 1; }
        sleep 10
    done
    echo "[$arm] ready after ${SECONDS}s"
    use_env speculators
    local client_args=(); [[ -n "$spec" ]] || client_args=(--no-spec)   # the control arm has no draft counters to read
    # Greedy-free settings of the RL rollouts (T=1, no top-p/top-k), as in Luke's runs; seeds are per request.
    python "$REPO_DIR/pipelines/speculator_training/tools/eval_acceptance/run_eval.py" --target "http://127.0.0.1:$PORT" --model "$MODEL" \
        --prompts-dir "$PROMPTS_DIR" --subsets first_turn --output-dir "$out" --max-prompts "$MAX_PROMPTS" --max-tokens "$MAX_TOKENS" \
        --temperature 1.0 --top-p 1.0 --top-k -1 --concurrency "$CONCURRENCY" --checkpoint "${spec:+$DRAFT}" "${client_args[@]}"
    echo "[$arm] done; results in $out"
    stop_server
}

stage_plain()  { serve_and_measure plain; }
stage_dspark() { serve_and_measure dspark "{\"model\": \"$DRAFT\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"method\": \"dspark\"}"; }
for s in "${STAGES[@]}"; do "stage_$s"; done
echo "Stages finished: ${STAGES[*]}"
