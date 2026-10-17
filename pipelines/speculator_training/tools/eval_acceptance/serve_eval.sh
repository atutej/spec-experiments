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
    use_env vllm
    local spec_args=() client_args=()
    if [[ -n "$draft" ]]; then
        spec_args=(--speculative-config "$(printf '{"model": "%s", "num_speculative_tokens": %s, "method": "dspark"%s}' "$draft" "$NUM_SPEC_TOKENS" "${SPEC_EXTRA:-}")")
        client_args=(--checkpoint "$draft")
    else client_args=(--no-spec); fi
    setsid vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --max-model-len ${MAX_MODEL_LEN:-32768} \
        --gpu-memory-utilization 0.9 "${spec_args[@]}" > "$d/server.log" 2>&1 &
    local PGID=$!
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 "$PGID" 2>/dev/null || { echo "[$name] server exited; tail of $d/server.log:" >&2; tail -30 "$d/server.log" >&2; return 1; }; sleep 10; done
    echo "[$name] ready after ${SECONDS}s"
    use_env speculators
    python "$HERE/run_eval.py" --target "http://127.0.0.1:$PORT" --model "$MODEL" --prompts-dir "$PROMPTS_DIR" --subsets ${SUBSETS:-first_turn} --output-dir "$d" \
        --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --top-p 1.0 --top-k -1 --concurrency 16 "${client_args[@]}" "${EXTRA[@]}"
    kill -TERM -- "-$PGID" 2>/dev/null; sleep 20; kill -KILL -- "-$PGID" 2>/dev/null; sleep 10
}
[[ -z "${PLAIN:-}" ]] || run_arm plain ""
for arm in ${ARMS:-}; do run_arm "${arm%%=*}" "${arm#*=}"; done
echo "== all arms done; results in $OUT"
