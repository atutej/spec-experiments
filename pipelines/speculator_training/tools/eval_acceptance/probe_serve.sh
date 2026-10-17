#!/bin/bash
# For each drafter arm: serve Grug-67B-A2B with it (DP4 x EP4, probabilistic drafting + standard verification, the sampling the trainer's accept_len assumes) and run
# probe_random_anchor.py. The FIRST arm generates the target's own replies into $OUT/seqs.json; later arms probe the same text. Run on a 4-GPU gb node.
#   DATASET=<trainer-format dataset dir> probes that dataset's held-out rows instead (no generation).
#   ARMS="targetgen=<drafter dir> itw=<drafter dir>" PROMPTS_DIR=runs/val_prompts_1024 OUT=<dir> bash probe_serve.sh
source ~/.bashrc 2>/dev/null || true
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator
set +u; source $ROOT/spec-experiments/env.sh; set -u
use_env() { set +u; conda activate "$1"; set -u; }
HERE=$ROOT/spec-experiments/pipelines/speculator_training/tools/eval_acceptance
MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21}; PORT=${PORT:-8000}
PROMPTS_DIR=${PROMPTS_DIR:-$ROOT/runs/val_prompts_1024}; OUT=${OUT:?set OUT}; mkdir -p "$OUT"
if [[ -n "${DATASET:-}" ]]; then SRC_ARGS=(--dataset "$DATASET"); else SRC_ARGS=(--prompts-dir "$PROMPTS_DIR" --seqs-file "$OUT/seqs.json"); fi
for arm in $ARMS; do
    name=${arm%%=*}; draft=${arm#*=}; d="$OUT/$name"; mkdir -p "$d"
    use_env vllm
    SPEC=$(printf '{"model": "%s", "num_speculative_tokens": 8, "method": "dspark", "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}' "$draft")
    setsid vllm serve "$MODEL" --port "$PORT" --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel --max-model-len 32768 \
        --gpu-memory-utilization 0.9 --speculative-config "$SPEC" > "$d/server.log" 2>&1 &
    PGID=$!
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        kill -0 "$PGID" 2>/dev/null || { echo "[$name] server exited" >&2; tail -20 "$d/server.log" >&2; break; }; sleep 10; done
    echo "[$name] ready after ${SECONDS}s"
    use_env speculators
    python "$HERE/probe_random_anchor.py" --target "http://127.0.0.1:$PORT" --model "$MODEL" --out "$d/probe.json" ${ROWS:+--rows $ROWS} ${PER_ROW:+--per-row $PER_ROW} \
        "${SRC_ARGS[@]}"
    kill -TERM -- "-$PGID" 2>/dev/null; sleep 20; kill -KILL -- "-$PGID" 2>/dev/null; sleep 10
done
echo ALLDONE
