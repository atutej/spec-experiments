#!/bin/bash
# Acceptance length of one DSpark drafter on Dr. Doom over Marin eval-policy tasks, run with the policy's own Evalchemy client (pinned commits). Serve the target with
# the drafter on this node once, then for each seed and task: scrape the server's /metrics, run the policy client for that task, scrape again. Only the
# speculative-decoding counters are kept; task scores and sample files are never read or printed and the Evalchemy output of each task is deleted right after it
# (Marin eval policy #10197: nobody looks at scores before the model is frozen). Submit through submit_evalchemy_al.sh, not by hand.
#
# env (exported by the submit script): ROOT (workspace), ARM (label), OUT (fresh result dir), DRAFTER (checkpoint dir or Hugging Face repo id; empty = plain decoding),
#   TASKS (space-separated policy task files, default below), SEEDS (default "policy 43"), TEMPERATURE (default empty = sampling exactly as the policy sets it; a number overrides every task),
#   MODEL, GPUS (default 4), VLLM_ENV (default vllm_fork), EVAL_ENV_DIR (default $ROOT/envs/evalchemy_pin), POLICY_DIR (policy evalchemy-configs; fetched if absent),
#   SERVER_EXTRA, VLLM_OVERLAY, LIMIT (items per task; smoke tests only), KEEP_OUTPUT=1 (keep each task's Evalchemy output and log under $OUT/seed*/<task>/evalchemy, unread; default: delete)
#SBATCH -N 1
#SBATCH --exclusive
set -uo pipefail
: "${ROOT:?}" "${ARM:?}" "${OUT:?}"
DRAFTER=${DRAFTER:-}; GPUS=${GPUS:-4}; VLLM_ENV=${VLLM_ENV:-vllm_fork}; SERVER_EXTRA=${SERVER_EXTRA:-}; VLLM_OVERLAY=${VLLM_OVERLAY:-}
MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom}
TASKS=${TASKS:-"gsm8k-0shot humanevalplus mbppplus ifbench cruxeval triviaqa math500 aime24 gpqa-diamond"}
SEEDS=${SEEDS:-"policy 43"}   # "policy" = the task file's own seed (none if it sets none); a number overrides it
TEMPERATURE=${TEMPERATURE:-}
EVAL_ENV_DIR=${EVAL_ENV_DIR:-$ROOT/envs/evalchemy_pin}
POLICY_SHA=75a5690b0ff2e1510d73e576649e3e059f06da9a
PIN_DIR=${PIN_DIR:-$ROOT/evalchemy_pin_20261114/marin_policy_full}
TOOLS=$ROOT/spec-experiments/pipelines/speculator_training/tools
[[ ! -e "$OUT" ]] || { echo "$OUT exists: results are never overwritten"; exit 1; }
mkdir -p "$OUT"
set +u; source ~/.bashrc 2>/dev/null || true; source "$ROOT/spec-experiments/env.sh" || exit 1; set -u
use_env() { set +u; conda activate "$1"; set -u; }
# A Hugging Face token (from ~/.bashrc) is kept out of the vLLM server and given only to the Evalchemy client of the tasks that need a gated dataset (HF_TOKEN_TASKS,
# default: gpqa-diamond). It is never printed or written to a file.
HFT=${HF_TOKEN:-}; unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
[[ -x $EVAL_ENV_DIR/bin/python ]] || { echo "no Evalchemy env at $EVAL_ENV_DIR (build_evalchemy_env.sbatch)"; exit 1; }
if [[ ! -d $PIN_DIR/lib ]]; then
    mkdir -p "$PIN_DIR" && curl -sfL --retry 3 "https://codeload.github.com/marin-community/marin/tar.gz/$POLICY_SHA" | tar -xz -C "$PIN_DIR" --strip-components=1 \
        --wildcards "*/lib/marin/src/marin/*" "*/experiments/evaluation/*" || { echo "could not fetch the policy pin"; exit 1; }
fi
POLICY_DIR=${POLICY_DIR:-$PIN_DIR/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled/evalchemy-configs}
CLIENT=$PIN_DIR/lib/marin/src/marin/evaluation/evalchemy/client.py
[[ -f $CLIENT && -d $POLICY_DIR ]] || { echo "policy files missing under $PIN_DIR"; exit 1; }

# Serve the target (flags as in run_agentic_eval.sbatch: DP x EP over the node's GPUs, the policy's context and batching settings for the Dr. Doom config).
use_env "$VLLM_ENV"
[[ -z "$VLLM_OVERLAY" ]] || export PYTHONPATH="$VLLM_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
spec=()
[[ -z "$DRAFTER" ]] || spec=(--speculative-config "{\"model\": \"$DRAFTER\", \"num_speculative_tokens\": 8, \"method\": \"dspark\", \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"standard\"}")
setsid vllm serve "$MODEL" --port 8000 --tensor-parallel-size 1 --data-parallel-size "$GPUS" --enable-expert-parallel --max-model-len 73728 \
    --max-num-batched-tokens 7168 --max-num-seqs 32 --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.9 \
    $SERVER_EXTRA "${spec[@]}" > "$OUT/server.log" 2>&1 &
PGID=$!
cleanup() { kill -TERM -- "-$PGID" 2>/dev/null; sleep 20; kill -KILL -- "-$PGID" 2>/dev/null; }
trap cleanup EXIT
until curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; do
    kill -0 "$PGID" 2>/dev/null || { echo "server exited; tail of $OUT/server.log:"; tail -30 "$OUT/server.log"; exit 1; }
    (( SECONDS < ${READY_TIMEOUT:-1800} )) || { echo "server not ready after ${SECONDS}s (hung at startup?); tail of $OUT/server.log:"; tail -5 "$OUT/server.log" | cut -c1-200; exit 1; }
    sleep 10
done
echo "[$ARM] server ready after ${SECONDS}s"

# Tasks. Evalchemy output (scores, samples) goes to a scratch dir that is deleted after each task; its log is reduced to the exit code and, on failure, error lines.
EP="$EVAL_ENV_DIR/bin/python"
for SEED in $SEEDS; do
  for T in $TASKS; do
    R=$OUT/seed$SEED/$T; W=$OUT/.work_${SEED}_$T
    [[ "${KEEP_OUTPUT:-0}" == 1 ]] && W=$R/evalchemy   # kept output is written where it stays (its manifests record absolute paths)
    mkdir -p "$R" "$W"
    targs=(--policy-dir "$POLICY_DIR" --task "$T" --model "$MODEL" --out-path "$W/out")
    [[ "$SEED" == policy ]] || targs+=(--seed "$SEED")
    [[ -z "$TEMPERATURE" ]] || targs+=(--temperature "$TEMPERATURE")
    [[ -z "${LIMIT:-}" ]] || targs+=(--limit "$LIMIT")
    EVALCHEMY_CLIENT_CONFIG=$("$EP" -I "$TOOLS/evalchemy_al/make_client_config.py" "${targs[@]}") || { echo "config for $T failed"; echo config_failed > "$R/status"; continue; }
    export EVALCHEMY_CLIENT_CONFIG
    curl -s http://127.0.0.1:8000/metrics > "$R/metrics_before.prom"
    echo "[$ARM] seed $SEED task $T start $(date +%T)"
    if [[ -n "$HFT" && " ${HF_TOKEN_TASKS:-gpqa-diamond} " == *" $T "* ]]; then
        ( cd "$W" && HF_TOKEN="$HFT" "$EP" -I "$CLIENT" > "$W/run.log" 2>&1 ); rc=$?
    else
        ( cd "$W" && "$EP" -I "$CLIENT" > "$W/run.log" 2>&1 ); rc=$?
    fi
    curl -s http://127.0.0.1:8000/metrics > "$R/metrics_after.prom"
    echo "exit=$rc" > "$R/status"
    [[ $rc == 0 ]] || grep -E 'Traceback|Error|Exception' "$W/run.log" | cut -c1-300 | head -15 > "$R/errors.txt"
    python3 "$TOOLS/agentic_eval/spec_metrics.py" "$R/metrics_before.prom" "$R/metrics_after.prom" --out "$R/spec_metrics.json" > /dev/null
    if [[ "${KEEP_OUTPUT:-0}" == 1 ]]; then   # keep Evalchemy's output (scores, samples) and log unread, for later; owner-only
        chmod -R go-rwx "$W"
    else
        rm -rf "$W"
    fi
    echo "[$ARM] seed $SEED task $T exit $rc at $(date +%T), elapsed ${SECONDS}s"
  done
done
echo "[$ARM] done, elapsed ${SECONDS}s"
