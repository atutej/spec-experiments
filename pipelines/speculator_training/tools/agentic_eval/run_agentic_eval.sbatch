#!/bin/bash
# One agentic eval on one GPU node: serve the Dr. Doom target (with or without a DSpark drafter) on this node, run a Marin eval-policy Harbor benchmark against
# it with Daytona sandboxes, and record the server's speculative-decoding counters over the run. Submit through submit_agentic_eval.sh, not by hand.
#
# The agent is Terminus-2, which runs in the Harbor process on this node and calls the model at http://127.0.0.1:8000/v1; only the shell commands run in the
# Daytona sandbox. So nothing has to reach this node from outside. (mini-swe-agent, used for SWE-bench Verified, runs inside the sandbox and would need that.)
#
# env (exported by submit_agentic_eval.sh): ROOT (workspace: holds spec-experiments/, envs/, logs/), ARM (label), BENCH (ot-tblite | tb2), OUT (fresh result dir),
#   DRAFTER (DSpark checkpoint dir or Hugging Face repo id; empty = plain decoding), HARBOR_EXTRA (extra `harbor run` flags, for example "--n-tasks 3 --n-attempts 1"),
#   SERVER_EXTRA (extra `vllm serve` flags), MODEL (target), GPUS (GPUs on the node = data-parallel size, default 4), VLLM_ENV (conda env with vLLM, default vllm),
#   VLLM_OVERLAY (directory put first on PYTHONPATH, only for a vLLM without Grug DSpark support), KEYS_ENV (file with `export DAYTONA_API_KEY=...`),
#   AGENTIC_DIR (holds the fetched policy pin; default $ROOT/runs/agentic_eval_20261113)
# Slurm settings (partition, account, reservation, time, log file) come from the submit command, not from #SBATCH headers, so the same script runs on any machine.
#SBATCH -N 1
#SBATCH --exclusive
set -uo pipefail
: "${ROOT:?exported by submit_agentic_eval.sh}" "${ARM:?}" "${BENCH:?}" "${OUT:?}"
DRAFTER=${DRAFTER:-}; HARBOR_EXTRA=${HARBOR_EXTRA:-}; SERVER_EXTRA=${SERVER_EXTRA:-}; GPUS=${GPUS:-4}; VLLM_ENV=${VLLM_ENV:-vllm}; VLLM_OVERLAY=${VLLM_OVERLAY:-}
MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom}
AGENTIC_DIR=${AGENTIC_DIR:-$ROOT/runs/agentic_eval_20261113}
KEYS=${KEYS_ENV:-$HOME/keys.env}
POLICY_SHA=75a5690b0ff2e1510d73e576649e3e059f06da9a   # Marin eval policy #10197 pin (campaign eval-campaign-09-25-unlabeled)
HARBOR_SHA=016dd78b301b621ee4089f2da5776ebe80236820   # marin-community/harbor commit named by that pin (lib/marin/src/marin/external_dependencies.py)
[[ ! -e "$OUT" ]] || { echo "$OUT exists: results are never overwritten"; exit 1; }
mkdir -p "$OUT"
set +u; source ~/.bashrc 2>/dev/null || true; source "$ROOT/spec-experiments/env.sh" || exit 1; set -u
use_env() { set +u; conda activate "$1"; set -u; }

# Network: compute nodes must reach Daytona (sandboxes), GitHub (policy files, Harbor) and the Hugging Face Hub (weights).
reach() { local c; c=$(curl -s -o /dev/null -m 25 -w '%{http_code}' "$1"); [[ "$c" != 000 ]]; }
for u in https://app.daytona.io/api/ https://codeload.github.com https://huggingface.co; do
    reach "$u" || { echo "no outbound network to $u from $(hostname): this eval needs it"; exit 1; }
done

# Marin eval-policy files (Harbor configs), fetched once as the pinned commit's tarball.
PIN=$AGENTIC_DIR/marin_pin/experiments/evaluation/campaigns/eval-campaign-09-25-unlabeled
if [[ ! -d "$PIN" ]]; then
    mkdir -p "$AGENTIC_DIR/marin_pin" && curl -sfL --retry 3 "https://codeload.github.com/marin-community/marin/tar.gz/$POLICY_SHA" \
        | tar -xz -C "$AGENTIC_DIR/marin_pin" --strip-components=1 --wildcards "*/experiments/evaluation/*" "*/lib/marin/src/marin/external_dependencies.py" \
        || { echo "could not fetch the policy pin $POLICY_SHA"; exit 1; }
fi
CFG=$PIN/harbor-configs/$( [[ $BENCH == tb2 ]] && echo tb2-recovery || echo ot-tblite-recovery ).yaml
[[ -f "$CFG" ]] || { echo "no Harbor config $CFG"; exit 1; }

# Daytona key: read only this variable from the secrets file, never print it.
DAYTONA_API_KEY=$(grep -E '^export DAYTONA_API_KEY=' "$KEYS" 2>/dev/null | head -1 | sed -E 's/^export DAYTONA_API_KEY=//; s/^["'"'"']//; s/["'"'"']$//'); export DAYTONA_API_KEY
[[ -n "$DAYTONA_API_KEY" ]] || { echo "no DAYTONA_API_KEY in $KEYS (set KEYS_ENV)"; exit 1; }

# Pinned Harbor (the policy's marin-community/harbor commit and runtime requirements), built once.
HARBOR_VENV=$ROOT/envs/harbor_policy
if [[ ! -x $HARBOR_VENV/bin/harbor ]]; then
    command -v uv >/dev/null || python3 -m pip install --user -q uv || { echo "uv is not installed and pip could not install it"; exit 1; }
    export PATH="$HOME/.local/bin:$PATH"
    (flock 9
     if [[ ! -x $HARBOR_VENV/bin/harbor ]]; then
        echo "building $HARBOR_VENV"
        uv venv --python 3.12 "$HARBOR_VENV" && uv pip install --python "$HARBOR_VENV/bin/python" \
            "harbor @ git+https://github.com/marin-community/harbor.git@$HARBOR_SHA" \
            "daytona==0.200.2" "gcsfs==2026.7.0" "pydantic-settings==2.14.2" "s3fs==2026.7.0" || { rm -rf "$HARBOR_VENV"; exit 1; }
     fi) 9>"$ROOT/envs/.harbor_policy.lock" || exit 1
fi

# Serve the target on this node (DP x EP over its GPUs), with the policy's context and batching settings for the Dr. Doom model config.
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
    sleep 10
done
echo "[$ARM] server ready after ${SECONDS}s"
curl -s http://127.0.0.1:8000/metrics > "$OUT/metrics_before.prom"

# Harbor run (the policy invocation: staged config, hosted_vllm model, api_base to this node).
echo "[$ARM] harbor $BENCH start $(date +%T)"
"$HARBOR_VENV/bin/harbor" run --config "$CFG" --model "hosted_vllm/$MODEL" --ak "api_base=http://127.0.0.1:8000/v1" --ae "OPENAI_API_KEY=EMPTY" \
    --jobs-dir "$OUT/harbor_jobs" --job-name "$ARM" --yes $HARBOR_EXTRA > "$OUT/harbor.log" 2>&1
echo "[$ARM] harbor exit $? at $(date +%T), elapsed ${SECONDS}s"

curl -s http://127.0.0.1:8000/metrics > "$OUT/metrics_after.prom"
python3 "$ROOT/spec-experiments/pipelines/speculator_training/tools/agentic_eval/spec_metrics.py" "$OUT/metrics_before.prom" "$OUT/metrics_after.prom" --out "$OUT/spec_metrics.json"
