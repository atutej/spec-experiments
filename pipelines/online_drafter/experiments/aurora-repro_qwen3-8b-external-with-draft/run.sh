#!/bin/bash
# Aurora's examples/qwen3-8b-external-with-draft, with the Marin vLLM fork serving the target instead of SGLang: the ONLINE run (trainer + server + request stream).
#   bash run.sh <NEW run dir> <data dir from tools/build_aurora_prompts.py> [extra aurora config overrides...]
# GPUs: 0,1 = Aurora trainer (FSDP/DP, as in the example), 2 = vLLM server (Qwen3-8B fits on one GPU). Env knobs: BATCH (default 48 requests per rollout call),
# MAX_TOKENS (512), SERVER_ARGS (extra args for tools/vllm_capture_server.py), REQUESTS (stop after this many; default: all prompts once), TEMPERATURE (0, Aurora's
# example). FOREVER=1 (the long job): no step budget (the caller passes training.num_train_steps), the prompts cycle (use TEMPERATURE=1.0 then, or every pass repeats the
# same replies), and the script runs until Slurm ends the job or a process dies; drafts are copied to $RUN/drafts/step_N (forever_watch in tools/lib.sh).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; TOOLS="$HERE/../../tools"; source "$TOOLS/lib.sh"
init_run "${1:?usage: run.sh <new run dir> <data dir> [overrides]}"; DATA=${2:?data dir}; shift 2
[[ -f "$DATA/dataset.jsonl" ]] || { echo "missing dataset.jsonl in $DATA" >&2; exit 1; }
for p in 6380 18080 30000 50052 8090; do port_free $p || { echo "port $p is in use" >&2; exit 1; }; done
IP=$(local_ip); BATCH=${BATCH:-48}; MAX_TOKENS=${MAX_TOKENS:-512}; N=$(wc -l < "$DATA/dataset.jsonl"); REQUESTS=${REQUESTS:-$N}; FOREVER=${FOREVER:-0}; TEMPERATURE=${TEMPERATURE:-0}
TARGET=Qwen/Qwen3-8B; DRAFT=Tengyunw/qwen3_8b_eagle3; TRAIN_GPUS=0,1; SERVER_GPU=2
RAYDIR=/tmp/aurora_ray_$$; export RAY_ADDRESS="$IP:6380" RAY_AUTH_MODE=token   # Ray 2.59 heads require token auth; the token lives in ~/.ray/auth_token
cleanup() {
    echo "stopping..."; touch "$RUN/stop"
    for n in client server training; do stop_one $n; done
    ( set +u; source "$ROOT/spec-experiments/env.sh" >/dev/null 2>&1; conda activate aurora; set -u; ray stop >/dev/null 2>&1 ) || true
    rm -rf "$RAYDIR"
}
trap cleanup EXIT INT TERM
echo "IP $IP | requests $REQUESTS of $N | batch $BATCH | data $DATA"

# 1. Ray (own temp dir: a short path, Ray's socket paths are length-limited), GPUs 0,1
( set +u; source "$ROOT/spec-experiments/env.sh" >/dev/null 2>&1; conda activate aurora; set -u; unset TRANSFORMERS_CACHE
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS ray start --head --num-gpus 2 --port 6380 --temp-dir "$RAYDIR" --disable-usage-stats ) > "$RUN/logs/ray.log" 2>&1 || { echo "ray failed"; tail -5 "$RUN/logs/ray.log"; exit 1; }

# 2. Aurora trainer (starts the Mooncake master and the callback server; creates nothing to serve here: the draft is the pretrained one)
run_in aurora training CUDA_VISIBLE_DEVICES=$TRAIN_GPUS HF_HUB_OFFLINE=1 AURORA_LOG_LEVEL=INFO TORCHINDUCTOR_CACHE_DIR="$RUN/inductor" RAY_ADDRESS="$RAY_ADDRESS" -- \
    bash -c "cd '$RUN' && exec python -m aurora.train_entry --config '$HERE/config.yaml' \
        dataset.train_data_path='$DATA/dataset.jsonl' output_dir='$RUN/out' cache_dir='$RUN/cache' model.draft_model_config='$HERE/draft_config.json' \
        training.training_num_gpus_per_node=2 inference.inference_num_gpus_per_node=2 online_serving.enabled=true online_serving.port=18080 \
        online_serving.sglang_url=http://localhost:30000 mooncake.master_server_address=$IP:50052 mooncake.metadata_port=8090 $*"
wait_http "http://localhost:18080/health" 900 training || exit 1

# 3. vLLM capture server (the Marin fork, unmodified) with the pretrained draft
run_in vllm server CUDA_VISIBLE_DEVICES=$SERVER_GPU HF_HUB_OFFLINE=1 VLLM_LOGGING_LEVEL=WARNING PYTHONPATH="$TOOLS" TOKENIZERS_PARALLELISM=false -- \
    python "$TOOLS/vllm_capture_server.py" --target $TARGET --draft $DRAFT --num-spec-tokens 5 --max-model-len 8192 --gpu-memory-utilization 0.85 --port 30000 --scratch-dir /tmp ${SERVER_ARGS:-}
wait_http "http://localhost:30000/health" 1500 server || exit 1

# 4. the request stream (Aurora's send_requests.sh: dataset order, temperature 0)
run_in aurora client HF_HUB_OFFLINE=1 -- python "$TOOLS/rollout_client.py" --server http://localhost:30000 --callback http://localhost:18080 --prompts "$DATA/dataset.jsonl" --tokenizer "$TARGET" --chat-template-kwargs '{"enable_thinking": true}' --max-prompt-tokens 6000 \
    --batch "$BATCH" --max-tokens "$MAX_TOKENS" --temperature $TEMPERATURE --mooncake-master "$IP:50052" --mooncake-metadata "http://$IP:8090/metadata" --local-hostname "$IP" \
    --stop-file "$RUN/stop" --log "$RUN/rollouts.jsonl" ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT" --wandb-entity "${WANDB_ENTITY:-}" --wandb-group "${WANDB_GROUP:-}" --wandb-dir "$RUN/wandb" --steps-per-sync "${SYNC_INTERVAL:-0}"} \
    $([[ $FOREVER == 1 ]] && echo "--loop" || echo "--max-batches $(( (REQUESTS + BATCH - 1) / BATCH ))")
echo "running: training log $RUN/logs/training.log, server log $RUN/logs/server.log, per-batch results $RUN/rollouts.jsonl"
[[ $FOREVER == 1 ]] && { forever_watch; exit 1; }
# In external mode Aurora never exits on its own: after its last step it logs "Max steps (N) reached, but external training server active. Waiting for more data..."
# and keeps serving the callback. So: stop once the request stream is done AND the trainer has reached its step budget (plus 60 s to drain), or after TRAIN_GRACE
# seconds from the end of the stream (the trainer may be waiting for samples that will never come, e.g. dropped windows).
TRAIN_GRACE=${TRAIN_GRACE:-1800}; client_done_at=""; marker_at=""
while alive training; do
    if ! alive client; then
        [[ -z "$client_done_at" ]] && { client_done_at=$SECONDS; echo "client finished; waiting for the trainer to reach its step budget (up to ${TRAIN_GRACE}s)"; }
        if grep -aq "Max steps (.*) reached" "$RUN/logs/training.log" 2>/dev/null; then
            [[ -z "$marker_at" ]] && { marker_at=$SECONDS; echo "trainer reached its step budget; draining for 60 s"; }
            [[ $((SECONDS - marker_at)) -ge 60 ]] && { echo "done; stopping"; break; }
        fi
        [[ $((SECONDS - client_done_at)) -ge $TRAIN_GRACE ]] && { echo "grace period over; stopping"; break; }
    fi
    sleep 20
done
echo "training alive: $(alive training && echo yes || echo no) | client alive: $(alive client && echo yes || echo no)"
