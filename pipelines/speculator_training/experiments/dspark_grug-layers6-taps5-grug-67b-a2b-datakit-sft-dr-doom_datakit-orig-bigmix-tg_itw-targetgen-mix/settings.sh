# DSpark drafter for Dr. Doom (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom), trained FROM SCRATCH on Datakit ORIGINAL + BigMix TARGETGEN together, on TACC Horizon.
# Architecture = the layers6-taps5 drafters (6 Grug layers, FFN 7,936, taps 2/7/12/17/22, block 8, Markov rank 1024, 32k draft vocabulary), so the result compares with the Datakit-orig 24h and BigMix-TG 28h drafters.
# Data: data/build_mix.py (natural proportions, about 46.0B original + 3.2B targetgen tokens, no upsampling; the trainer shuffles all rows). One epoch (about 409k steps) with the learning rate decayed over the whole epoch;
# the run is STOPPED BY HAND where the effective time (training + regeneration credit of the targetgen rows seen) reaches 24 h: tools/effective_time.py. See NOTES.md.
# Self-contained (no parent settings chain). Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). Overridable from the environment: WORK_DIR, DATA_DIR, MAX_STEPS, SMOKE, RUN_NAME, LOGGER.
SPEC_DIR=${SPEC_DIR:-$PROJECT_ROOT/speculators}   # branch grug-dspark-gate (the launcher checks it)
MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
SEQ_LENGTH=${SEQ_LENGTH:-32768}                  # the dataset's clip length AND the trainer's --total-seq-len
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_drdoom_layers6-taps5_datakit-orig-bigmix-tg-mix}
DATA_DIR=${DATA_DIR:-$WORK_DIR/data}             # built by data/build_mix.py (horizon/build_data.sbatch); the trainer writes its d2t/t2d vocabulary cache here
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
VLLM_PORT=8000
MAX_MODEL_LEN=$((SEQ_LENGTH + 1024))             # the server must accept a full row
MOONCAKE_PORT=50051                              # hidden states move through Mooncake's in-RAM store (per node), not files
MOONCAKE_GLOBAL_GIB=${MOONCAKE_GLOBAL_GIB:-32}  # a 32,768-token row is 5 taps + final = 6 x 2,560 x bf16 x 32,768 = 1.0 GB: 32 / 16 GiB hold tens of rows (Vista's values; Horizon nodes have 1.7 TB RAM)
MOONCAKE_LOCAL_GIB=${MOONCAKE_LOCAL_GIB:-16}
GPU_MEM_UTIL=0.9

# ---- training (the layers6-taps5 recipe) ----
SPECULATOR_TYPE="dspark"
EPOCHS=1
LR=3e-4
BLOCK_SIZE=8
MAX_ANCHORS=${MAX_ANCHORS:-12288}
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # 6 layers, FFN 7,936, partial_rotary_factor 0.5
TARGET_LAYER_IDS="2 7 12 17 22"                  # of the target's 26 layers; launch_vllm.py appends the final output (26)
DRAFT_VOCAB_SIZE=32000                           # built by the trainer from DATA_DIR/token_freq.pt (summed counts of both sources)
MASK_TOKEN_ID=128004
MARKOV_RANK=1024
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
# Held-out fraction: exactly the validation shard of the mix (manifest.json written by build_mix.py).
TRAIN_DATA_RATIO=${TRAIN_DATA_RATIO:-$(python3 -c "import json,sys; print(json.load(open('$DATA_DIR/manifest.json'))['train_data_ratio'])" 2>/dev/null || echo 0.999)}
MAX_STEPS=${MAX_STEPS:-}                         # empty: one full epoch, learning rate decayed over it (the real run). A smoke test sets MAX_STEPS=10.
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-0}          # 0 = KEEP EVERY step checkpoint (from the resume at step 7,752 on; the first segment ran with 3, so steps 408-6,528 are gone). 3 GB each, one per 408 steps (about 8 min): about 22 GB per hour of training.
CKPT_FREQ=${CKPT_FREQ:-0.001}                    # a checkpoint every 0.1% of the epoch (about 409 steps, about 8 min)
SMOKE=${SMOKE:-0}                                # SMOKE=1: no W&B, no upload

# ---- metric logging ----
if [[ "$SMOKE" == 1 ]]; then LOGGER=${LOGGER-}; else LOGGER=${LOGGER-wandb}; fi
RUN_NAME=${RUN_NAME:-$(basename "$EXP_DIR")}
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_speculator
[[ -f "$PROJECT_ROOT/../keys.env" ]] && { set -a; source "$PROJECT_ROOT/../keys.env"; set +a; }   # WANDB_API_KEY, HF_TOKEN (never printed)
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO" --draft-layer-type grug --grug-qk-mult 1.75 --num-workers 48 --prefetch-factor 2)
[[ -z "$MAX_STEPS" ]] || TRAIN_EXTRA+=(--max-steps "$MAX_STEPS")

# ---- upload (by hand once the run is stopped: HF_REPO=auto bash horizon/run_2node_ddp.sh upload) ----
HF_REPO=${HF_REPO-}
HF_NAMESPACE=${HF_NAMESPACE:-}
HF_PRIVATE=${HF_PRIVATE:-false}
if [[ "$HF_REPO" == auto ]]; then HF_REPO="${HF_NAMESPACE:+$HF_NAMESPACE/}$RUN_NAME-${HF_DATE:-$(date +%Y%m%d)}"; fi
HF_DESCRIPTION="DSpark drafter (6 Grug-style layers, FFN 7,936, 32,000-token draft vocabulary built from the combined data, taps on target layers 2, 7, 12, 17 and 22, block size 8, Markov rank 1024) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained from scratch on the Datakit SFT mixture's own replies plus Dr. Doom's own replies to BigMix prompts (natural proportions, about 46.0B + 3.2B tokens), stopped at 24 h of effective time."

# ---- envs: activate by hand (PATH + the env's activate.d hooks); on Horizon the serving env is `vllm_fork` ----
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
# Node-local Triton/Inductor caches for the trainer (two nodes compiling into one shared cache hit `Stale file handle` on Vista, job 1120063), expandable segments (OOM fix of the 10-layer run).
torchrun() {
    local d="/tmp/${USER}_mix_cache_${SLURM_JOB_ID:-nojob}"; mkdir -p "$d/triton" "$d/inductor"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR="$d/triton" TORCHINDUCTOR_CACHE_DIR="$d/inductor" command torchrun "$@"
}
