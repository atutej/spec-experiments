# Settings of the DSpark / Qwen3-0.6B / Nemotron-Terminal experiment, shared by the machine launchers.
# Sourced after env.sh (needs PROJECT_ROOT). Block 1 is copied from genai/run.sh; a launcher may set
# REGEN_CONCURRENCY or others before sourcing, or you can override from the environment:
#   REGEN_LIMIT=300 WORK_DIR=$PROJECT_ROOT/runs/test MAX_STEPS=10 bash vista/run.sh
# Performance-only knobs (concurrency, GPU memory) may differ per machine; everything that changes what
# is trained (dataset, sampling, DSpark settings) must stay the same everywhere.
SPEC_DIR=${SPEC_DIR:-$PROJECT_ROOT/speculators}
MAX_STEPS=${MAX_STEPS:-}         # smoke runs only: stop training after this many steps (empty = full epoch)
MODEL="Qwen/Qwen3-0.6B"
DATASET="nemotron-terminal"      # preset in speculators' DATASET_CONFIGS (nvidia/Nemotron-Terminal-Corpus)
SUBSET=""                        # "" = the preset default (dataset_adapters); or skill_based_{easy,medium,mixed}
SAMPLE_SEED=0                    # seed of the step-0 random sample
REGEN_LIMIT=${REGEN_LIMIT:-100000}               # conversations to sample and regenerate (dataset_adapters has ~226k)
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_regen_online_think_mooncake_100k}
SOURCE_FILE="$WORK_DIR/source/${DATASET}${SUBSET:+_$SUBSET}_${REGEN_LIMIT}_seed${SAMPLE_SEED}.jsonl"   # step 0 output
REGEN_FILE="$WORK_DIR/regen/nemotron_qwen3_0_6b.jsonl"   # step 1 output
DATA_DIR="$WORK_DIR/data"        # step 2 output; kept separate because prepare-data
                                 # --overwrite refuses directories holding other files
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
VLLM_PORT=8000
MOONCAKE_PORT=50051              # hidden states move through Mooncake's in-RAM store, not files
MOONCAKE_GLOBAL_GIB=4
MOONCAKE_LOCAL_GIB=2
GPU_MEM_UTIL=0.9                 # assumes the GPUs are (almost) free; lower it if other jobs share them

# Generation settings -- match how you will SERVE the model.
ENABLE_THINKING="true"           # "true" if you serve Qwen3 with thinking on
MAX_GEN_TOKENS=8192              # per-request max_tokens; also stops a conversation once its
                                 # total length passes this (regenerate-responses behaviour).
                                 # = SEQ_LENGTH: longer rows would be clipped by prepare-data anyway
REGEN_MAX_MODEL_LEN=32768        # Qwen3-0.6B native context. Must exceed prompt + MAX_GEN_TOKENS, and
                                 # a prompt can pass MAX_GEN_TOKENS once the next terminal-output turn
                                 # is appended; 12288 failed ~15% of conversations with HTTP 400
REGEN_CONCURRENCY=${REGEN_CONCURRENCY:-512}            # scale with NUM_ALL_GPUS so every replica stays busy
MAX_ERROR_FRAC=0.02              # abort if more conversations than this fail

# Training settings (from the official DSpark Qwen3-0.6B example)
SEQ_LENGTH=8192                  # training rows longer than this are clipped by prepare-data
SPECULATOR_TYPE="dspark"
EPOCHS=1
LR=3e-4
BLOCK_SIZE=8                     # tokens drafted per step
MAX_ANCHORS=3072
NUM_LAYERS=3
TARGET_LAYER_IDS="2 14 25"       # Qwen3-0.6B has 28 layers; launch_vllm.py also appends layer 28
DRAFT_VOCAB_SIZE=32000           # reduced vocab: a full 152K head would dwarf a 3-layer draft
                                 # for this small model. Set to "" to use the full vocabulary.
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"       # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0


# ---- derived ----
if [[ "$ENABLE_THINKING" == "true" ]]; then   # Qwen3's recommended sampling per mode
    SAMPLING_PARAMS='{"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": true}}'
else
    SAMPLING_PARAMS='{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": false}}'
fi
EXPORT_ARGS=()
[[ -n "$SUBSET" ]] && EXPORT_ARGS+=(--subset "$SUBSET")
VOCAB_ARGS=()
[[ -n "$DRAFT_VOCAB_SIZE" ]] && VOCAB_ARGS=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
TRAIN_EXTRA=(); [[ -n "$MAX_STEPS" ]] && TRAIN_EXTRA=(--max-steps "$MAX_STEPS")
