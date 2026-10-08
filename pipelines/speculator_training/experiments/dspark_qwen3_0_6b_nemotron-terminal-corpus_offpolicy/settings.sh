# Settings of the DSpark / Qwen3-0.6B / Nemotron-Terminal experiment that trains on the CORPUS'S OWN completions
# (DeepSeek-V3.2 terminus-2 trajectories) instead of Qwen3-0.6B's regenerated ones. Sibling of
# ../dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy (on-policy): everything that changes what is trained is the same as that
# experiment's Vista run (16384 / 6144 anchors, DSpark settings, same 100k conversations); only the data differs.
# Sourced after env.sh (needs PROJECT_ROOT). Override from the environment, e.g.
#   SAMPLE_LIMIT=300 MAX_STEPS=10 WORK_DIR=<abs path> bash vista/run.sh
SPEC_DIR=${SPEC_DIR:-$PROJECT_ROOT/speculators}
MAX_STEPS=${MAX_STEPS:-}         # smoke runs only: stop training after this many steps (empty = full epoch)
MODEL="Qwen/Qwen3-0.6B"
DATASET="nemotron-terminal"      # preset in speculators' DATASET_CONFIGS (nvidia/Nemotron-Terminal-Corpus)
SUBSET=""                        # "" = the preset default (dataset_adapters); or skill_based_{easy,medium,mixed}
SAMPLE_SEED=0                    # seed of the step-0 random sample (same as the on-policy experiment)
SAMPLE_LIMIT=${SAMPLE_LIMIT:-100000}             # conversations to sample (dataset_adapters has ~226k)
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_corpus_offpolicy_mooncake_100k}
SOURCE_FILE="$WORK_DIR/source/${DATASET}${SUBSET:+_$SUBSET}_${SAMPLE_LIMIT}_seed${SAMPLE_SEED}.jsonl"   # step 0 output
# The on-policy experiment's export of the same sample: linked instead of re-exported, so both runs use the SAME
# conversations (same seed and limit give the same rows, but sharing the file makes it certain).
SHARED_SOURCE_FILE=${SHARED_SOURCE_FILE:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_regen_online_think_mooncake_100k/source/${DATASET}${SUBSET:+_$SUBSET}_${SAMPLE_LIMIT}_seed${SAMPLE_SEED}.jsonl}
DATA_DIR="$WORK_DIR/data"        # prepare output (written to $DATA_DIR.tmp first, then renamed)
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
VLLM_PORT=8000
RENDER_PORT=8091                 # the GPU-less render server used by prepare
RENDER_API_SERVERS=6             # as launch_vllm.py picks for a 144-CPU node; the server was far from saturated in tests
RENDER_WORKERS=2                 # renderer threads per API server
MOONCAKE_PORT=50051              # hidden states move through Mooncake's in-RAM store, not files
MOONCAKE_GLOBAL_GIB=4
MOONCAKE_LOCAL_GIB=2
GPU_MEM_UTIL=0.9                 # assumes the GPUs are (almost) free; lower it if other jobs share them

# Training settings (as the on-policy Vista run; the first two differ from genai's 8192 and 3072)
SEQ_LENGTH=${SEQ_LENGTH:-16384}  # rows longer than this are clipped by prepare-data; a turn whose context alone fills it is skipped
SPECULATOR_TYPE="dspark"
EPOCHS=1
LR=3e-4
BLOCK_SIZE=8                     # tokens drafted per step
MAX_ANCHORS=${MAX_ANCHORS:-6144} # scaled with SEQ_LENGTH (3072 at 8192)
NUM_LAYERS=3
TARGET_LAYER_IDS="2 14 25"       # Qwen3-0.6B has 28 layers; launch_vllm.py also appends layer 28
DRAFT_VOCAB_SIZE=32000           # reduced vocab: a full 152K head would dwarf a 3-layer draft
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"       # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# ---- derived ----
EXPORT_ARGS=()
if [[ -n "$SUBSET" ]]; then EXPORT_ARGS+=(--subset "$SUBSET"); fi
VOCAB_ARGS=()
if [[ -n "$DRAFT_VOCAB_SIZE" ]]; then VOCAB_ARGS=(--draft-vocab-size "$DRAFT_VOCAB_SIZE"); fi
# ---- metric logging (the trainer's --logger) ----
# Weights & Biases for real runs; smoke runs (MAX_STEPS set) log nowhere unless LOGGER is given. LOGGER= (empty) turns it off.
if [[ -n "$MAX_STEPS" ]]; then LOGGER=${LOGGER-}; else LOGGER=${LOGGER-wandb}; fi
RUN_NAME=${RUN_NAME:-$(basename "$(dirname "${BASH_SOURCE[0]}")")}   # the experiment's folder name
# Entity and project are set outright, not defaulted: this account's default W&B entity is the team "dogml", not "atutej".
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_speculator
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
TRAIN_EXTRA=()
# An `if`, not `[[ ]] && ...`: as the last command of a sourced file a failing `[[ ]]` would make `source` return 1 and
# kill any caller running under `set -e`.
if [[ -n "$MAX_STEPS" ]]; then TRAIN_EXTRA=(--max-steps "$MAX_STEPS"); fi
