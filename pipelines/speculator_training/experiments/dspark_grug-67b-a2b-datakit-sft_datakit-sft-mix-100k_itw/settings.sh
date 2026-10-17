# Settings of the DSpark drafter for Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 (the "Snowball" SFT model), trained ITW (in-the-wild text, no regeneration) on a 100k-row sample of the
# Datakit SFT mixture (the data Grug itself was fine-tuned on, taken as its own completions): no regeneration. Sibling of
# ../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw (the full mixture; see its PLAN.md). Sourced after env.sh (needs PROJECT_ROOT).
# Override from the environment, e.g.
#   DATA_DIR=<dataset> MAX_STEPS=10 WORK_DIR=<abs path> bash vista/run.sh train
# The speculators checkout: the main one, `speculators/`, on branch grug-dspark-gate (gated verifier head in the DFlash/DSpark targets, grug_moe config
# registration, a launch_vllm.py that reads the layer count from config.json). The env's editable install points at it, so no PYTHONPATH is needed. On any
# other branch (e.g. nemotron-terminal-preset) Grug does not work: run.sh checks for the patch and stops with a message.
SPEC_DIR=${SPEC_DIR:-$PROJECT_ROOT/speculators}
MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"   # the target: hidden states in training, lm_head / gated final norm for the TV loss
# ---- data: built by the Datakit command, not by this chain (see NOTES.md): select --rows 100000, write --seq-length 32768, finalize ----
SEQ_LENGTH=${SEQ_LENGTH:-32768}                  # the dataset's clip length AND the trainer's --total-seq-len
DATA_DIR=${DATA_DIR:-/scratch/09749/atutej/marin_datakit/dataset_100k_seq${SEQ_LENGTH}}
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_100k}
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
VLLM_PORT=8000
MAX_MODEL_LEN=$((SEQ_LENGTH + 1024))             # the server must accept a full row
MOONCAKE_PORT=50051                              # hidden states move through Mooncake's in-RAM store, not files
# A 32,768-token row is 671 MB of hidden states (4 layers x 2,560 x bf16 x 32,768); the Qwen runs' 4 / 2 GiB held a handful. 32 / 16 GiB hold ~50 / 25 rows.
MOONCAKE_GLOBAL_GIB=${MOONCAKE_GLOBAL_GIB:-32}
MOONCAKE_LOCAL_GIB=${MOONCAKE_LOCAL_GIB:-16}
GPU_MEM_UTIL=0.9                                 # assumes the GPUs are (almost) free; lower it if other jobs share them

# ---- training settings (the DSpark / Qwen runs' recipe, with Grug's shapes) ----
SPECULATOR_TYPE="dspark"
EPOCHS=1
LR=3e-4
BLOCK_SIZE=8                                     # tokens drafted per step
MAX_ANCHORS=${MAX_ANCHORS:-12288}                # 0.375 per token of SEQ_LENGTH (3072 at 8192, 6144 at 16384, as in the Qwen runs)
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # 3 layers, FFN 3840 (1.5 x hidden), hidden 2560 = Grug's. REQUIRED: the default synthesized config reads
                                                 # hidden_act from the verifier config and Grug's has none. Mutually exclusive with --num-layers.
TARGET_LAYER_IDS="2 12 22"                       # 26 Grug layers; launch_vllm.py also appends the final layer (26)
DRAFT_VOCAB_SIZE=32000                           # reduced vocab from the dataset's token_freq.pt (supervised tokens only)
MASK_TOKEN_ID=128004                             # <|finetune_right_pad_id|>: Grug's tokenizer has no mask token; vLLM serving needs the same id in the draft config
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"                       # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
MAX_STEPS=${MAX_STEPS:-}                         # smoke runs only: stop training after this many steps (empty = full epoch)

# ---- metric logging (the trainer's --logger) ----
if [[ -n "$MAX_STEPS" ]]; then LOGGER=${LOGGER-}; else LOGGER=${LOGGER-wandb}; fi
RUN_NAME=${RUN_NAME:-$(basename "$EXP_DIR")}     # the experiment's folder name
# Entity and project are set outright: this account's default W&B entity is the team "dogml", not "atutej".
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_speculator
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
TRAIN_EXTRA=()
if [[ -n "$MAX_STEPS" ]]; then TRAIN_EXTRA=(--max-steps "$MAX_STEPS"); fi

# ---- upload of the finished drafter to the Hugging Face Hub (best and latest checkpoint; tools/upload_drafter_to_hf.py) ----
# HF_REPO=auto (the default for full runs) names the repo <experiment folder name>-<YYYYMMDD> under the logged-in account, or under HF_NAMESPACE;
# HF_REPO= (empty) turns the upload off. Smoke runs (MAX_STEPS set) never upload unless HF_REPO is given. PUBLIC unless HF_PRIVATE=true.
if [[ -n "$MAX_STEPS" ]]; then HF_REPO=${HF_REPO-}; else HF_REPO=${HF_REPO-auto}; fi
HF_NAMESPACE=${HF_NAMESPACE:-}
HF_PRIVATE=${HF_PRIVATE:-false}
if [[ "$HF_REPO" == auto ]]; then HF_REPO="${HF_NAMESPACE:+$HF_NAMESPACE/}$RUN_NAME-${HF_DATE:-$(date +%Y%m%d)}"; fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21, trained itw (on the dataset's own text, no regeneration) on a 100k-row sample (~0.7B tokens, 32K context) of the Datakit SFT mixture the target was fine-tuned on."
