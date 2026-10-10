# Settings of the SFT experiment: full fine-tune of Qwen3-0.6B on the Nemotron-Terminal corpus (the SAME 100k conversations the
# speculator pipeline uses), one example per assistant turn, loss on that turn's thinking and response only. Its checkpoint is meant to
# become the target model of a drafter experiment (pipelines/speculator_training). See ../../README.md for the decisions behind this.
# Sourced after env.sh (needs PROJECT_ROOT). Override from the environment, e.g.
#   SAMPLE_LIMIT=300 MAX_STEPS=10 WORK_DIR=<abs path> bash vista/run.sh
# Values marked (placeholder) were not tuned or decided yet.
MAX_STEPS=${MAX_STEPS:-}         # smoke runs only: stop training after this many steps (empty = the full run)
MODEL="Qwen/Qwen3-0.6B"
TEMPLATE="qwen3"                 # LLaMA-Factory template; with mask_history, earlier turns' thinking is dropped (upstream discarding_history_cot)
DATASET="nemotron-terminal"      # preset in speculators' DATASET_CONFIGS (nvidia/Nemotron-Terminal-Corpus), used by the export script
SUBSET=""                        # "" = the preset default (dataset_adapters)
SAMPLE_SEED=0
SAMPLE_LIMIT=${SAMPLE_LIMIT:-100000}
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/sft_qwen3_0_6b_nemotron_terminal_corpus_100k}
SOURCE_FILE="$WORK_DIR/source/${DATASET}${SUBSET:+_$SUBSET}_${SAMPLE_LIMIT}_seed${SAMPLE_SEED}.jsonl"
# The speculator pipeline's export of the same sample: linked instead of re-exported, so SFT and the drafter use the SAME conversations.
SHARED_SOURCE_FILE=${SHARED_SOURCE_FILE:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_regen_online_think_mooncake_100k/source/${DATASET}${SUBSET:+_$SUBSET}_${SAMPLE_LIMIT}_seed${SAMPLE_SEED}.jsonl}
# sha256 of that 100k sample (seed 0, dataset revision a1667c4f...): checked after linking or exporting, so a different sample fails loudly.
# Empty = no check (any other SAMPLE_LIMIT or SUBSET).
if [[ "$SAMPLE_LIMIT" == 100000 && -z "$SUBSET" ]]; then SAMPLE_SHA256=${SAMPLE_SHA256-791b7f071394ddd209ad892eef9e124533a0bafb2372cd100b894af986a54aad}; else SAMPLE_SHA256=${SAMPLE_SHA256-}; fi

# Dataset: made by the prepare stage (tools/build_sft_dataset.py, LLaMA-Factory's own encoder), or provide a prebuilt one.
DATA_DIR="$WORK_DIR/data"        # a datasets.DatasetDict {train, validation} with input_ids / attention_mask / labels
# The convert stage makes the same dataset from the speculator pipeline's prepared data (one tokenized row per assistant turn) instead:
# no tokenization, minutes instead of hours. Either stage writes $DATA_DIR (whose manifest.json says how it was made).
PREPARED_DATA=${PREPARED_DATA:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_corpus_offpolicy_mooncake_100k/data}
PREBUILT_DATASET=${PREBUILT_DATASET:-}   # path of an existing dataset of that format: used for training instead of $DATA_DIR, prepare is skipped
MAX_LEN=${MAX_LEN:-16384}        # examples longer than this are dropped (not truncated)
VAL_FRACTION=${VAL_FRACTION:-0.002}               # fraction of conversations (by hash) held out for validation
BUILD_WORKERS=${BUILD_WORKERS:-64}       # processes of the prepare stage (a gg node has 144 cores)
BUILD_LIMIT=${BUILD_LIMIT:-}     # smoke runs only: build from the first N conversations

# Training (LLaMA-Factory, full fine-tuning, bf16, liger kernels, gradient checkpointing; data parallel over the node's GPUs)
OUTPUT_DIR="$WORK_DIR/model"     # checkpoint-N/ during training; the final model (HF format, usable as a vLLM target) at its root
LOG_DIR="$WORK_DIR/logs"
EPOCHS=${EPOCHS:-1}              # (placeholder)
LR=${LR:-2e-5}                   # (placeholder)
WARMUP_RATIO=0.03                # (placeholder)
GLOBAL_BATCH=${GLOBAL_BATCH:-32} # sequences per optimizer step = per-device batch x grad accumulation x GPUs = 4 x 2 x 4 (placeholder); accumulation is derived
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}  # tuned with vista/tune.sh (see NOTES.md): 8 runs out of memory at 16k tokens, 4 peaks at ~110 GiB of 184
GRAD_CKPT=${GRAD_CKPT:-true}      # gradient checkpointing (off = faster, more memory)
SAMPLING=${SAMPLING:-group_by_length}  # random | group_by_length: batches of similar length, less padding (1.9x faster at batch 4, see NOTES.md)
LOGGING_STEPS=${LOGGING_STEPS:-5}
SAVE_STEPS=${SAVE_STEPS:-1000}
EVAL_STEPS=${EVAL_STEPS:-1000}
# ---- metric logging ----
# W&B project/naming is not decided yet (to discuss after the pipeline works): off by default. REPORT_TO=wandb turns it on.
REPORT_TO=${REPORT_TO:-none}
RUN_NAME=${RUN_NAME:-$(basename "$(dirname "${BASH_SOURCE[0]}")")}
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_speculator   # this account's default entity is the team "dogml", not "atutej"
