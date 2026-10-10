#!/bin/bash
# SFT of Qwen3-0.6B on the Nemotron-Terminal corpus (the speculator pipeline's 100k conversations), one example per assistant turn,
# loss on that turn only, with LLaMA-Factory, on Vista. Settings: ../settings.sh. Design notes: ../../../README.md.
#
#   bash vista/run.sh [stage ...]       stages: export prepare convert train   (default: export prepare train)
#
#   export   CPU        link the speculator pipeline's 100k sample if it exists, else export it with the same script; then check
#                       its sha256 (skipped if $SOURCE_FILE exists, but the sha256 is always checked)
#   prepare  CPU        tools/build_sft_dataset.py: per-turn examples encoded by LLaMA-Factory's own encoder -> $DATA_DIR
#                       (skipped if $DATA_DIR or PREBUILT_DATASET exists)
#   convert  CPU        alternative to prepare: tools/convert_prepared_data.py turns the speculator pipeline's prepared data
#                       ($PREPARED_DATA, one tokenized row per assistant turn) into the same format -> $DATA_DIR. No tokenization,
#                       minutes. Rows clipped at $MAX_LEN are dropped; validation = random rows (see that tool). Skipped if $DATA_DIR exists.
#   train    gb node    llamafactory-cli train on all GPUs of the node (resumes from $OUTPUT_DIR/checkpoint-* if present)
#
# Runs as-is on an idev node (train needs a GPU node). Small test of everything:
#   SAMPLE_LIMIT=300 BUILD_LIMIT=300 BUILD_WORKERS=16 MAX_STEPS=5 WORK_DIR=<abs path> bash vista/run.sh
# (the sample sha256 is only checked for the full 100k sample.) PREBUILT_DATASET=<path> trains on an existing dataset instead.
set -euo pipefail
set -E; trap 'echo "run.sh: command failed (exit $?) at line $LINENO: $BASH_COMMAND" >&2' ERR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"   # repo root (this file: pipelines/<pipeline>/experiments/<name>/vista/)
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$REPO_DIR/pipelines/sft/tools"
source "$REPO_DIR/env.sh"
use_env() { set +u; conda activate "$1"; set -u; }   # conda's activate scripts are not nounset-safe
source "$EXP_DIR/settings.sh"

STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(export prepare train)
for s in "${STAGES[@]}"; do case "$s" in
    export|prepare|convert) ;;
    train) [[ "$NUM_GPUS" -ge 1 ]] || { echo "stage 'train' needs a GPU node (NODE_KIND=$NODE_KIND)" >&2; exit 1; } ;;
    *) echo "unknown stage '$s' (export prepare convert train)" >&2; exit 2 ;;
esac; done
mkdir -p "$LOG_DIR" "$(dirname "$SOURCE_FILE")"
# Work from the run directory, not from wherever the job was submitted (a Slurm job starts in the submit directory, and worker
# processes re-enter the parent's working directory; renaming that folder while a job runs crashed a run once).
cd "$WORK_DIR"
echo "node=$(hostname) cwd=$PWD kind=$NODE_KIND gpus=$NUM_GPUS stages=${STAGES[*]} WORK_DIR=$WORK_DIR SAMPLE_LIMIT=$SAMPLE_LIMIT MAX_STEPS=${MAX_STEPS:-none}"

check_sample_sha() {
    [[ -n "$SAMPLE_SHA256" ]] || { echo "sample sha256 not checked (not the 100k sample)"; return 0; }
    local got; got=$(sha256sum "$SOURCE_FILE" | cut -d' ' -f1)
    [[ "$got" == "$SAMPLE_SHA256" ]] || { echo "sample $SOURCE_FILE has sha256 $got, expected $SAMPLE_SHA256" >&2; exit 1; }
    echo "sample sha256 ok ($SAMPLE_SHA256)"
}

stage_export() {
    echo "=== Export: the $SAMPLE_LIMIT-conversation sample of $DATASET ==="
    if [[ ! -e "$SOURCE_FILE" ]]; then
        if [[ -f "$SHARED_SOURCE_FILE" ]]; then
            echo "linking the speculator pipeline's sample $SHARED_SOURCE_FILE"
            ln -s "$SHARED_SOURCE_FILE" "$SOURCE_FILE"
        else
            use_env speculators   # the export script imports speculators' dataset presets
            local args=(--dataset "$DATASET" --limit "$SAMPLE_LIMIT" --seed "$SAMPLE_SEED" --out "$SOURCE_FILE")
            [[ -z "$SUBSET" ]] || args+=(--subset "$SUBSET")
            python "$REPO_DIR/pipelines/speculator_training/tools/export_registry_dataset.py" "${args[@]}"
        fi
    fi
    check_sample_sha
}

stage_prepare() {
    if [[ -n "$PREBUILT_DATASET" ]]; then echo "=== Prepare: PREBUILT_DATASET=$PREBUILT_DATASET is used, skipping ==="; return 0; fi
    [[ -f "$SOURCE_FILE" ]] || { echo "missing $SOURCE_FILE (run the export stage first)" >&2; exit 1; }
    if [[ -d "$DATA_DIR" ]]; then echo "=== Prepare: $DATA_DIR exists, skipping (delete it to redo) ==="; return 0; fi
    echo "=== Prepare: per-turn examples, encoded by LLaMA-Factory ($BUILD_WORKERS workers, max length $MAX_LEN) ==="
    use_env sft
    local args=(--source "$SOURCE_FILE" --out "$DATA_DIR" --model "$MODEL" --template "$TEMPLATE" --max-len "$MAX_LEN"
                --workers "$BUILD_WORKERS" --val-fraction "$VAL_FRACTION")
    [[ -z "$BUILD_LIMIT" ]] || args+=(--limit-conversations "$BUILD_LIMIT")
    python "$TOOLS/build_sft_dataset.py" "${args[@]}"   # writes $DATA_DIR.tmp, then renames
}

stage_convert() {
    if [[ -n "$PREBUILT_DATASET" ]]; then echo "=== Convert: PREBUILT_DATASET=$PREBUILT_DATASET is used, skipping ==="; return 0; fi
    if [[ -d "$DATA_DIR" ]]; then echo "=== Convert: $DATA_DIR exists, skipping (delete it to redo) ==="; return 0; fi
    [[ -d "$PREPARED_DATA" ]] || { echo "missing $PREPARED_DATA (the speculator pipeline's prepared data; run its prepare stage, or use the prepare stage here)" >&2; exit 1; }
    echo "=== Convert: $PREPARED_DATA -> $DATA_DIR ($BUILD_WORKERS workers, drop rows >= $MAX_LEN) ==="
    use_env sft
    local args=(--prepared "$PREPARED_DATA" --out "$DATA_DIR" --max-len "$MAX_LEN" --workers "$BUILD_WORKERS")
    [[ -z "$BUILD_LIMIT" ]] || args+=(--limit-rows "$BUILD_LIMIT")
    python "$TOOLS/convert_prepared_data.py" "${args[@]}"   # writes $DATA_DIR.tmp, then renames
}

write_train_config() {  # $1 = the dataset to train on; writes $WORK_DIR/train_config.yaml (kept with the run)
    local per_step=$(( PER_DEVICE_BATCH * NUM_GPUS ))
    (( GLOBAL_BATCH % per_step == 0 )) || { echo "GLOBAL_BATCH=$GLOBAL_BATCH is not a multiple of PER_DEVICE_BATCH x GPUs = $per_step" >&2; exit 1; }
    local accum=$(( GLOBAL_BATCH / per_step ))
    {
        cat <<YAML
### model
model_name_or_path: $MODEL
trust_remote_code: true
flash_attn: sdpa          # flash-attn has no aarch64 build here
enable_liger_kernel: true # fused linear + cross entropy: no 16k x 152k logits tensor

### method
stage: sft
do_train: true
finetuning_type: full

### dataset: the examples are already tokenized; LLaMA-Factory loads them and ignores the other data arguments
tokenized_path: $1
dataset: unused           # required by the argument check, not read when tokenized_path exists
val_size: 0.001           # likewise (validation comes from the tokenized dataset's validation split)
template: $TEMPLATE
cutoff_len: $MAX_LEN
mask_history: true        # train on the last turn only; earlier thinking is dropped from the context (discarding_history_cot)
enable_thinking: true

### output
output_dir: $OUTPUT_DIR
overwrite_output_dir: false   # an existing checkpoint-N is resumed
logging_steps: 5
save_strategy: steps
save_steps: $SAVE_STEPS
save_total_limit: 2
save_only_model: false
plot_loss: false
report_to: $REPORT_TO
run_name: $RUN_NAME

### train
per_device_train_batch_size: $PER_DEVICE_BATCH
gradient_accumulation_steps: $accum
learning_rate: $LR
num_train_epochs: $EPOCHS
lr_scheduler_type: cosine
warmup_ratio: $WARMUP_RATIO
optim: adamw_torch_fused
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
dataloader_num_workers: 4

### eval
eval_strategy: steps
eval_steps: $EVAL_STEPS
per_device_eval_batch_size: $PER_DEVICE_BATCH
YAML
        [[ -z "${MAX_STEPS:-}" ]] || echo "max_steps: $MAX_STEPS"
    } > "$WORK_DIR/train_config.yaml"
    echo "global batch $GLOBAL_BATCH = $PER_DEVICE_BATCH x $accum accumulation x $NUM_GPUS GPUs"
}

stage_train() {
    local data=${PREBUILT_DATASET:-$DATA_DIR}
    [[ -d "$data" ]] || { echo "missing $data (run the prepare or convert stage, or give PREBUILT_DATASET)" >&2; exit 1; }
    echo "=== Train: $MODEL on $data, output $OUTPUT_DIR ==="
    use_env sft
    write_train_config "$data"
    export TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton-$USER"
    llamafactory-cli train "$WORK_DIR/train_config.yaml" 2>&1 | tee -a "$LOG_DIR/train.log"
    echo "Done. The model is in $OUTPUT_DIR (a vLLM target: vllm serve $OUTPUT_DIR)."
}

for s in "${STAGES[@]}"; do "stage_$s"; done
