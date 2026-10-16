# Settings of the DSpark drafter for Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 (the "Snowball" SFT model), trained OFFLINE on the FULL Datakit SFT mixture
# (6.48M rows, 53.7B tokens at a 32,768-token clip; the data Grug itself was fine-tuned on, taken as its own completions): no regeneration. Sibling of
# ../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_offpolicy (the 100k-row sample whose recipe this reuses). See PLAN.md. Sourced after env.sh (needs PROJECT_ROOT).
# Default (TOKEN_BUDGET=full): ONE EPOCH over the whole dataset, as close to Grug's own SFT as we can get (the same mixture and allocation). A smaller budget is possible: TOKEN_BUDGET=<tokens>
# -> MAX_STEPS steps, and the trainer decays the learning rate linearly over exactly MAX_STEPS and stops there, so an early stop is annealed (the dataset is shuffled into proportional
# shards, so a prefix has the right source mix). Either way the run is a series of resumable 12 h jobs: vista/drive_chain.sh (one job at a time) or vista/submit_chain.sh (a fixed count).
# Override from the environment, e.g. a smoke test:
#   SMOKE=1 MAX_STEPS=10 DATA_DIR=<dataset> WORK_DIR=<abs path> bash vista/run.sh train
# The speculators checkout: the main one, `speculators/`, on branch grug-dspark-gate (gated verifier head in the DFlash/DSpark targets, grug_moe config
# registration, a launch_vllm.py that reads the layer count from config.json). The env's editable install points at it, so no PYTHONPATH is needed. On any
# other branch (e.g. nemotron-terminal-preset) Grug does not work: run.sh checks for the patch and stops with a message.
SPEC_DIR=${SPEC_DIR:-$PROJECT_ROOT/speculators}
MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"   # the target: hidden states in training, lm_head / gated final norm for the TV loss
# ---- data: built by the Datakit command, not by this chain: select (all 203 sources), write --seq-length 32768, finalize (data/submit_write.sh) ----
SEQ_LENGTH=${SEQ_LENGTH:-32768}                  # the dataset's clip length AND the trainer's --total-seq-len
DATA_DIR=${DATA_DIR:-/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}}
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_full}
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
# Budget. About 54,000 tokens per step and 0.51 s per step were measured on the 100k run (one gb node: 2 GPUs serve hidden states, 2 train), i.e. about 106k tokens/s:
# Full epoch: about 47.6B tokens after the 32,768 clip (11.4% of the 53.7B allocation is clipped away) is about 880k steps, about 125 h, 11 jobs of 12 h (estimate, not measured at this scale).
# 12.5B tokens would be about 232k steps, about 33 h, three jobs.
TOKENS_PER_STEP=${TOKENS_PER_STEP:-54000}
TOKEN_BUDGET=${TOKEN_BUDGET:-full}                  # "full" (one epoch) or a number of tokens
if [[ "$TOKEN_BUDGET" == full ]]; then MAX_STEPS=${MAX_STEPS:-}; else MAX_STEPS=${MAX_STEPS:-$((TOKEN_BUDGET / TOKENS_PER_STEP))}; fi   # empty: the trainer runs the epoch and decays the LR over it
TRAIN_DATA_RATIO=${TRAIN_DATA_RATIO:-0.999}      # held-out fraction for the end-of-run validation: the trainer's default of 0.9 would hold out 5.4B tokens (hours of validation)
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}                         # step checkpoints kept (checkpoint-<step>/, about 1 GB each)
CKPT_FREQ=${CKPT_FREQ:-0.001}                    # a checkpoint every 0.1% of the EPOCH (about 880 steps of a 880k-step epoch, about 7.5 min; a save took 1-2 s in the smoke test, under 0.5% of the time): a stop at the 12 h limit loses at most that much
SMOKE=${SMOKE:-0}                                # SMOKE=1: no W&B, no upload

# ---- metric logging (the trainer's --logger) ----
if [[ "$SMOKE" == 1 ]]; then LOGGER=${LOGGER-}; else LOGGER=${LOGGER-wandb}; fi
RUN_NAME=${RUN_NAME:-$(basename "$EXP_DIR")}     # the experiment's folder name
# Entity and project are set outright: this account's default W&B entity is the team "dogml", not "atutej".
export WANDB_ENTITY=atutej WANDB_PROJECT=marin_speculator
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
TRAIN_EXTRA=()
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO")
[[ -z "$MAX_STEPS" ]] || TRAIN_EXTRA+=(--max-steps "$MAX_STEPS")

# ---- upload of the finished drafter to the Hugging Face Hub (best and latest checkpoint; tools/upload_drafter_to_hf.py) ----
# HF_REPO=auto (the default for full runs) names the repo <experiment folder name>-<YYYYMMDD> under the logged-in account, or under HF_NAMESPACE;
# HF_REPO= (empty) turns the upload off. Chain jobs do NOT upload (the default is off: an intermediate checkpoint should not be published, and a job cut by the time limit may still
# reach the upload stage); upload the finished run by hand: HF_REPO=auto bash vista/run.sh upload. PUBLIC unless HF_PRIVATE=true.
HF_REPO=${HF_REPO-}
HF_NAMESPACE=${HF_NAMESPACE:-}
HF_PRIVATE=${HF_PRIVATE:-false}
if [[ "$HF_REPO" == auto ]]; then HF_REPO="${HF_NAMESPACE:+$HF_NAMESPACE/}$RUN_NAME-${HF_DATE:-$(date +%Y%m%d)}"; fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21, trained offline on the Datakit SFT mixture the target was fine-tuned on (token budget in settings.sh, 32K context)."
