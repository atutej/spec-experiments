# Settings of the DSpark drafter experiment whose TARGET MODEL is the Qwen3-0.6B fine-tuned on the Nemotron-Terminal corpus
# (pipelines/sft/experiments/sft_qwen3_0_6b_nemotron-terminal-corpus, on the Hub), trained on the corpus's own completions
# (DeepSeek-V3.2 terminus-2 trajectories). Sibling of ../dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy: everything is the same
# (same 100k conversations, same prepared data, same DSpark settings and 16384 / 6144 anchors) except the target model: its hidden
# states (online training) and its lm_head come from the fine-tuned model.
# Sourced after env.sh (needs PROJECT_ROOT). Override from the environment, e.g.
#   SAMPLE_LIMIT=300 MAX_STEPS=10 WORK_DIR=<abs path> bash vista/run.sh
SPEC_DIR=${SPEC_DIR:-$PROJECT_ROOT/speculators}
MAX_STEPS=${MAX_STEPS:-}         # smoke runs only: stop training after this many steps (empty = full epoch)
# The target: the SFT model on the Hub (public). The repo is not pinned to a commit (the tools used here take only a model id): the commit this
# experiment was built against is recorded in NOTES.md. A local path of the same model also works, e.g.
#   MODEL=/scratch/.../runs/sft_qwen3_0_6b_nemotron_terminal_corpus_100k/model bash vista/run.sh ...
MODEL=${MODEL:-atutej/sft_qwen3_0_6b_nemotron-terminal-corpus-20261011}
BASE_MODEL="Qwen/Qwen3-0.6B"     # the model it was fine-tuned from: its tokenizer and chat template must equal the target's (checked before data is reused)
TARGET_DTYPE=bfloat16            # the SFT model is saved in float32; vLLM would pick another dtype for it, Qwen3 runs in bf16
DATASET="nemotron-terminal"      # preset in speculators' DATASET_CONFIGS (nvidia/Nemotron-Terminal-Corpus)
SUBSET=""                        # "" = the preset default (dataset_adapters); or skill_based_{easy,medium,mixed}
SAMPLE_SEED=0                    # seed of the step-0 random sample (same as the on-policy experiment)
SAMPLE_LIMIT=${SAMPLE_LIMIT:-100000}             # conversations to sample (dataset_adapters has ~226k)
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_sft_corpus_offpolicy_mooncake_100k}
SOURCE_FILE="$WORK_DIR/source/${DATASET}${SUBSET:+_$SUBSET}_${SAMPLE_LIMIT}_seed${SAMPLE_SEED}.jsonl"   # step 0 output
# The on-policy experiment's export of the same sample: linked instead of re-exported, so both runs use the SAME
# conversations (same seed and limit give the same rows, but sharing the file makes it certain).
SHARED_SOURCE_FILE=${SHARED_SOURCE_FILE:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_regen_online_think_mooncake_100k/source/${DATASET}${SUBSET:+_$SUBSET}_${SAMPLE_LIMIT}_seed${SAMPLE_SEED}.jsonl}
# The prepared data of the sibling experiment (base-model target): linked instead of re-rendering the corpus (about 3 hours), after
# checking that BASE_MODEL and MODEL tokenize and render chat identically (tools/check_same_tokenizer.py). The data holds tokens, a
# loss mask and the reduced-vocabulary maps computed from the data, nothing that depends on the target's weights.
SHARED_DATA_DIR=${SHARED_DATA_DIR:-$PROJECT_ROOT/runs/nemotron_qwen3_0_6b_corpus_offpolicy_mooncake_100k/data}
DATA_DIR="$WORK_DIR/data"        # prepare output (written to $DATA_DIR.tmp first, then renamed), or a link to SHARED_DATA_DIR
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
VLLM_PORT=8000
RENDER_PORT=8091                 # the GPU-less render server used by prepare
RENDER_WORKERS=8                 # renderer threads. `vllm launch render` is ONE process (it ignores --api-server-count) and is GIL-bound:
                                 # 27 clients got 171 calls/s with 2 threads, 333 with 8, 296 with 32 (so 8; more does not help)
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

# ---- upload of the finished drafter to the Hugging Face Hub (best and latest checkpoint; tools/upload_drafter_to_hf.py) ----
# HF_REPO=auto (the default for full runs) names the repo <experiment folder name>-<YYYYMMDD> under the logged-in account, or under
# HF_NAMESPACE=<user or org>; HF_REPO=<namespace>/<name> is explicit; HF_REPO= (empty) turns the upload off. Smoke runs (MAX_STEPS set)
# never upload unless HF_REPO is given. The repo is PUBLIC unless HF_PRIVATE=true. Needs a write token (HF_TOKEN or HUG_TOKEN).
if [[ -n "$MAX_STEPS" ]]; then HF_REPO=${HF_REPO-}; else HF_REPO=${HF_REPO-auto}; fi
HF_NAMESPACE=${HF_NAMESPACE:-}
HF_PRIVATE=${HF_PRIVATE:-false}
if [[ "$HF_REPO" == auto ]]; then HF_REPO="${HF_NAMESPACE:+$HF_NAMESPACE/}$RUN_NAME-${HF_DATE:-$(date +%Y%m%d)}"; fi
HF_DESCRIPTION="The target is Qwen3-0.6B fine-tuned on nvidia/Nemotron-Terminal-Corpus (the SFT model above); the drafter is trained on the corpus's own completions (DeepSeek-V3.2 terminus-2 trajectories) of the same 100k conversations, which are close to what that target writes."
