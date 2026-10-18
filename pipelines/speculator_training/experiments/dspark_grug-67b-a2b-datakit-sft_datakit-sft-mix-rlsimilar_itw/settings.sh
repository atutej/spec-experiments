# itw DSpark drafter for the 09.21 Snowball SFT model, trained FROM STEP 0 on the Datakit SFT rows nearest to the RL prompts ("rlsimilar"): the same drafter, recipe and 80k-step budget as the full itw run
# (../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw), but on a filtered dataset built by tools/select_similar_rows instead of a prefix of the whole mixture. See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the itw settings and assigns every path that must be fresh OUTRIGHT (never `${VAR:-default}` on a variable the itw settings also use).
ITW_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw" && pwd)"
source "$ITW_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"                 # the target (as the itw run)
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # the Datakit dataset, READ ONLY here
RL_DATASET="open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30"       # the RL prompts the rows are matched to (train.parquet, 84,403 rows)

# ---- data creation (vista/submit_data.sh): embeddings -> neighbours and selection -> the training dataset, all under BUILD_DIR ----
# RLSIM_BUILD_DIR / RLSIM_WORK_DIR exist only for the smoke tests (vista/smoke_data.sh, vista/smoke_train.sh); they are names no other experiment uses.
BUILD_DIR="${RLSIM_BUILD_DIR:-$PROJECT_ROOT/runs/grug_rlsimilar_data}"
EMB_DIR="$BUILD_DIR/emb"
SEL_DIR="$BUILD_DIR/sel"
DATA_DIR="$BUILD_DIR/dataset"                    # the trainer's dataset: it writes its d2t/t2d vocabulary cache here, so it is this experiment's own dataset, not a view of the Datakit one
SELECT_TOPK=${SELECT_TOPK:-64}                   # neighbours kept per RL prompt (the budget needs about 16 rounds; more allows a larger budget or dedup)
SELECT_TOKEN_BUDGET=${SELECT_TOKEN_BUDGET:-9.7e9}   # packed tokens: the itw run's 80,813 steps x 120,043 tokens per step; one epoch of the filtered set is then about 80.8k steps
SELECT_SEED=${SELECT_SEED:-0}
SELECT_DEDUP=${SELECT_DEDUP:-0}                  # 1: --dedup (a row is selected at most once). Off by default; the selection summary reports the duplicate count either way

# ---- training: fresh work dir, never the itw run's runs/grug_dspark_full ----
WORK_DIR="${RLSIM_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_rlsimilar}"
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DRAFT_CONFIG="$ITW_EXP/draft_config.json"        # the itw settings point at $EXP_DIR/draft_config.json, which this folder does not have
# Budget: TOKEN_BUDGET=full is one epoch of the FILTERED dataset (about 9.7B tokens, about 80.8k steps, about 11.7 h at 0.523 s/step on 2 nodes), with the learning rate decayed over that epoch (annealed).
# The itw settings' checkpoint rate (every 0.1% of the epoch) assumed a 382k-step epoch (383 steps); an 80k-step epoch at that rate would save every 80 steps. Every 1% is about 808 steps, about 7 min.
TRAIN_DATA_RATIO=0.999                           # held-out validation fraction; with duplicates in the selection some validation rows may also occur in training: see NOTES.md
CKPT_FREQ=0.01
# Derived at source time by the itw settings from the itw run's values: recompute.
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO")
[[ -z "$MAX_STEPS" ]] || TRAIN_EXTRA+=(--max-steps "$MAX_STEPS")
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi

HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21, trained itw from step 0 on the Datakit SFT rows nearest (EmbeddingGemma 2, round-robin top-k) to the Nemotron RL Ultra RLVR1 prompts, about 9.7B tokens."
