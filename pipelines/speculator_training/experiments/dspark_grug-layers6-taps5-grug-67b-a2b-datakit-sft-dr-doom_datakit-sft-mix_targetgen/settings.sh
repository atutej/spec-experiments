# TARGETGEN DSpark drafter for Dr. Doom, 6 Grug layers (FFN 7,936), five taps (2/7/12/17/22), block 8, Markov rank 1024, 32,000-token draft vocabulary built from the regenerated text, trained FROM RANDOM INITIALISATION on Dr. Doom's own /think replies.
# Data: 281,000 rows (a random 281 of each shard's 400 parts of the 400,000-row regeneration). One epoch over these rows, with the learning-rate schedule of the original-completion 6-layer runs (horizon 382,747 steps, 1% warmup, linear decay), so the rate per step is identical and the run ends UNANNEALED, as theirs do at 12 h / 24 h.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). Architecture, taps, block, Markov rank, peak LR, seed and the env helpers come from the Datakit 6-layer folder's settings; every path that must be fresh is assigned OUTRIGHT below.
SIX_EXP="$(cd "$EXP_DIR/../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw" && pwd)"
source "$SIX_EXP/settings.sh"
MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
DRAFT_CONFIG="$EXP_DIR/draft_config.json"            # THIS folder's copy: 6 layers, FFN 7,936
WORK_DIR="${GRUG6T5_DATAKIT_TG_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_drdoom_grug6t5_datakit_targetgen281k}"   # fresh, assigned outright
PARTS_DIR="$WORK_DIR/regen_parts"                    # filled by data/select_parts.py; the source runs' parts are only read
DATA_DIR="$WORK_DIR/data"                            # the merged dataset (regenerate_token_rows.py merge, run at the start of the train job): Arrow rows + token_freq.pt + the draft-vocabulary maps made from the regenerated text
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
REGEN_PARTS_SRC=""                                   # no import: the parts are placed in PARTS_DIR by the script above
REGEN_ROWS=0                                         # documentation only: nothing is regenerated here
# ---- schedule: the original-completion 6-layer runs' (382,747 steps, peak 3e-4, warmup 1% = 3,827 steps, linear decay). This run covers one epoch of its rows (about 22-27k steps), so the learning rate only falls to about 0.94-0.95 of the peak. ----
EPOCHS=1
TRAIN_DATA_RATIO=0.995                               # 0.5% held out for the end-of-run validation (the targetgen convention)
SCHEDULER_TOTAL_STEPS=382747
MAX_STEPS=""                                         # empty: the epoch ends the run (no --max-steps)
CKPT_FREQ=0.02                                       # a checkpoint every 2% of the epoch
SAVE_TOTAL_LIMIT=3                                   # the epoch-end checkpoint (epoch0_end) is never pruned
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO" --draft-layer-type grug --grug-qk-mult 1.75 --scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
