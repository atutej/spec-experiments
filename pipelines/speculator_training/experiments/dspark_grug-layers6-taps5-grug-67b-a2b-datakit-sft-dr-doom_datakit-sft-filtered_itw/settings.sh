# itw DSpark drafter for Dr. Doom, 6 Grug layers, five taps, block 8, trained for ONE EPOCH on the Datakit SFT mixture WITHOUT the finance_v1 rows (nemotron_sft_v3/finance_v1/train, 326,695 rows, 9.4B tokens).
# Everything else is the Datakit 6-layer run's recipe (../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw). Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR).
SIX_EXP="$(cd "$EXP_DIR/../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw" && pwd)"
source "$SIX_EXP/settings.sh"
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}_nofinance"   # READ ONLY: built by data/build_filtered_dataset.py (finalized with data/finalize_dataset.py)
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # THIS folder's copy: 6 layers, FFN 7,936
WORK_DIR="${GRUG_L6T5_NOFIN_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_grug-layers6-taps5-nofinance}"   # fresh, assigned outright
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh)
# ---- schedule: ONE EPOCH of the filtered dataset, the learning rate decays linearly over it (the epoch is shorter than the full Datakit epoch of 382,747 steps, so the decay is steeper per step). To follow the
# ---- full-dataset run's rate per step instead, add --scheduler-total-steps 382747 (then the run ends unannealed at the filtered epoch's end).
unset MAX_STEPS 2>/dev/null; MAX_STEPS=""
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO" --draft-layer-type grug --grug-qk-mult 1.75)
SAVE_TOTAL_LIMIT=3
CKPT_FREQ=${NOFIN_CKPT_FREQ:-0.0013}              # a checkpoint every 0.13% of the epoch (about 400 steps, set once the epoch length is known)
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
