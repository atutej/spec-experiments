# itw DSpark drafter for the 09.21 Snowball SFT model with a 3x WIDER FFN (intermediate_size 11,520 instead of 3,840: about 457M parameters instead of 280M), trained from step 0 on the SAME data, recipe,
# seed and full-epoch learning-rate schedule as ../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw (the existing itw run is the baseline: stopped by the 12 h limit at step 80,813). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the itw settings and assigns, OUTRIGHT (never `${VAR:-default}` on a variable the itw settings also use), every path that must be fresh.
ITW_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw" && pwd)"
source "$ITW_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"                 # the target (as the itw run)
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # the Datakit dataset, READ ONLY here
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # THIS folder's: the only difference from the itw run is intermediate_size 11520
WORK_DIR="${FFN3X_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_ffn3x}"      # fresh: never the itw run's runs/grug_dspark_full
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh): the trainer writes its d2t/t2d vocabulary cache here, so the shared dataset is not touched
# Checkpoints every 0.1% of the epoch (383 steps) as in the itw run; keep more of them so the checkpoint at step 80,813 (the itw run's last one, 211 x 383) survives if training runs past it.
SAVE_TOTAL_LIMIT=8
STOP_AFTER_STEP=80813                            # the comparison point; vista/run_train.sbatch does not start a job once a checkpoint at or past this step exists
# Derived at source time by the itw settings from the itw run's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 11,520, about 457M parameters) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21, trained itw (on the Datakit SFT mixture's own replies, no regeneration), same recipe as the itw drafter with a 3x wider FFN."
