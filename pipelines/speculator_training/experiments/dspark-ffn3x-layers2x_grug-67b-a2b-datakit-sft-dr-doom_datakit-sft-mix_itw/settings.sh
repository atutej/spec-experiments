# itw DSpark drafter for the DR. DOOM target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) with a 3x WIDER FFN (intermediate_size 11,520 instead of 3,840) and TWICE THE LAYERS (6 instead of 3): about 772M trainable parameters instead of 280M, trained from step 0
# on the same data, recipe, seed and full-epoch learning-rate schedule as the Dr. Doom itw run (../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw): the only differences are the drafter's FFN width, its number of layers and this folder's paths.
# UNANNEALED on purpose: the learning rate decays linearly over the FULL epoch (382,747 steps) and the run goes on as long as we can afford it (80k steps, then further, as the itw runs did); see NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the 09.21 itw settings and assigns, OUTRIGHT (never `${VAR:-default}` on a variable those settings also use), every path that must be fresh.
ITW_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw" && pwd)"
source "$ITW_EXP/settings.sh"

# THE TARGET: Dr. Doom (the 09.21 itw settings set the 09.21 model; this line replaces it, and the guard stops the run if it is ever not Dr. Doom). MODEL is the model vLLM serves for the hidden states in training, the trainer's
# --verifier-name-or-path (lm_head, gated final norm for the loss, vocabulary) and the one the card names.
MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
[[ "$MODEL" == *-dr-doom ]] || { echo "settings.sh: this experiment's target must be the Dr. Doom checkpoint, got MODEL=$MODEL" >&2; return 1; }
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # the Datakit dataset, READ ONLY here
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # THIS folder's: the differences from the Dr. Doom itw drafter are intermediate_size 11520 and num_hidden_layers 6
WORK_DIR="${DRDOOM_FFN3X_L6_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_ffn3x_l6}"   # fresh: never another run's directory; the override name exists only for a smoke test
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh): the trainer writes its d2t/t2d vocabulary cache here, so the shared dataset is not touched
# Checkpoints every 0.1% of the epoch (383 steps), the last SAVE_TOTAL_LIMIT kept. The comparison points of the itw runs (step 80,813 = 211 x 383 and step 160,860 = 420 x 383) are kept by vista/run_train.sbatch
# (hard links in $WORK_DIR/keep, KEEP_STEPS) so they survive the rotation when training goes past them.
SAVE_TOTAL_LIMIT=8
KEEP_STEPS="80813 160860"
# Derived at source time by the itw settings from the itw run's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (6 layers, FFN 11,520, about 772M trainable parameters) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on the Datakit SFT mixture's own replies, no regeneration) from step 0 with the Dr. Doom hidden states, same recipe as the itw drafter with a 3x wider FFN and twice the layers."
