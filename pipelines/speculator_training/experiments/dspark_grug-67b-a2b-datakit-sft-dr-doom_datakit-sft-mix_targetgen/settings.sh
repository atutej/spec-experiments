# TARGETGEN DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom), trained FROM RANDOM INITIALISATION on Dr. Doom's OWN replies only (no itw text anywhere in the data).
# Same chain as ../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen (regen -> merge -> train) with the target changed to Dr. Doom and 400k rows instead of 270k (about 24 h of regen + training in total). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the itw settings and assigns, OUTRIGHT, every path that must be fresh.
ITW_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw" && pwd)"
source "$ITW_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"   # the target: rollouts (regen), hidden states, lm_head / gated final norm. NOT the 09.21 model.
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # the Datakit rows (token ids + loss mask) whose conversations are replayed; READ ONLY here

# ---- the fresh work dir: assigned outright, never an environment default (the old runs' dirs are guarded below) ----
WORK_DIR="$PROJECT_ROOT/runs/grug_dspark_drdoom_targetgen"
case "$(basename "$WORK_DIR")" in
    grug_dspark_full|grug_dspark_onpolicy|grug_dspark_itw_drdoom|grug_dspark_drdoom_targetgen_warm80k|grug_dspark_itw_rlsimilar|grug_dspark_itw_ffn3x) echo "settings.sh: WORK_DIR $WORK_DIR is an old run's directory" >&2; return 1 2>/dev/null || exit 1 ;;
esac
PARTS_DIR="$WORK_DIR/regen_parts"
DATA_DIR="$WORK_DIR/data"                         # the merged dataset of Dr. Doom's replies (+ token_freq.pt + the draft-vocabulary maps made from them)
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
REGEN_PARTS_SRC=""                                # no parts to import from another work dir
DRAFT_CONFIG="$ITW_EXP/draft_config.json"         # random initialisation: the 3-layer, FFN 3840 architecture (no --from-pretrained)

# ---- the rows: 400k sampled uniformly without replacement from the Datakit train split (6.2% of it) ----
REGEN_ROWS=400000
REGEN_SEED=0
REGEN_CONCURRENCY=512                             # requests in flight per node (= the server's --max-num-seqs)
REGEN_MAX_NEW_TOKENS=32768                        # per assistant turn; the row's 32,768-token window is the real limit
REGEN_MODEL_LEN=$SEQ_LENGTH
# Estimate from the warm80k run (Dr. Doom, 250k rows: 1.54B generated tokens, 6,168 per row, 10.6 h on 2 nodes = 23.6k rows/h, 2.38B merged tokens = 9.5k per row): 400k rows are about 17 h of regeneration,
# about 3.8B tokens = about 31.7k steps of 120k tokens = about 4.6 h of training at 0.523 s/step, plus about 2 h of startup / merge / queue gaps: about 24 h in all.

# ---- training: one epoch over the regenerated rows from random weights (the learning rate decays linearly to zero over this epoch) ----
TRAIN_DATA_RATIO=0.995                            # 0.5% held out for the end-of-run validation
CKPT_FREQ=0.03                                    # a checkpoint every 3% of the epoch (about 950 steps)
MAX_STEPS=""                                      # empty: one epoch (not annealed early, not stopped early)
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO")
[[ -z "$MAX_STEPS" ]] || TRAIN_EXTRA+=(--max-steps "$MAX_STEPS")
# Derived at source time by the itw settings from the itw run's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained from random initialisation on Dr. Doom's own replies to Datakit SFT conversations (targetgen only)."
