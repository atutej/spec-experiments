# TARGETGEN DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom), WARM-STARTED from the Dr. Doom-trained itw drafter at step 80,813 and fine-tuned on Dr. Doom's OWN replies.
# Same pipeline as ../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen (regen -> merge -> train), with three differences: the target (rollouts, hidden states, lm_head) is Dr. Doom, not 09.21; training starts from
# the itw checkpoint instead of random weights; and the rows are sampled from the Datakit rows the itw run had NOT seen by step 80,813 (those of its steps 80,813-160,860). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the itw settings and assigns, OUTRIGHT, every path that must be fresh.
ITW_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw" && pwd)"
source "$ITW_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"   # the target: rollouts (regen), hidden states, lm_head / gated final norm. NOT the 09.21 model.
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # the Datakit rows (token ids + loss mask) whose conversations are replayed; READ ONLY here
DRAFT_CONFIG=""                                   # EMPTY on purpose: --from-pretrained loads the whole draft model (3 layers, FFN 3840: the checkpoint's own architecture) and the trainer refuses it together with
                                                  # --draft-config; the itw run scripts omit the flag when this is empty (the smoke test of 2026-10-19 hit exactly this error)
WARM_START="$PROJECT_ROOT/runs/eval_ckpts/drdoom_itw_step80813"     # the Dr. Doom-trained itw drafter at step 80,813 (model.safetensors, config.json, config.py); read only

# ---- the fresh work dir (never runs/grug_dspark_itw_drdoom, which holds the itw run's checkpoints, nor any other run's) ----
WORK_DIR="${DRDOOM_TG_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_drdoom_targetgen_warm80k}"
PARTS_DIR="$WORK_DIR/regen_parts"
DATA_DIR="$WORK_DIR/data"                         # the merged dataset of Dr. Doom's replies (+ token_freq.pt + the CHECKPOINT's draft-vocabulary maps)
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
REGEN_PARTS_SRC=""                                # no parts to import from another work dir

# ---- the rows: 250k sampled uniformly from the rows of steps [80,813, 160,860) of the itw run's batch order (about 1.37M rows), none of which the 0-80,813 steps saw ----
POOL_START_STEP=80813
POOL_END_STEP=160860
POOL_FILE="$WORK_DIR/pool_rows_steps${POOL_START_STEP}-${POOL_END_STEP}.npy"
SEEN_FILE="$PROJECT_ROOT/runs/itw_seen_rows/rows_step80813.npy"     # the rows of steps 0-80,813 (from itw_seen_rows.py): the disjointness check
REGEN_ROWS=${REGEN_ROWS:-250000}
REGEN_SEED=${REGEN_SEED:-0}
REGEN_CONCURRENCY=${REGEN_CONCURRENCY:-512}          # requests in flight per node (= the server's --max-num-seqs)
REGEN_MAX_NEW_TOKENS=${REGEN_MAX_NEW_TOKENS:-32768} # per assistant turn; the row's 32,768-token window is the real limit
REGEN_MODEL_LEN=$SEQ_LENGTH
# Estimate, not a measurement for Dr. Doom: on 09.21, 270k rows took about 8.7 h on 2 nodes (4,413 generated tokens per row) and training on them about 2.6 h. Dr. Doom's replies are longer (about 1.3x on the RL prompts),
# so 250k rows should take about 9.5 to 11 h of regeneration plus about 2.4 h of training: a little over 12 h in total.

# ---- training: one epoch over the regenerated rows, starting from the checkpoint (a fresh optimizer and a fresh schedule: linear decay to zero over this epoch) ----
# (the DRDOOM_TG_* names exist only for vista/smoke.sh; no other experiment uses them, so they cannot be inherited by accident)
TRAIN_DATA_RATIO=${DRDOOM_TG_TRAIN_DATA_RATIO:-0.995}   # 0.5% held out for the end-of-run validation
CKPT_FREQ=${DRDOOM_TG_CKPT_FREQ:-0.03}                  # a checkpoint every 3% of the epoch
MAX_STEPS="${DRDOOM_TG_MAX_STEPS:-}"                    # empty: one epoch (not annealed early, not stopped early)
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO" --from-pretrained "$WARM_START")
[[ -z "$MAX_STEPS" ]] || TRAIN_EXTRA+=(--max-steps "$MAX_STEPS")
# Derived at source time by the itw settings from the itw run's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom: the Dr. Doom-trained itw drafter (step 80,813) fine-tuned on Dr. Doom's own replies to Datakit SFT conversations (targetgen)."
