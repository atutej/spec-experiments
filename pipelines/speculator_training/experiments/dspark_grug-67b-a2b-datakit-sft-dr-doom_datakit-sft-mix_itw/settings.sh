# itw DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, an FTPO de-looping fine-tune of the 09.21 Snowball SFT model), trained from step 0 on
# the same text as the existing itw drafter (the replies of the Datakit SFT mix, no regeneration): the ONLY change is the target whose hidden states, lm_head and gated final norm the drafter learns from.
# Sibling of ../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw (same drafter, recipe, data and seed). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the itw settings and assigns, OUTRIGHT (never `${VAR:-default}`, which would keep the itw run's values), every path that must be fresh.
ITW_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw" && pwd)"
source "$ITW_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"   # the target: hidden states in training, lm_head / gated final norm for the TV loss (same architecture, tokenizer and chat template as 09.21)
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # the Datakit dataset, READ ONLY here (the itw run, the targetgen run and the eval prompts use it)
WORK_DIR="${DRDOOM_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom}"     # fresh: never the itw run's runs/grug_dspark_full
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
# DATA_DIR is a VIEW of SOURCE_DATA (symlinked shards + copied metadata, made by vista/make_data_view.sh): the trainer writes its d2t/t2d vocabulary cache into the data directory, and the shared
# Datakit dataset must not be modified by a training run.
DATA_DIR="$WORK_DIR/data"
DRAFT_CONFIG="$ITW_EXP/draft_config.json"

# Derived at source time by the itw settings, from the itw run's LOG_DIR: recompute (the targetgen run wrote its W&B files into the itw run's logs/tracker because of this).
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi

# Budget: the itw run's own (a full-epoch learning-rate schedule, stopped by the 12 h job limit at about step 82k: 0.523 s/step on 2 nodes), so the two drafters differ only in the target.
# To anneal inside the 12 h instead: TOKEN_BUDGET=<tokens> (or MAX_STEPS=<n>, about 78000), exported BEFORE sourcing this file; the trainer decays the rate linearly to zero over exactly that many steps.
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on the Datakit SFT mixture's own replies, no regeneration) from step 0 with the Dr. Doom hidden states."
