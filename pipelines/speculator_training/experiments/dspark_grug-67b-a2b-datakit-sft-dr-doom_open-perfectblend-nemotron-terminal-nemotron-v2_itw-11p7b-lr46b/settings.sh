# Longer itw DSpark drafter for the Dr. Doom target on the big dataset (Open-PerfectBlend + Nemotron-Terminal-Corpus + Nemotron-Post-Training-Dataset-v2 English splits, existing text, no regeneration).
# Sibling of ../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw (2 epochs, 8.7B tokens, learning rate annealed to zero over those 2 epochs). Same data, drafter, recipe and seed;
# the changes are the budget and the learning-rate schedule, chosen for an 11 h job: stop at about 11.7B tokens (92,000 steps), with a linear schedule whose horizon is about 46B tokens (the Datakit itw-80k run's schedule), so the rate has
# only decayed to about 0.75 of its peak when the run stops (the Datakit run: a full-epoch schedule stopped by its 12 h limit at step 80,813, 9.7B tokens). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the big-pool itw settings and assigns, OUTRIGHT, every path that must be fresh.
BIGPOOL_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw" && pwd)"
# Tokens per step on this dataset: 4.35B training tokens in 34,167 steps of one epoch (job 1111721) = about 127.3k per step.
MAX_STEPS=${SMOKE_MAX_STEPS:-92000}                 # about 11.7B tokens (92,000 x 127.3k, about 11 h; SMOKE_MAX_STEPS only for a smoke test): read by the Datakit itw settings, which add `--max-steps` to the trainer's arguments; the loop stops here
SCHEDULER_TOTAL_STEPS=${SMOKE_SCHED_STEPS:-361000}    # about 46B tokens (361,000 x 127.3k): the horizon of the linear schedule (1% warmup = 3,610 steps, then a decay to zero at this step)
source "$BIGPOOL_EXP/settings.sh"
MAX_STEPS=${SMOKE_MAX_STEPS:-92000}                 # (re)assigned: the sourced settings must not change it
TRAIN_EXTRA+=(--scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
EPOCHS=3                        # 34,167 steps per epoch: 92,000 steps are 2.69 epochs, so the loop ends inside the third epoch; the epochs only bound the loop, `--max-steps` stops it
WORK_DIR="${BIG_POOL_11P7B_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_big_pool_11p7b}"   # fresh: never the 2-epoch big-pool run's runs/grug_dspark_itw_drdoom_big_pool
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"
FINAL_EPOCH_CHECKPOINT="epoch$((EPOCHS - 1))_end"   # the trainer saves it when --max-steps ends the loop inside the last epoch; vista/run_train.sbatch starts no job once it exists
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on Open-PerfectBlend, Nemotron-Terminal-Corpus and Nemotron-Post-Training-Dataset-v2 text, no regeneration) for about 11.7B tokens with a learning-rate schedule that decays over about 46B tokens."
