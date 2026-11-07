# itw DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) trained on mlabonne/open-perfectblend instead of the Datakit SFT mix, for 10 epochs.
# Same drafter (3 Qwen3 layers, FFN 3840, 280M parameters), recipe, 4-rank layout and seed as ../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw; the only changes are the data (data/build_perfectblend.py: 1.36M rows,
# 0.687B tokens, rendered and masked like the Datakit mix), EPOCHS=10 and the checkpoint cadence for a 5.3k-step epoch. The end-of-epoch checkpoints (checkpoints/0/ = epoch0_end, 1/, ...) are separate from the step
# checkpoints and are never pruned, so the 1-epoch checkpoint survives the 10-epoch run. See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the parent's settings and assigns, OUTRIGHT (never `${VAR:-default}` on a variable the parent's settings also use), every path that must be fresh.
PARENT_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw" && pwd)"
# Read by the parent's settings with ${VAR:-default}, so they are set BEFORE sourcing it:
TRAIN_DATA_RATIO=0.995      # 0.5% of the rows (about 6.8k) held out: per-epoch validation shows when ten passes start to overfit (the parent holds out 0.1% of a 6.5M-row dataset)
CKPT_FREQ=0.1               # a step checkpoint every 0.1 epoch (about 530 steps, about 5 min); the parent's 0.001 is every 383 steps of a 383k-step epoch but would be every 5 steps here
SAVE_TOTAL_LIMIT=3
source "$PARENT_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"         # the target (as the parent)
SOURCE_DATA="/scratch/09749/atutej/marin_perfectblend/dataset_seq${SEQ_LENGTH}"   # built by data/build_perfectblend.py + finalize_dataset.py, READ ONLY here
EPOCHS=10
WORK_DIR="${PERFECTBLEND_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_perfectblend}"   # fresh: never the parent's runs/grug_dspark_itw_drdoom
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh): the trainer writes its d2t/t2d vocabulary cache here, so the shared dataset is not touched
FINAL_EPOCH_CHECKPOINT="epoch$((EPOCHS - 1))_end"   # vista/run_train.sbatch starts no job once $CKPT_DIR/$FINAL_EPOCH_CHECKPOINT exists
# Dataloader concurrency (job 1109877: 12 workers, the trainer default). A step is ~65 PerfectBlend rows, each fetched with its own hidden-state request inside one worker, so a worker needs ~15 s per batch
# and 12 workers deliver bursts of 12 batches, then a 12-16 s stall: mean step 1.48 s although the median compute step is 0.29 s (the Datakit run's steps hold 3-4 rows, so 12 workers sufficed there).
# 4x the workers = 4x the requests in flight; prefetch 2 keeps the buffered hidden states (workers x prefetch x ~670 MB per rank) in check. Does not change the batches' order.
TRAIN_EXTRA+=(--num-workers 48 --prefetch-factor 2)
# Derived at source time by the parent's settings from the parent's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on mlabonne/open-perfectblend's own replies, no regeneration) for 10 epochs."
# `conda activate` hangs on Vista nodes when /work is slow (2026-10-29: `conda --version` timed out on the login node; conda lives on /work). Redefine the run script's use_env (defined just before this file is sourced) to
# activate the env by hand, which is all conda does for these two envs: PATH, plus the env's etc/conda/activate.d/*.sh (the vllm env's cuda_home.sh sets CUDA_HOME).
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH                      # what `conda deactivate` of the other env would undo
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
