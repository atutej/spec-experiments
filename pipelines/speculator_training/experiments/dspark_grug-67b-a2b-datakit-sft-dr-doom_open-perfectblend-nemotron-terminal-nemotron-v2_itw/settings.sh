# itw DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) trained on the "big dataset" mlabonne/open-perfectblend + nvidia/Nemotron-Terminal-Corpus + nvidia/Nemotron-Post-Training-Dataset-v2
# (English splits chat, code, stem, math, as is: 1,358,785 + 174,064 + 1,396,856 rows, 4.37B tokens) instead of the Datakit SFT mix. Existing text, no regeneration (itw); each row keeps its own reasoning mode: PerfectBlend and Terminal `/nothink`, the v2 rows
# `/think` (with a reasoning block) when their `reasoning` flag is "on" (545k chat rows) and `/nothink` otherwise.
# Same drafter (3 Qwen3 layers, FFN 3840, 280M parameters), recipe, 4-rank layout and seed as ../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw; the changes are the data (data/build_big_pool.py in
# archive_perfectblend_only, `--mode itw`), EPOCHS=2 (about 8.7B tokens, close to the Datakit itw-80k run's 9.7B) and the checkpoint cadence. The end-of-epoch checkpoints (checkpoints/0/ = epoch0_end, 1/)
# are separate from the step checkpoints and are never pruned, so the 1-epoch checkpoint survives. See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the parent's settings and assigns, OUTRIGHT (never `${VAR:-default}` on a variable the parent's settings also use), every path that must be fresh.
PARENT_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw" && pwd)"
# Read by the parent's settings with ${VAR:-default}, so they are set BEFORE sourcing it:
TRAIN_DATA_RATIO=0.995      # 0.5% of the rows (about 7.7k) held out: per-epoch validation shows when repeated passes start to overfit (the parent holds out 0.1% of a 6.5M-row dataset)
CKPT_FREQ=0.03              # a step checkpoint every 0.03 epoch (about 1k steps, about 9 min); the parent's 0.001 is every 383 steps of a 383k-step epoch
SAVE_TOTAL_LIMIT=3
source "$PARENT_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"         # the target (as the parent)
SOURCE_DATA="/scratch/09749/atutej/marin_big_pool/dataset_itw_seq${SEQ_LENGTH}"   # built by data/build_big_pool.py --mode itw + finalize_dataset.py, READ ONLY here
EPOCHS=2                    # about 35k steps per epoch (4.37B tokens): 2 epochs are about 70k steps = about 8.7B tokens (the Datakit itw-80k run: 80,813 steps, 9.7B tokens)
WORK_DIR="${BIG_POOL_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_big_pool}"   # fresh: never the earlier PerfectBlend runs' directories nor the parent's runs/grug_dspark_itw_drdoom
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
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on Open-PerfectBlend, Nemotron-Terminal-Corpus and Nemotron-Post-Training-Dataset-v2 text, no regeneration) for 2 epochs."
# `conda activate` hangs on Vista nodes when /work is slow (2026-10-29: `conda --version` timed out on the login node; conda lives on /work). Redefine the run script's use_env (defined just before this file is sourced) to
# activate the env by hand, which is all conda does for these two envs: PATH, plus the env's etc/conda/activate.d/*.sh (the vllm env's cuda_home.sh sets CUDA_HOME).
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH                      # what `conda deactivate` of the other env would undo
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
