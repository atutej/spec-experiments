# 6-layer Grug DSpark drafter (taps 2/7/12/17/22, block 8, Markov rank 1024, 32k draft vocabulary) for Dr. Doom, trained itw on the BIG POOL (Open-PerfectBlend + Nemotron-Terminal-Corpus + Nemotron v2),
# to compare with the same architecture on the Datakit SFT mixture (../dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). The big-pool settings (which source the Datakit itw settings) supply the data, 0.5% validation split, 48 workers and the recipe; everything below assigns OUTRIGHT.
BIGPOOL_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw" && pwd)"
source "$BIGPOOL_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
SOURCE_DATA="/scratch/09749/atutej/marin_big_pool/dataset_itw_seq${SEQ_LENGTH}"   # the big pool (Open-PerfectBlend + Nemotron-Terminal-Corpus + Nemotron v2), READ ONLY
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # THIS folder's: 6 layers, FFN 7,936
WORK_DIR="${GRUG_L6T5_BIGMIX_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_grug-layers6-taps5-bigmix}"   # fresh
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh)
# ---- what differs from the parent ----
TARGET_LAYER_IDS="2 7 12 17 22"                  # five evenly spaced taps of the target's 26 layers (residual stream before layer k), a superset of the earlier drafters' 2, 12, 22; launch_vllm.py appends the final output (26)
BLOCK_SIZE=8                                     # tokens drafted per step (the parent's value; the 6- and 10-layer runs used 5)
MARKOV_RANK=1024                                # Markov head rank (the parent chain uses 256): W1 128,256 x 1024 = 131.3M, W2 1024 x 32,000 = 32.8M, 164.1M in all (256: 41.0M); head type stays vanilla
DRAFT_VOCAB_SIZE=32000                           # reduced draft vocabulary built from the dataset's token_freq.pt (as the earlier drafters); the head rows are a frozen copy of the target's
# Full epoch: the parent's STOP_AFTER_STEP is not used here; vista/run_train.sbatch stops when the trainer's epoch0_end checkpoint link exists.
# ---- schedule: the Datakit run's, END TO END. That run is one epoch of 382,747 steps (1% warmup = 3,827 steps, linear decay to zero at step 382,747) and will be run to completion. The big pool has 34,167
# steps per epoch, so this run covers the same 382,747 steps in 12 epochs (11.2 epochs: the loop ends inside epoch 12) with the same horizon. Stopping after any 12 h job (e.g. the 24 h comparison point near
# step 75,834) is only a pause: the learning rate is NOT annealed there (0.81 of the peak), and the next job resumes from the newest checkpoint.
MAX_STEPS=${SMOKE_MAX_STEPS:-382747}
SCHEDULER_TOTAL_STEPS=${SMOKE_SCHED_STEPS:-382747}
EPOCHS=12                                        # 12 x 34,167 = 410,004 >= 382,747; --max-steps ends the loop inside epoch 12 (the epochs only bound the loop)
FINAL_EPOCH_CHECKPOINT="epoch$((EPOCHS - 1))_end"   # epoch11_end: saved when --max-steps ends the loop in epoch 12; vista/run_train.sbatch starts no job once it exists
CKPT_FREQ=0.01121                                # the trainer saves every round(34,167 x 0.01121) = 383 steps, on the Datakit run's grid (37,917 = 99 x 383, 75,834 = 198 x 383)
SAVE_TOTAL_LIMIT=3                               # newest step checkpoints kept (end-of-epoch checkpoints are never pruned); vista/run_train.sbatch copies the newest one into $WORK_DIR/milestones at the start of every job
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO" --draft-layer-type grug --grug-qk-mult 1.75 --num-workers 48 --prefetch-factor 2 --max-steps "$MAX_STEPS" --scheduler-total-steps "$SCHEDULER_TOTAL_STEPS")
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (6 Grug-style layers, FFN 7,936, 32,000-token draft vocabulary built from the big pool's token_freq, taps on target layers 2, 7, 12, 17 and 22, block size 8, Markov rank 1024, 669M trainable parameters) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on Open-PerfectBlend, Nemotron-Terminal-Corpus and Nemotron-Post-Training-Dataset-v2 text, no regeneration) with the learning-rate schedule of the Datakit run (382,747 steps)."
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
# 2026-11-03 (job 1120063 crashed): the two nodes' four trainer ranks compiled the new kernels (new shapes) into the SHARED Triton/Inductor cache on /scratch at once and one read hit `OSError: Stale file handle`.
# The trainer's torchrun gets NODE-LOCAL caches (/tmp is local to each node; two ranks on one node share a local disk safely). Only torchrun: the vLLM server keeps the shared, warm cache.
torchrun() {
    local d="/tmp/${USER}_grug6t5bm_cache_${SLURM_JOB_ID:-nojob}"; mkdir -p "$d/triton" "$d/inductor"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR="$d/triton" TORCHINDUCTOR_CACHE_DIR="$d/inductor" command torchrun "$@"
}
# 2026-11-06 (later): Markov rank 1024 instead of 256.
# 2026-11-06: 6 Grug layers x FFN 7,936, FIVE taps (2, 7, 12, 17, 22 + final), block size 8, 32,000-token draft vocabulary, full-epoch schedule. Untested: that the trainer and the vLLM loader take 5 taps (the first job is the test). Node-local Triton/Inductor caches and expandable segments carried over from the 10-layer run.
# 2026-11-08: created from the Datakit 6-layer folder; data = big pool; schedule = the Datakit run's full 382,747 steps (12 big-pool epochs); pause after any 12 h job by not submitting the next.
