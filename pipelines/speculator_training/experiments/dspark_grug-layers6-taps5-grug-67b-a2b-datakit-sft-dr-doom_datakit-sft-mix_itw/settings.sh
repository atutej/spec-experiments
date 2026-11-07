# itw DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom): 6 Grug layers, dense shared-expert FFN 7,936, hidden 2,560 (20 query / 5 KV heads, head size 128, full attention),
# 32,000-token draft vocabulary (frozen copy of the target's head rows, chosen from token_freq.pt as in the earlier drafters; Markov head rank 256), taps on target layers 2, 7, 12, 17, 22 (the earlier drafters' 2, 12, 22 plus two between), block size 8.
# Trained for ONE FULL EPOCH of the Datakit SFT mixture (382,747 steps, about 46B tokens), learning rate decayed linearly over the epoch (the original itw recipe). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the Grug-layers settings (Dr. Doom target, Datakit data, recipe, seed, `--draft-layer-type grug --grug-qk-mult 1.75`) and assigns, OUTRIGHT, every path that must be fresh.
PARENT_EXP="$(cd "$EXP_DIR/../dspark-grug-layers_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw" && pwd)"
source "$PARENT_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # READ ONLY
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # THIS folder's: 6 layers, FFN 7,936
WORK_DIR="${GRUG_L6T5_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_grug-layers6-taps5}"   # fresh
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh)
# ---- what differs from the parent ----
TARGET_LAYER_IDS="2 7 12 17 22"                  # five evenly spaced taps of the target's 26 layers (residual stream before layer k), a superset of the earlier drafters' 2, 12, 22; launch_vllm.py appends the final output (26)
BLOCK_SIZE=8                                     # tokens drafted per step (the parent's value; the 6- and 10-layer runs used 5)
MARKOV_RANK=1024                                # Markov head rank (the parent chain uses 256): W1 128,256 x 1024 = 131.3M, W2 1024 x 32,000 = 32.8M, 164.1M in all (256: 41.0M); head type stays vanilla
DRAFT_VOCAB_SIZE=32000                           # reduced draft vocabulary built from the dataset's token_freq.pt (as the earlier drafters); the head rows are a frozen copy of the target's
# Full epoch: the parent's STOP_AFTER_STEP is not used here; vista/run_train.sbatch stops when the trainer's epoch0_end checkpoint link exists.
unset MAX_STEPS 2>/dev/null; MAX_STEPS=""
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO" --draft-layer-type grug --grug-qk-mult 1.75)   # no --max-steps: the trainer runs the epoch and decays the learning rate over it
SAVE_TOTAL_LIMIT=3                               # newest step checkpoints kept; vista/run_train.sbatch copies the newest one into $WORK_DIR/milestones at the start of every job
CKPT_FREQ=0.001                                  # a checkpoint every 0.1% of the epoch (383 steps)
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (6 Grug-style layers, FFN 7,936, 32,000-token draft vocabulary (the Datakit token_freq selection, as the earlier drafters), taps on target layers 2, 7, 12, 17 and 22, block size 8, Markov rank 1024, 669M trainable parameters (excluding the frozen embedding, draft head and verifier head copy; 1.16B counting them)) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on the Datakit SFT mixture's own replies, no regeneration) for one epoch."
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
# 2026-11-03 (job 1120063 crashed): the two nodes' four trainer ranks compiled the new kernels (new shapes) into the SHARED Triton/Inductor cache on /scratch at once and one read hit `OSError: Stale file handle`.
# The trainer's torchrun gets NODE-LOCAL caches (/tmp is local to each node; two ranks on one node share a local disk safely). Only torchrun: the vLLM server keeps the shared, warm cache.
torchrun() {
    local d="/tmp/${USER}_grug6t5_cache_${SLURM_JOB_ID:-nojob}"; mkdir -p "$d/triton" "$d/inductor"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR="$d/triton" TORCHINDUCTOR_CACHE_DIR="$d/inductor" command torchrun "$@"
}
# 2026-11-06 (later): Markov rank 1024 instead of 256.
# 2026-11-06: 6 Grug layers x FFN 7,936, FIVE taps (2, 7, 12, 17, 22 + final), block size 8, 32,000-token draft vocabulary, full-epoch schedule. Untested: that the trainer and the vLLM loader take 5 taps (the first job is the test). Node-local Triton/Inductor caches and expandable segments carried over from the 10-layer run.
