# itw DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom) with GRUG layers instead of Qwen3 layers (3 layers, shared-expert width 3,664: the same 137.7M decoder-layer parameters as the Qwen3 layers' FFN 3,840),
# trained from step 0 on the SAME data, recipe, seed and full-epoch learning-rate schedule as ../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw (its checkpoint at step 80,813 is the baseline: the 12 h job's last). See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the parent's settings and assigns, OUTRIGHT (never `${VAR:-default}` on a variable the parent's settings also use), every path that must be fresh.
PARENT_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw" && pwd)"
source "$PARENT_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"         # the target (as the parent)
SOURCE_DATA="/scratch/09749/atutej/marin_datakit/dataset_seq${SEQ_LENGTH}"   # the Datakit dataset, READ ONLY here
DRAFT_CONFIG="$EXP_DIR/draft_config.json"        # THIS folder's: intermediate_size 3664 is the Grug shared-expert width; partial_rotary_factor 0.5 is Grug's half-RoPE
WORK_DIR="${GRUG_LAYERS_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_drdoom_grug-layers}"   # fresh: never the parent's runs/grug_dspark_itw_drdoom
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh): the trainer writes its d2t/t2d vocabulary cache here, so the shared dataset is not touched
# Checkpoints every 0.1% of the epoch (383 steps) as in the parent; keep more of them so the checkpoint at step 80,813 (211 x 383, the parent's last one) survives if training runs past it.
SAVE_TOTAL_LIMIT=8
STOP_AFTER_STEP=80813                            # the comparison point; vista/run_train.sbatch does not start a job once a checkpoint at or past this step exists
# The only change to the training command: Grug layers (speculators `DSparkSpeculatorConfig.layer_type="grug"`; full attention, half-RoPE from the draft config, qk_mult 1.75, XSA, output gate, gated norms, shared-expert FFN).
TRAIN_EXTRA+=(--draft-layer-type grug --grug-qk-mult 1.75)
# Derived at source time by the parent's settings from the parent's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 Grug-style layers, shared-expert width 3,664, about 280M parameters) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained itw (on the Datakit SFT mixture's own replies, no regeneration), same recipe as the Dr. Doom itw drafter with parameter-matched Grug layers instead of Qwen3 layers."
# `conda activate` hangs on Vista nodes (2026-10-29: `conda shell.posix activate` idle for 16 min on both gb nodes, `conda --version` times out on the login node; conda lives on /work).
# Redefine the run script's use_env (defined just before this file is sourced) to activate the env by hand, which is all conda does for these two envs: PATH, plus the env's etc/conda/activate.d/*.sh (the vllm env's cuda_home.sh sets CUDA_HOME).
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH                      # what `conda deactivate` of the other env would undo
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
