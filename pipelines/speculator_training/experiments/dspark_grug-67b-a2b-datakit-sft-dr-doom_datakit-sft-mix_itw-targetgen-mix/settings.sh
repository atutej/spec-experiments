# DSpark drafter for Dr. Doom trained from random initialisation, ONE epoch on ONE shuffled dataset that mixes the two sources: Datakit rows with their original replies (itw, about 4.85B tokens = 6 h of the itw run)
# and Dr. Doom's own regenerated replies to 105k other Datakit rows (targetgen, reused from the warm80k run's regeneration; about 4.5 h of regeneration in the same accounting) = about 12 h aggregate.
# Parent (recipe, drafter, 4-rank layout, seed): ../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw. See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the parent's settings and assigns, OUTRIGHT, every path that must be fresh.
PARENT_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw" && pwd)"
MIX_WORK_DIR="${MIX_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_itw_targetgen_mix_drdoom}"     # fresh; assigned to WORK_DIR AFTER the parent's settings (they set WORK_DIR to the parent's old run directory)
MIX_ARGS=(--itw-tokens 4.85e9 --tg-rows 105000 --val-itw-rows 3000 --val-tg-rows 500 --seed 0)
# Read by the parent's settings with ${VAR:-default}, so they are set BEFORE sourcing it:
TRAIN_DATA_RATIO="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['train_data_ratio'])" "$MIX_WORK_DIR/mixed_dataset/manifest.json" 2>/dev/null || echo 0.9965)"   # makes the validation split exactly the mix's last shard (3.5k rows)
CKPT_FREQ=0.01                                    # a step checkpoint every 1% of the epoch (about 490 steps, about 4 min); the parent's 0.001 would be every 49 steps here
SAVE_TOTAL_LIMIT=3
source "$PARENT_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"         # the target (as the parent)
EPOCHS=1
WORK_DIR="$MIX_WORK_DIR"                         # fresh, assigned OUTRIGHT here (never the parent's runs/grug_dspark_itw_drdoom)
case "$(basename "$WORK_DIR")" in
    grug_dspark_full|grug_dspark_onpolicy|grug_dspark_itw_drdoom|grug_dspark_drdoom_targetgen|grug_dspark_drdoom_targetgen_warm80k|grug_dspark_drdoom_targetgen_210k|grug_dspark_itw_rlsimilar|grug_dspark_itw_ffn3x|grug_dspark_itw_drdoom_perfectblend|grug_dspark_itw_drdoom_big_pool|grug_dspark_drdoom_big_pool_targetgen) echo "settings.sh: WORK_DIR $WORK_DIR is an old run's directory" >&2; return 1 2>/dev/null || exit 1 ;;
esac
SOURCE_DATA="$WORK_DIR/mixed_dataset"            # built by vista/run_mix.sh (data/build_mix.py) in the train job; symlinks to the Datakit shards, so READ ONLY there
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
DATA_DIR="$WORK_DIR/data"                        # a VIEW of SOURCE_DATA (vista/make_data_view.sh): the trainer writes its d2t/t2d vocabulary cache here (from the Datakit token_freq.pt, i.e. the itw runs' vocabulary)
FINAL_EPOCH_CHECKPOINT="epoch$((EPOCHS - 1))_end"   # vista/run_train.sbatch starts no job once $CKPT_DIR/$FINAL_EPOCH_CHECKPOINT exists
# Derived at source time by the parent's settings from the parent's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained from random initialisation for one epoch on a shuffled mix of Datakit SFT rows (original replies) and Dr. Doom's own replies to 105k other Datakit rows."
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH                      # what `conda deactivate` of the other env would undo
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
