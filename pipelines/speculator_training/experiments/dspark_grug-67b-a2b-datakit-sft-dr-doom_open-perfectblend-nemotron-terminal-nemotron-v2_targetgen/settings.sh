# TARGETGEN DSpark drafter for the Dr. Doom target (open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom), trained FROM RANDOM INITIALISATION on Dr. Doom's OWN replies, in /think mode, to prompts sampled from the "big dataset"
# mlabonne/open-perfectblend + nvidia/Nemotron-Terminal-Corpus + nvidia/Nemotron-Post-Training-Dataset-v2 (English splits, as is: 2,929,705 rows, 4.37B tokens).
# Same chain, drafter (3 Qwen3 layers, FFN 3840, 280M parameters) and recipe as ../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen (regen -> merge -> train); only the prompts differ. See NOTES.md.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). It starts from the parent's settings and assigns, OUTRIGHT (never `${VAR:-default}` on a variable the parent's settings also use), every path that must be fresh.
PARENT_EXP="$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen" && pwd)"
source "$PARENT_EXP/settings.sh"

MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"          # the target: rollouts (regen), hidden states, lm_head / gated final norm
# Every row starts with the system header `Reasoning: /think`, the model's own default at serving (its inference chat_template.jinja) and the header of every eval prompt. Built by
# ../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_itw/data/build_big_pool.py --mode think (then finalize_dataset.py): PerfectBlend and v2 rendered with the Marin template (v2 reasoning rows with their `<|start_think|>` block), the Terminal rows are the
# Datakit pipeline's own normalized rows with their `/nothink` header changed to `/think`. The original replies in it are discarded: regen replaces every assistant turn with Dr. Doom's own (tool outputs and later user turns are kept as they are).
SOURCE_DATA="/scratch/09749/atutej/marin_big_pool/dataset_think_seq${SEQ_LENGTH}"   # READ ONLY here
WORK_DIR="$PROJECT_ROOT/runs/grug_dspark_drdoom_big_pool_targetgen"              # fresh, assigned outright
case "$(basename "$WORK_DIR")" in
    grug_dspark_full|grug_dspark_onpolicy|grug_dspark_itw_drdoom|grug_dspark_drdoom_targetgen|grug_dspark_drdoom_targetgen_warm80k|grug_dspark_itw_rlsimilar|grug_dspark_itw_ffn3x|grug_dspark_itw_drdoom_perfectblend|grug_dspark_itw_drdoom_pblend_terminal|grug_dspark_itw_drdoom_big_pool|grug_dspark_drdoom_pblend_terminal_targetgen) echo "settings.sh: WORK_DIR $WORK_DIR is an old run's directory" >&2; return 1 2>/dev/null || exit 1 ;;
esac
PARTS_DIR="$WORK_DIR/regen_parts"
DATA_DIR="$WORK_DIR/data"                         # the merged dataset of Dr. Doom's replies (+ token_freq.pt + the draft-vocabulary maps made from them)
CKPT_DIR="$WORK_DIR/checkpoints"
LOG_DIR="$WORK_DIR/logs"
REGEN_PARTS_SRC=""                                # no parts to import from another work dir
# ---- the rows: sampled uniformly without replacement from the train split of the big dataset (the first 99.9% of its 2,929,705 rows) ----
# Pilots (replies capped at 12,288 tokens, one node): PerfectBlend prompts only (runs/targetgen_pblend_pilot_2026-10-30, 384 rows): 5,292 generated tokens per row, 14% of the rows capped; the big pool, 512 rows sampled uniformly
# (runs/targetgen_pool_pilot_2026-10-30): 6,162 generated tokens per row, median row 4,624 tokens, 27% of the rows capped, 0 errors. The real cap is 32,768, so the full-cap mean is probably 8-11k generated tokens per row (the Datakit regeneration: 6.2k per row,
# 23.6k rows/h on 2 nodes = about 40k generated tokens/s). 180k rows (at about 9k generated tokens per row): about 1.62B generated tokens = about 11.3 h of regeneration (close to one 12 h job: the chained second regen job covers the rest), about 1.66B merged tokens = about 13.7k steps = about 2 h of training, plus about 1.5 h of merge,
# startup and queue gaps: about 15 h in all (13-18 h for 8-11k tokens per row). 150k rows would be about 12.5 h in all, 200k rows about 16-17 h, 100k rows about 9 h. Estimates, see NOTES.md.
REGEN_ROWS=${PBLEND_REGEN_ROWS:-180000}
REGEN_SEED=0
REGEN_CONCURRENCY=512                             # requests in flight per node (= the server's --max-num-seqs)
REGEN_MAX_NEW_TOKENS=32768                        # per assistant turn; the row's 32,768-token window is the real limit
REGEN_MODEL_LEN=$SEQ_LENGTH
# ---- training: one epoch over the regenerated rows from random weights (the learning rate decays linearly to zero over this epoch) ----
TRAIN_DATA_RATIO=0.995                            # 0.5% held out for the end-of-run validation
CKPT_FREQ=0.03                                    # a checkpoint every 3% of the epoch
MAX_STEPS=""                                      # empty: one epoch (not annealed early, not stopped early)
TRAIN_EXTRA=(--train-data-ratio "$TRAIN_DATA_RATIO")
[[ -z "$MAX_STEPS" ]] || TRAIN_EXTRA+=(--max-steps "$MAX_STEPS")
# Derived at source time by the parent's settings from the parent's LOG_DIR: recompute.
LOGGER_ARGS=()
if [[ -n "$LOGGER" ]]; then LOGGER_ARGS=(--logger "$LOGGER" --run-name "$RUN_NAME" --log-dir "$LOG_DIR/tracker"); fi
HF_DESCRIPTION="DSpark drafter (3 layers, FFN 3840) for open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom, trained from random initialisation on Dr. Doom's own /think replies to Open-PerfectBlend, Nemotron-Terminal-Corpus and Nemotron-Post-Training-Dataset-v2 prompts (targetgen only)."
# `conda activate` hangs on Vista nodes when /work is slow (2026-10-29: `conda --version` timed out on the login node; conda lives on /work). Redefine the scripts' use_env (defined just before this file is sourced) to
# activate the env by hand, which is all conda does for these two envs: PATH, plus the env's etc/conda/activate.d/*.sh (the vllm env's cuda_home.sh sets CUDA_HOME).
_BASE_PATH="$PATH"
use_env() {
    unset CUDA_HOME _VLLM_OLD_PATH                      # what `conda deactivate` of the other env would undo
    export CONDA_PREFIX="$PROJECT_ROOT/envs/$1" PATH="$PROJECT_ROOT/envs/$1/bin:$_BASE_PATH"
    set +u; for _f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [[ -f "$_f" ]] && source "$_f"; done; set -u
}
