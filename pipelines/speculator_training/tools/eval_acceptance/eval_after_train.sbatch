#!/bin/bash
# Evaluate the LATEST complete checkpoint of a training run with the usual two Dr. Doom evals, one after the other on one gb node (submit it behind the training job: --dependency=afterany:<job>, run from a LOGIN node).
#   CKPTS=<run>/checkpoints NAME=<eval name> [MODEL=<target>] [DRY=1] sbatch --dependency=afterany:<train job> -p gb --reservation=CGAI-gb -t 03:00:00 -J eval-<name> --export=ALL eval_after_train.sbatch
# Copies checkpoint-<N> (the highest N; a `.tmp` directory is a half-written save and is skipped) to runs/eval_ckpts/<NAME>_step<N> (config.json, config.py, model.safetensors), then runs
#   1. RLVR: the first 256 prompts of runs/nemotron_rlvr_prompts_1024, up to 49,152 new tokens, 102,400-token window      -> runs/serve_eval_nemotron_rlvr_<NAME>
#   2. Datakit held-out multi-turn 1,024 (833 first-turn + 191 later-turn), up to 4,096 new tokens, 32,768-token window  -> runs/serve_eval_val_multiturn1024_<NAME>
# with T=1.0, probabilistic draft sampling, 8 drafted tokens: the settings of the earlier Dr. Doom evals. No plain arm (add PLAIN=1 for same-session speedups). DRY=1 only prints what it would do.
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator; HERE=$ROOT/spec-experiments/pipelines/speculator_training/tools/eval_acceptance
: "${CKPTS:?CKPTS=<training run>/checkpoints}"; : "${NAME:?NAME=<eval name>}"; MODEL=${MODEL:-open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom}
last=$(ls -d "$CKPTS"/checkpoint-* 2>/dev/null | grep -v '\.tmp$' | sed 's#.*/checkpoint-##' | sort -n | tail -1)
[[ -n "$last" ]] || { echo "no complete checkpoint in $CKPTS" >&2; exit 1; }
SRC="$CKPTS/checkpoint-$last"; E="$ROOT/runs/eval_ckpts/${NAME}_step$last"
echo "target $MODEL | latest complete checkpoint: $SRC (step $last) -> $E"
python3 - "$SRC/config.json" <<'PY' || exit 1
import json, sys
c = json.load(open(sys.argv[1])); v = c["speculators_config"]["verifier"]["name_or_path"]
print("checkpoint verifier:", v, "| layers", c["transformer_layer_config"]["num_hidden_layers"], "| FFN", c["transformer_layer_config"]["intermediate_size"], "| block", c["block_size"])
PY
[[ "${DRY:-0}" != 1 ]] || { echo "DRY=1: stopping before the copy and the evals"; exit 0; }
[[ ! -e "$E" ]] || { echo "refusing: $E exists" >&2; exit 1; }
mkdir -p "$E" && cp "$SRC/config.json" "$SRC/config.py" "$SRC/model.safetensors" "$E/" && cmp "$SRC/model.safetensors" "$E/model.safetensors" && echo "copied, identical" || exit 1
set +u; source ~/.bashrc 2>/dev/null || true; source "$ROOT/spec-experiments/env.sh"; conda activate speculators; set -u
python - "$MODEL" <<'PY' || exit 1
import os, sys
from huggingface_hub import snapshot_download
print("target weights:", snapshot_download(sys.argv[1], ignore_patterns=["configs/*", "evidence/*", "source/*", "*.png", "*.svg"], max_workers=8, token=os.environ.get("HUG_TOKEN") or None), flush=True)
PY
export MODEL PLAIN=${PLAIN:-} ARMS="${NAME}=$E" SPEC_EXTRA=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"'
[[ -n "${PLAIN:-}" ]] || unset PLAIN
PROMPTS_DIR=$ROOT/runs/nemotron_rlvr_prompts_1024 MAX_PROMPTS=256 MAX_TOKENS=49152 MAX_MODEL_LEN=102400 SUBSETS=first_turn OUT=$ROOT/runs/serve_eval_nemotron_rlvr_$NAME bash "$HERE/serve_eval.sh"; echo "RLVR eval exit $?"
PROMPTS_DIR=$ROOT/runs/val_prompts_multiturn_1024 MAX_TOKENS=4096 MAX_MODEL_LEN=32768 SUBSETS="first_turn later_turn" OUT=$ROOT/runs/serve_eval_val_multiturn1024_$NAME bash "$HERE/serve_eval.sh"; echo "Datakit eval exit $?"
