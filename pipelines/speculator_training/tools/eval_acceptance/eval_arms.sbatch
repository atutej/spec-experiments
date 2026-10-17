#!/bin/bash
# Slurm wrapper for serve_eval.sh (one gb node, arms run one after the other). Submit with submit_eval.sh from a login node; the settings travel in the environment
# (MODEL, ARMS, PLAIN, SPEC_EXTRA, PROMPTS_DIR, MAX_PROMPTS, MAX_TOKENS, MAX_MODEL_LEN, SUBSETS, OUT; see serve_eval.sh).
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -A ASC26088
#SBATCH -o /scratch/09749/atutej/marin_speculator/logs/slurm/%x-%j.out
set -uo pipefail
# Make sure the target's weights are in the HF cache first (a no-op when they are; resumes a partial download otherwise).
if [[ -n "${MODEL:-}" ]]; then
    set +u; source ~/.bashrc 2>/dev/null || true   # ~/.bashrc uses unset variables: under `set -u` it aborts the job silently (empty log)
    source /scratch/09749/atutej/marin_speculator/spec-experiments/env.sh; conda activate speculators; set -u
    python - "$MODEL" <<'PY' || exit 1
import os, sys
from huggingface_hub import snapshot_download
print("target weights:", snapshot_download(sys.argv[1], ignore_patterns=["configs/*", "evidence/*", "source/*", "*.png", "*.svg"], max_workers=8, token=os.environ.get("HUG_TOKEN") or None), flush=True)
PY
fi
exec bash /scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/tools/eval_acceptance/serve_eval.sh
