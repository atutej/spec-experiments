#!/bin/bash
# Replay the contexts served in the crashed RL job (1109533) against the Dr. Doom target with each drafter; accept length by context length. Run on a 4-GPU gb node (idev ok).
# Needs prompts from build_replay_prompts.py. Results under $OUT/<label>/ (acceptance.csv has one row per subset). Reuses serve_eval.sh read-only.
#   bash run_replay.sh            (A: RL's own draft settings; B: probabilistic drafting; C: Grug-layer drafter, needs the vLLM overlay)
set -uo pipefail
ROOT=/scratch/09749/atutej/marin_speculator
BASE=$ROOT/runs/rl_speculation/rollout_replay_drdoom_20261029
export MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom
export PROMPTS_DIR=$BASE/prompts SUBSETS="turn0 ctx_0_4k ctx_4_8k ctx_8_16k ctx_16_24k" MAX_TOKENS=${MAX_TOKENS:-2048} TEMPERATURE=1.0 MAX_MODEL_LEN=32768
SE=$ROOT/spec-experiments/pipelines/speculator_training/tools/eval_acceptance/serve_eval.sh
WARM=$ROOT/runs/eval_ckpts/drdoom_targetgen_warm80k_final; ITW=$ROOT/runs/eval_ckpts/drdoom_itw_step80813; GRUG=$ROOT/runs/eval_ckpts/drdoom_grug_layers_step76600
PROB=', "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"'
# A: exactly the speculative_config of the RL job (method, model, 8 tokens; no sampling-mode fields)
OUT=$BASE/A_rl_settings ARMS="warm80k=$WARM" SPEC_EXTRA="" bash $SE
# B: same drafter with probabilistic drafting, the setting our earlier evals used at T=1
OUT=$BASE/B_probabilistic ARMS="warm80k=$WARM itw80k=$ITW" SPEC_EXTRA="$PROB" bash $SE
# C: Grug-layer drafter (overlay of the grug-dspark-layers branch over the vllm env's wheel)
PYTHONPATH=$ROOT/tmp/grug_dspark_serving_smoke_2026-10-27/vllm_overlay OUT=$BASE/C_grug ARMS="grug76600=$GRUG" SPEC_EXTRA="$PROB" bash $SE
