# RL smoke, split layout, Dr. Doom target + Grug-layer DSpark drafter. Sourced after env.sh. Nothing here touches older runs: everything is written under WORK_DIR.
TARGET="$PROJECT_ROOT/cache/hf/hub/models--open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom/snapshots/fd4a272b78d4c224851aaeae7f51987c5d19649c"
DRAFT="$PROJECT_ROOT/runs/eval_ckpts/drdoom_grug_layers_step76600"   # read-only here
WORK_DIR="$PROJECT_ROOT/runs/rl_speculation/drdoom_grug_layers_split_20261029"
OVERLAY="$WORK_DIR/vllm_overlay"   # the grug-dspark-layers changes over the skyrl env's vLLM wheel (PYTHONPATH)
