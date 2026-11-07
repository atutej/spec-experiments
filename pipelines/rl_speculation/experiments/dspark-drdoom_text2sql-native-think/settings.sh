# RL smoke, split layout, multi-turn text-to-SQL with an outcome-only reward (native thinking template), Dr. Doom target + targetgen-warm80k DSpark drafter. Writes only under WORK_DIR.
TARGET="$PROJECT_ROOT/cache/hf/hub/models--open-athena--Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom/snapshots/fd4a272b78d4c224851aaeae7f51987c5d19649c"
DRAFT="$PROJECT_ROOT/runs/eval_ckpts/drdoom_targetgen_warm80k_final"   # read-only here
WORK_DIR="$PROJECT_ROOT/runs/rl_speculation/drdoom_text2sql_native_think_20261102"
