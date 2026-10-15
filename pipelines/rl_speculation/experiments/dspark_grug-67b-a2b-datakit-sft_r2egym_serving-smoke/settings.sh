# Serving smoke test: Grug-67B-A2B-Datakit-SFT-262K with and without our DSpark drafter, on one 4-GPU GB200 node, no RL. Sourced after env.sh.
# Override from the environment, e.g. MAX_PROMPTS=8 MAX_TOKENS=512 bash vista/run.sh dspark
MODEL="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
# The drafter: a local checkpoint dir (speculators format, config.json + weights). The published copy is
# atutej/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_offpolicy-20261014 (download it and point DRAFT at the folder).
DRAFT=${DRAFT:-$PROJECT_ROOT/runs/grug_dspark_100k/checkpoints/0}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-8}            # the drafter's block size
# The fork's Grug GPU path needs TP=1; DP x expert-parallel over all GPUs, like the RL engines (one DP4 x EP4 engine per node).
SERVE_DP=${SERVE_DP:-4}
PORT=8000
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}
# Prompts: R2E-Gym instructions through the target's chat template (tools/build_r2egym_prompts.py); first turns only.
TASKS_DIR=${TASKS_DIR:-/scratch/09749/atutej/tasks/repo-29-DCAgent%2Fr2egym-patched-full__revision-40-cb071d2678ba02ccc7c591181788546b478bd0ad}
PROMPTS_DIR=${PROMPTS_DIR:-$PROJECT_ROOT/runs/r2egym_prompts_128}
MAX_PROMPTS=${MAX_PROMPTS:-128}
MAX_TOKENS=${MAX_TOKENS:-4096}
CONCURRENCY=${CONCURRENCY:-16}                   # Luke's per-node figure was 16 streams
WORK_DIR=${WORK_DIR:-$PROJECT_ROOT/runs/rl_speculation/serving_smoke}
