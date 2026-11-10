#!/bin/bash
# Rebuilds the masked prompts of SPEED-Bench's qualitative split with NVIDIA's own script (Model-Optimizer, examples/specdec_bench/prepare_data.py).
# It downloads public Hugging Face datasets (and cais/hle, which is gated: it needs HUG_TOKEN with accepted terms) and one pinned MT-Bench-101 file from GitHub.
# Output: /scratch/09749/atutej/marin_speed_bench/prepared/speed/qualitative/   Log: prepare_qualitative.log in the same folder as this script.
set -uo pipefail
source ~/.bashrc 2>/dev/null
HERE=/scratch/09749/atutej/marin_speed_bench
export HF_TOKEN="${HUG_TOKEN:?HUG_TOKEN is not set}" HF_HOME="$HERE/hf"
cd "$HERE/src/examples/specdec_bench" || exit 1
/scratch/09749/atutej/marin_speculator/envs/sft/bin/python prepare_data.py --dataset speed --config qualitative --output_dir "$HERE/prepared" 2>&1 | tr '\r' '\n' | grep -v "examples/s\]\|files/s\]\|it/s\]" | tee "$HERE/prepare_qualitative.log"
echo "exit status of prepare_data.py: ${PIPESTATUS[0]}"
find "$HERE/prepared" -type f | head
