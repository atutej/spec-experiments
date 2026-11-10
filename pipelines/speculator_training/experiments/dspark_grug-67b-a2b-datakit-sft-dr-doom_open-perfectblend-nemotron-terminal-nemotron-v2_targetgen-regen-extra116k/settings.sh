# STEP 1 of the BigMix targetgen (24 h) comparison: regenerate 116,000 MORE rows with Dr. Doom, regeneration only (no merge, no training; the training folders come next).
# Same source, model, caps and recipe as ../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_targetgen (which regenerated 180,000 rows); only the rows and the output folder differ.
# Sourced after env.sh (needs PROJECT_ROOT and EXP_DIR). Every path that must be fresh is assigned OUTRIGHT below.
source "$(cd "$EXP_DIR/../dspark_grug-67b-a2b-datakit-sft-dr-doom_open-perfectblend-nemotron-terminal-nemotron-v2_targetgen" && pwd)/settings.sh"
WORK_DIR="$PROJECT_ROOT/runs/grug_dspark_drdoom_big_pool_targetgen_extra116k"     # fresh; the 180k run's work dir is only read, never written
PARTS_DIR="$WORK_DIR/regen_parts"
DATA_DIR="$WORK_DIR/data"; CKPT_DIR="$WORK_DIR/checkpoints"; LOG_DIR="$WORK_DIR/logs"
REGEN_PARTS_SRC=""
# The 116k rows are sampled uniformly without replacement from the train rows the 180k run did NOT regenerate (unseen_rows.npy = the first 99.9% of the 2,929,705 rows minus its 180,000 rows, 2,746,775 left),
# so the 296k rows together are a uniform sample without replacement. Built 2026-11-09 from the parts of runs/grug_dspark_drdoom_big_pool_targetgen (180,000 unique rows, 0 error lines).
REGEN_ROWS=${EXTRA_REGEN_ROWS:-116000}
REGEN_ROWS_FILE="$WORK_DIR/unseen_rows.npy"
REGEN_SEED=1
# Part files are named part-s<shard>-<n>.parquet, the SAME names as in the 180k run's parts dir: when the training folders gather both sets, rename these (offset the part number) so that nothing is overwritten or skipped.
LOGGER_ARGS=()
