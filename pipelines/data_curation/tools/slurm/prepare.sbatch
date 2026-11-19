#!/bin/bash
# One exclusive node runs the data_curation pipeline for a mix: prepare.sbatch <TARGET_DIR> <MIX>. Submit with submit.sh from a LOGIN node.
# Account, partition, time and the log path are set by submit.sh (sbatch command-line options), so this file is machine-independent.
# Every stage skips finished work, so resubmitting after a time limit continues where it stopped.
#SBATCH -N 1
#SBATCH --exclusive
source ~/.bashrc 2>/dev/null || true   # before `set -u`: bashrc uses unset variables
set -uo pipefail
bash "${DATA_CURATION_TOOLS:?set by submit.sh}/prepare.sh" "${1:?TARGET_DIR}" "${2:?MIX}"
