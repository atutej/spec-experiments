#!/bin/bash
# Submit the dataset-writing job (write -> finalize) to a gg CPU node. Run from a LOGIN node, after submit_fetch.sh has finished.
#   SEQ_LENGTH=32768 bash submit_write.sh [--dry-run]
# SEQ_LENGTH is the trainer's --total-seq-len: choose it from the `stats` output. The dataset goes to dataset_seq<N>/, so several cutoffs can coexist.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${SEQ_LENGTH:?set SEQ_LENGTH, e.g. SEQ_LENGTH=32768 bash submit_write.sh}"
cmd=(sbatch --parsable -p gg -t 2-00:00:00 -J "datakit-write-$SEQ_LENGTH" --export=ALL "$HERE/write.sbatch")
if [[ "${1:-}" == "--dry-run" ]]; then echo "${cmd[@]}"; exit 0; fi
command -v sbatch >/dev/null || { echo "sbatch not found (login node?)" >&2; exit 1; }
mkdir -p /scratch/09749/atutej/marin_speculator/logs/slurm
"${cmd[@]}"
