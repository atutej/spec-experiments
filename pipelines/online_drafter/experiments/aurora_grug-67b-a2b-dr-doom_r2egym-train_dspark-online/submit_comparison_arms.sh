#!/bin/bash
# Submit the two R2E-Gym comparison arms in sequence (run from a LOGIN node): big-mix targetgen first, then Datakit half itw / half targetgen after it (afterany, so it starts when the first ends or is cut by its time limit).
# Each is a 4 h, one-node job (see the sbatch headers); the gb QOS allows 3 submitted jobs per user and 2 nodes running.   bash submit_comparison_arms.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
a=$(sbatch --parsable "$HERE/run_long_bigmix_targetgen.sbatch") || exit 1
echo "bigmix targetgen: job $a"
b=$(sbatch --parsable --dependency=afterany:"$a" "$HERE/run_long_half_itw_targetgen.sbatch") || exit 1
echo "half itw / half targetgen: job $b (after $a)"
