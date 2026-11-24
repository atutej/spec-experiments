#!/usr/bin/env bash
# Submit a Slurm job through the Marin launcher and wait for it in the background.
#
#   launch.sh <run dir>             submit <run dir>/launch.yaml; outcome in launch.json, log in launch.err
#   launch.sh <run dir> --validate  check the config and render the job scripts, submit nothing
#
# A login shell (bash -l) is used on purpose: WANDB_API_KEY and similar secrets are exported from the user's profile and
# forwarded to the job by name; they are never written into a config or log.
set -euo pipefail
run_dir=$(cd "$1" && pwd)
mode=${2:-}
tools=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
launcher=$(python3 -c "import sys; sys.path.insert(0, '$tools'); import paths; print(paths.RUNTIME_ENV)")/bin/marinskyrl
if [[ "$mode" == "--validate" ]]; then
    exec bash -lc "cd '$run_dir' && '$launcher' slurm launch --config '$run_dir/launch.prepare.yaml'"
fi
bash -lc "cd '$run_dir' && nohup '$launcher' slurm launch --config '$run_dir/launch.yaml' > launch.json 2> launch.err &"
sleep 20
tail -n 2 "$run_dir/launch.err" | cut -c1-240
