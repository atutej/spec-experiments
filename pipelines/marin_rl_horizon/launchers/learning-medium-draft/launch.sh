#!/usr/bin/env bash
# Snowball learning-medium preset with a frozen speculative-decoding drafter on TACC Horizon (Slurm).
#
#   ./launch.sh [--policy ORG/REPO@COMMIT] [--name RUN] [--time-limit HH:MM:SS] [--validate]
#                  [--drafter ORG/REPO@COMMIT | /path/to/drafter] [--proposals N]
#
# 25 RL updates, replies stop naturally, no evaluations, no checkpoints; stand-in prompts. recipe.yaml is tools/dump_recipe.py --preset learning-medium.
# --policy: the model RL starts from, a Hugging Face model pinned to a full 40-hex commit or an absolute folder (default: the preset's 09.21 initializer).
# --drafter: the speculative drafter, in the same forms (default: the drafter the preset names, marin-community/marin#11975); --proposals defaults to the recipe's 4.
# Hub models resolve through the Hub cache ($HF_HOME, default $PROJECT_ROOT/cache/hf) and are downloaded once if missing, so the first run needs a login node with network.
# Needs PROJECT_ROOT exported (the workspace root); also RUNTIME_NAME and MARIN_REPO if the runtime and the marin checkout are not named as in the pipeline README.
# Logs to Weights & Biases only when WANDB_ENTITY and WANDB_PROJECT are exported (WANDB_API_KEY comes from your login profile). --validate writes the config and checks it; nothing is submitted.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tools=$here/../../tools
policy=""; name=""; time_limit=02:00:00; validate=""; out=""; drafter=""; proposals=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy) policy=$2; shift ;;
        --name) name=$2; shift ;;
        --time-limit) time_limit=$2; shift ;;
        --validate) validate=--validate ;;
        --out) out=$2; shift ;;
        --drafter) drafter=$2; shift ;;
        --proposals) proposals=$2; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
py=$(cd "$tools" && python3 -c "import paths; print(paths.RUNTIME_ENV)")/bin/python
args=(--recipe "$here/recipe.yaml" --time-limit "$time_limit")
[[ -n "$drafter" ]] && args+=(--drafter "$drafter")
[[ -n "$proposals" ]] && args+=(--proposals "$proposals")
[[ -n "$policy" ]] && args+=(--policy "$policy")
if [[ -n "$name" ]]; then args+=(--name "$name"); else args+=(--name-prefix horizon-learning-medium-draft); fi
[[ -n "${WANDB_ENTITY:-}" && -n "${WANDB_PROJECT:-}" ]] && args+=(--wandb-entity "$WANDB_ENTITY" --wandb-project "$WANDB_PROJECT")
[[ -n "$out" ]] && args+=(--out "$out")
run_dir=$(cd "$tools" && PYTHONPATH=. "$py" make_config.py "${args[@]}" | tail -1)
"$tools/launch.sh" "$run_dir" $validate
