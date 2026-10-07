#!/bin/bash
# Bring the marin_speculator workspace to a usable state on this machine. Idempotent.
#
#   setup/setup.sh                 clone missing repos, build missing/failing envs, check all
#   setup/setup.sh --check         only report repo and env status (changes nothing)
#   setup/setup.sh --update        also fast-forward clean repos to their remote ref
#   setup/setup.sh --rebuild ENV   re-run ENV's build steps even if its check passes
#   setup/setup.sh --freeze        write setup/locks/$MACHINE/<env>.txt (exact package lists)
#   setup/setup.sh --gpu           also run the GPU smoke test (setup/check_gpu.sh; GPU nodes only)
#
# Everything except --gpu works on CPU-only nodes (Vista gg), including building envs.
#
# Bootstrap on a fresh machine or after a $SCRATCH purge:
#   git clone git@github.com:atutej/spec-experiments.git <PROJECT_ROOT>/spec-experiments
#   bash <PROJECT_ROOT>/spec-experiments/setup/setup.sh
#
# Dependencies: repos in setup/repos.txt, env build steps in setup/envs/<env>.sh.
set -uo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SETUP_DIR")"
source "$REPO_DIR/env.sh"

CHECK_ONLY=0 UPDATE=0 FREEZE=0 GPU=0 REBUILD=()
while [[ $# -gt 0 ]]; do
    case "$1" in
    --check) CHECK_ONLY=1 ;;
    --update) UPDATE=1 ;;
    --freeze) FREEZE=1 ;;
    --gpu) GPU=1 ;;
    --rebuild) REBUILD+=("$2"); shift ;;
    -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

conda_env_exists() { conda env list | awk '{print $1}' | grep -qx "$1" || [[ -d "${CONDA_ENVS_PATH:-/nonexistent}/$1" ]]; }
FAILED=()
echo "MACHINE=$MACHINE  NODE_KIND=$NODE_KIND  NUM_GPUS=$NUM_GPUS  PROJECT_ROOT=$PROJECT_ROOT  CONDA_ROOT=$CONDA_ROOT${CONDA_ENVS_PATH:+  CONDA_ENVS_PATH=$CONDA_ENVS_PATH}"

# ---- workspace directories and the setup skill ----
if [[ $CHECK_ONLY -eq 0 ]]; then
    mkdir -p "$PROJECT_ROOT/runs" "${CONDA_ENVS_PATH:-$PROJECT_ROOT/runs}"
    # Claude Code loads project skills from the directory it starts in; expose this repo's
    # skills at the workspace root too, without replacing anything already there.
    mkdir -p "$PROJECT_ROOT/.claude/skills"
    for skill in "$REPO_DIR"/.claude/skills/*/; do
        link="$PROJECT_ROOT/.claude/skills/$(basename "$skill")"
        [[ -e "$link" || -L "$link" ]] || ln -s "$skill" "$link"
    done
fi

# ---- repos ---- (git gets </dev/null: ssh would otherwise eat the rest of repos.txt)
echo; echo "== repos (setup/repos.txt)"
while read -r dir url ref upstream; do
    [[ -z "$dir" || "$dir" == \#* ]] && continue
    path="$PROJECT_ROOT/$dir"
    if [[ ! -d "$path/.git" ]]; then
        if [[ $CHECK_ONLY -eq 1 ]]; then echo "MISSING  $dir"; FAILED+=("repo:$dir"); continue; fi
        echo "cloning  $dir ($url @ $ref)"
        git clone --branch "$ref" "$url" "$path" </dev/null || { FAILED+=("repo:$dir"); continue; }
        [[ -n "${upstream:-}" ]] && git -C "$path" remote add upstream "$upstream"
    fi
    git -C "$path" fetch -q origin </dev/null 2>/dev/null || echo "  (fetch failed for $dir)"
    head=$(git -C "$path" rev-parse --abbrev-ref HEAD)
    dirty=$(git -C "$path" status --porcelain --untracked-files=no | wc -l)
    status="$head"
    [[ "$head" != "$ref" ]] && status+=" (expected $ref)"
    [[ $dirty -gt 0 ]] && status+=", $dirty modified files"
    if git -C "$path" rev-parse -q --verify "origin/$ref" >/dev/null; then
        read -r ahead behind < <(git -C "$path" rev-list --left-right --count "HEAD...origin/$ref")
        status+=", ahead $ahead / behind $behind origin/$ref"
        if [[ $UPDATE -eq 1 && $CHECK_ONLY -eq 0 && $dirty -eq 0 && "$head" == "$ref" && $behind -gt 0 ]]; then
            git -C "$path" merge -q --ff-only "origin/$ref" && status+=" -> fast-forwarded"
        fi
    fi
    echo "ok       $dir: $status"
done < "$SETUP_DIR/repos.txt"

# ---- envs ----
for recipe in "$SETUP_DIR"/envs/*.sh; do
    (
        source "$recipe"
        echo; echo "== env $ENV_NAME ($(basename "$recipe"))"
        rebuild=0
        for r in "${REBUILD[@]:-}"; do [[ "$r" == "$ENV_NAME" ]] && rebuild=1; done
        if [[ $rebuild -eq 0 ]] && env_check; then
            echo "ok       $ENV_NAME"
        elif [[ $CHECK_ONLY -eq 1 ]]; then
            echo "FAILING  $ENV_NAME (run setup/setup.sh to build it)"; exit 1
        else
            echo "building $ENV_NAME ..."
            env_build && env_check && echo "ok       $ENV_NAME (built)" || { echo "FAILED   $ENV_NAME"; exit 1; }
        fi
        if [[ $FREEZE -eq 1 ]]; then
            lock="$SETUP_DIR/locks/$MACHINE/$ENV_NAME.txt"
            mkdir -p "$(dirname "$lock")"
            {
                echo "# $ENV_NAME on $MACHINE ($(uname -m)), $(date -u +%Y-%m-%dT%H:%MZ)"
                conda run -n "$ENV_NAME" python -c "import sys; print('# python', sys.version.split()[0])"
                conda run -n "$ENV_NAME" pip freeze --all
            } > "$lock" && echo "froze    $lock"
        fi
    ) || FAILED+=("env:$(basename "$recipe" .sh)")
done

if [[ $GPU -eq 1 ]]; then
    echo; echo "== GPU smoke test"
    if [[ "$NUM_GPUS" -eq 0 ]]; then
        echo "no GPU on this node (NODE_KIND=$NODE_KIND); run --gpu on a GPU node (Vista: gh or gb)"
        FAILED+=("gpu")
    else
        bash "$SETUP_DIR/check_gpu.sh" || FAILED+=("gpu")
    fi
fi

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then echo "NOT READY: ${FAILED[*]}"; exit 1; fi
echo "READY"
