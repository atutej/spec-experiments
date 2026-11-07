#!/bin/bash
# Two-node RL A/B on Vista GB200: MarinSkyRL (Megatron) on R2E-Gym, Datakit-SFT Grug target, with (dspark) and without (plain) our DSpark drafter.
# Run from a LOGIN node (sbatch is refused on compute nodes; login nodes cannot import polars/marin, which is fine here: nothing is imported).
#
#   bash vista/submit_ab.sh                 print what would be submitted (generates both sbatch files, patched)
#   bash vista/submit_ab.sh --submit        submit the dspark arm, then the plain arm (afterany: the reservation has exactly 2 nodes, so they run one after the other)
#   ARMS="dspark" bash vista/submit_ab.sh --submit
#   ARMS="plain" AFTER=<jobid of the dspark arm> bash vista/submit_ab.sh --submit
#
# Partition gb, account ASC26088, reservation CGAI-gb (override with PARTITION=, ACCOUNT=, RESERVATION=).
# Each arm: node 1 = Megatron policy (4 GPUs, EP=4, optimizer state in CPU RAM), node 2 = one vLLM engine (DP4 x EP4). 3 RL steps, 16 prompts x 4 samples,
# seed and tasks identical between arms; the only difference is generator.engine_init_kwargs.speculative_config (see configs/). Not run yet: the memory fit of the
# policy on 4 GPUs (estimated ~33 GiB bf16 weights per rank, from the 2-GPU OOM) and the rollouts (no run has reached Daytona sandboxes) are unverified.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$HERE/../../../../.." && pwd)"
CFG="$REPO_DIR/pipelines/rl_speculation/configs"
set +u; source "$REPO_DIR/env.sh"; set -u
OT="$PROJECT_ROOT/OpenThoughts-Agent"
TIME_LIMIT=${TIME_LIMIT:-03:00:00}      # qgb allows 12:00:00; the CGAI-gb reservation runs to 2026-10-27
RESERVATION=${RESERVATION:-CGAI-gb}
PARTITION=${PARTITION:-gb}             # gb (GB200), not gh (H100)
ACCOUNT=${ACCOUNT:-ASC26088}
ARMS=${ARMS:-"dspark plain"}
# LAYOUT=split (policy on node 1 with 4 GPUs EP=4, one DP4xEP4 engine on node 2) or coloc (policy EP=8 and two DP4xEP4 engines sharing all 8 GPUs; engines sleep
# while the policy trains). split ran out of GPU memory in the first training phase (2026-10-14); coloc halves the policy's per-GPU weights and gradients.
LAYOUT=${LAYOUT:-coloc}
case "$LAYOUT" in split) CFG_SUFFIX=""; POLICY_NODES=1;; coloc) CFG_SUFFIX="_coloc"; POLICY_NODES=2;; *) echo "LAYOUT must be split or coloc" >&2; exit 2;; esac
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1

# same environment as tools/launch_ot_rl.sh; exported here so that sbatch (--export=ALL) carries it into the job
export RL_ENV_DIR="$PROJECT_ROOT/envs/skyrl-megatron" DCFT_RL_ENV="$PROJECT_ROOT/envs/skyrl-megatron" RL_REPO_DIR="$PROJECT_ROOT/MarinSkyRL"
export OT_AGENT_NO_GPU_GRES=1 DCFT="$OT" LD_LIBRARY_PATH="$PROJECT_ROOT/envs/skyrl_extra_lib:${LD_LIBRARY_PATH:-}"

prev="${AFTER:-}"   # AFTER=<jobid>: make the first submitted arm wait for that job (afterany)
for arm in $ARMS; do
    job="grug-${arm}-r2egym-rl-ab${CFG_SUFFIX//_/-}"
    LAUNCH_ARGS="--num_nodes 2 --policy_num_nodes $POLICY_NODES --time_limit $TIME_LIMIT --reservation $RESERVATION --partition $PARTITION --account $ACCOUNT" \
        bash "$REPO_DIR/pipelines/rl_speculation/tools/launch_ot_rl.sh" "$CFG/2node_grug_datakit_${arm}_r2egym_rl${CFG_SUFFIX}.yaml" "$job" > /dev/null
    sb="$OT/experiments/${job}__dryrun/sbatch/${job}_rl.sbatch"
    [[ -f "$sb" ]] || { echo "no sbatch generated for $arm" >&2; exit 1; }
    echo "[$arm] $sb"; grep -E "^#SBATCH (--nodes|--time|-p|--account|--reservation)" "$sb"
    if [[ $SUBMIT -eq 1 ]]; then
        dep=(); [[ -z "$prev" ]] || dep=(--dependency "afterany:$prev")
        # sbatch from the repo root: the job checks that WORKDIR is the OpenThoughts-Agent root
        out=$(cd "$OT" && sbatch --parsable --reservation "$RESERVATION" "${dep[@]}" "$sb")
        # Vista's sbatch wrapper prints a banner before the id: take the last number in the output, not the whole output
        prev=$(printf '%s\n' "$out" | grep -oE '^[0-9]+' | tail -1)
        [[ -n "$prev" ]] || { echo "could not read the job id from sbatch output:" >&2; echo "$out" >&2; exit 1; }
        echo "[$arm] submitted job $prev"
    fi
done
[[ $SUBMIT -eq 1 ]] || echo "(dry: nothing submitted; add --submit)"
