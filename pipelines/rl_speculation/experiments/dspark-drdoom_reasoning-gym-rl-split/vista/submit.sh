#!/bin/bash
# Two-node SPLIT RL run (64 steps, chained 12 h jobs) on Vista GB200: node 1 = Megatron policy (4 GPUs, EP=4), node 2 = one vLLM engine (DP4 x EP4) with the Qwen3-layer targetgen-warm80k DSpark drafter. No colocation, engines never sleep.
# Run from a LOGIN node.   bash vista/submit.sh            generate the patched sbatch and print it (nothing submitted)
#                          bash vista/submit.sh --submit   submit it
# Pinned MarinSkyRL clone (5de48d34) + OpenThoughts-Agent with the Vista patch, env envs/skyrl-megatron; stock vLLM wheel, no overlay (Qwen3 layers).
# Reuses ../../tools/launch_ot_rl.sh and patch_vista_sbatch.py read-only; writes only under OpenThoughts-Agent/experiments/<JOB>__dryrun and $WORK_DIR.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
REPO_DIR="$(cd "$HERE/../../../../.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
source "$EXP/settings.sh"
# Direct wandb logging (2026-11-06): the launcher keeps trainer.logger=wandb only when WANDB_MODE=online; the default entity of this key is dogml, so pin atutej. WANDB_MODE=offline bash submit.sh falls back to console.
export WANDB_MODE="${WANDB_MODE:-online}" WANDB_ENTITY="${WANDB_ENTITY:-atutej}" WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
JOB=${JOB:-drdoom-reasoning-gym-rl-split-smoke}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
CHAIN=${CHAIN:-1}
RESERVATION=${RESERVATION:-CGAI-gb}; PARTITION=${PARTITION:-gb}; ACCOUNT=${ACCOUNT:-ASC26088}
SUBMIT=0; [[ "${1:-}" == "--submit" ]] && SUBMIT=1
OT="$PROJECT_ROOT/OpenThoughts-Agent"
[[ -e "$OT/experiments/${JOB}__dryrun" ]] && echo "note: $OT/experiments/${JOB}__dryrun exists from an earlier generation of THIS job; set JOB=<new name> to keep it" >&2
export MODEL="$TARGET"
# MarinSkyRL source: an export of the pinned commit 5de48d34 (git archive, no change to the clone). The clone itself sits on branch dspark-speculative-decoding, 538 files past
# the pin, whose code imports taskcompendium.resolve_shell_tool (missing from envs/skyrl-megatron, built for the pin; job 1109434/1109435 died on it) and the new rollout schema.
SKYRL_SRC="$PROJECT_ROOT/runs/rl_speculation/marinskyrl_5de48d34_src"
[[ -f "$SKYRL_SRC/skyrl-train/skyrl_train/entrypoints/terminal_bench.py" ]] || { echo "no MarinSkyRL export at $SKYRL_SRC (git -C MarinSkyRL archive 5de48d34 | tar -x -C ...)" >&2; exit 1; }
# Same environment as tools/launch_ot_rl.sh, exported HERE because sbatch --export=ALL carries this shell's environment into the job: without RL_ENV_DIR the job never
# activates envs/skyrl-megatron and runs the launcher on the system Python 3.9 (job 1109413 died in 19 s on `X | None` in hpc/arguments.py).
export RL_ENV_DIR="$PROJECT_ROOT/envs/skyrl-megatron" DCFT_RL_ENV="$PROJECT_ROOT/envs/skyrl-megatron" RL_REPO_DIR="$SKYRL_SRC"
export OT_AGENT_NO_GPU_GRES=1 DCFT="$OT" LD_LIBRARY_PATH="$PROJECT_ROOT/envs/skyrl_extra_lib:${LD_LIBRARY_PATH:-}"
LAUNCH_ARGS="--num_nodes 2 --policy_num_nodes 1 --time_limit $TIME_LIMIT --reservation $RESERVATION --partition $PARTITION --account $ACCOUNT --skyrl_override environment.env_class=reasoning_gym" \
    bash "$REPO_DIR/pipelines/rl_speculation/tools/launch_ot_rl.sh" "${YAML:-$EXP/rl_dspark.yaml}" "$JOB" > /dev/null
sb="$OT/experiments/${JOB}__dryrun/sbatch/${JOB}_rl.sbatch"
[[ -f "$sb" ]] || { echo "no sbatch generated" >&2; exit 1; }
# The launcher bakes whatever DAYTONA_API_KEY it sees into the sbatch as plain text (DAYTONA_API_KEY_OVERRIDE="..."): blank it. Runs that need Daytona still get the key from the secrets file the sbatch sources.
sed -i 's/^DAYTONA_API_KEY_OVERRIDE=".*"/DAYTONA_API_KEY_OVERRIDE=""/' "$sb"
grep -q '^DAYTONA_API_KEY_OVERRIDE="dtn' "$sb" && { echo "Daytona key still in $sb" >&2; exit 1; }
# NCCL memory probe (2026-11-02): every step-2 OOM so far had ~92 GiB per policy GPU held outside PyTorch. Marin documents that automatic NCCL channel selection OOMs on the second update
# (marin-community/marin docs/dev-guide/skyrl.md, "Grug training communication"). One channel is the value expert_block weight publication supports; NVLS (NVLink SHARP) buffers are off too.
sed -i "/^export CUDA_PATH=\"\$CUDA_HOME\"/a export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS_PROBE:-1} NCCL_NVLS_ENABLE=${NCCL_NVLS_PROBE:-0}   # memory probe" "$sb"
grep -q "SKYRL_VLLM_SLEEP_LEVEL" "$sb" && echo "warning: sleep level exported; not needed in the split layout" >&2
grep -q 'export CUDA_PATH="\$CUDA_HOME"' "$sb" || { echo "anchor missing in $sb" >&2; exit 1; }
sed -i "/^export CUDA_PATH=\"\$CUDA_HOME\"/a export PYTHONPATH=\"$SKYRL_SRC:$SKYRL_SRC/skyrl-train\${PYTHONPATH:+:\$PYTHONPATH}\"   # pinned MarinSkyRL export before the env's editable clone" "$sb"
# EXPANDABLE=1 (default): PyTorch expandable segments, as its OOM message suggests. Job 1109622 reserved 167 GiB for 75 GiB allocated in step 1 (fragmentation) and then ran out of
# memory in step 2's ppo_train. Expandable segments are not supported together with CUDA IPC; the weight sync here is NCCL (expert_block), so it should be fine, but it is untested.
if [[ "${EXPANDABLE:-1}" == "1" ]]; then
    grep -q "export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6" "$sb" || { echo "PYTORCH_ALLOC_CONF line not found in $sb" >&2; exit 1; }
    sed -i "s/^export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.6.*/export PYTORCH_ALLOC_CONF=expandable_segments:True   # spec-experiments submit.sh (EXPANDABLE=1)/" "$sb"
    grep -n "^export PYTORCH_ALLOC_CONF" "$sb"
fi
echo "sbatch: $sb"; grep -E "^#SBATCH (--nodes|--time|-p|--account|--reservation)" "$sb"
if [[ $SUBMIT -eq 1 ]]; then
    prev="${DEPEND_ON:-}"
    for i in $(seq 1 "$CHAIN"); do
        dep=(); [[ -z "$prev" ]] || dep=(--dependency "afterany:$prev")
        out=$(cd "$OT" && sbatch --parsable --reservation "$RESERVATION" "${dep[@]}" "$sb")
        prev=$(printf '%s\n' "$out" | grep -oE '^[0-9]+' | tail -1)
        [[ -n "$prev" ]] || { echo "could not read job id from: $out" >&2; exit 1; }
        echo "submitted chain job $i/$CHAIN: $prev (job name $JOB)"
    done
else
    echo "(dry: nothing submitted; add --submit)"
fi
