#!/bin/bash
# Submit the pipeline stages as a chain of Slurm jobs (run from a LOGIN node: sbatch is refused on compute nodes).
#   bash submit_chain.sh [--dry-run] [stage ...]     stages: export regen prepare train   (default: export regen regen prepare train)
# Each submitted stage waits for the previous submitted one (afterok); a stage repeated right after itself is a retry that runs only if the
# previous one failed or timed out (see the dependency comment below). Submit a subset to rerun or resume one
# stage (e.g. `bash submit_chain.sh train` resumes training from its checkpoints). Overrides pass through the
# environment of this command, e.g. REGEN_LIMIT=1000 WORK_DIR=... bash submit_chain.sh --dry-run
#
# AFTER=<jobid> makes the first submitted stage wait for an already queued job, e.g. when a chain broke after its
# first job: `AFTER=1056558 bash submit_chain.sh regen prepare train`.
#
# Time limits are the maximum the partition's QOS allows (rule of thumb for all jobs: qgb 12:00:00,
# qgg / qgh 2-00:00:00). The qgb QOS also allows only 3 submitted jobs per user: this chain uses 3 gb jobs (regen, regen, train): all that the QOS
# allows at once, so leave no other gb job queued while it runs (submit evals after).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(export regen regen prepare train)   # the second regen resumes if the first hit its time limit
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi

partition() { case "$1" in export|prepare) echo gg ;; regen|train) echo gb ;; *) return 1 ;; esac; }
maxtime()   { case "$1" in gb) echo 12:00:00 ;; *) echo 2-00:00:00 ;; esac; }

# On Vista, sbatch prints a welcome banner and its checks on stdout before the job id, so $(sbatch --parsable) is not
# just the id. Keep the last line that is a bare number (optionally "id;cluster").
submit() {
    local out id
    out=$("$@") || { printf '%s\n' "$out" >&2; return 1; }
    id=$(printf '%s\n' "$out" | grep -E '^[0-9]+(;[^ ]*)?$' | tail -1 | cut -d';' -f1)
    [[ -n "$id" ]] || { echo "no job id in the sbatch output:" >&2; printf '%s\n' "$out" >&2; return 1; }
    echo "$id"
}

# The export stage is only a symlink to the base on-policy experiment's sample when that sample exists: create it HERE, on the login node
# (it takes a moment), instead of queueing a gg job for it. If the sample is missing, the export stage stays in the chain and exports it
# for real on a node (minutes of CPU: never on the login node). Overrides in the environment (WORK_DIR, REGEN_LIMIT, ...) apply.
if [[ "${STAGES[0]:-}" == export ]]; then
    read -r SHARED LINK < <( PROJECT_ROOT="$(cd "$HERE/../../../../../.." && pwd)"; source "$HERE/../settings.sh" >/dev/null 2>&1; echo "$SHARED_SOURCE_FILE $SOURCE_FILE" )
    if [[ -f "$SHARED" ]]; then
        if [[ $DRY -eq 1 ]]; then echo "export: would link $LINK -> $SHARED on the login node (no export job)"
        else
            mkdir -p "$(dirname "$LINK")"; [[ -e "$LINK" ]] || ln -s "$SHARED" "$LINK"
            echo "export: linked $LINK -> $SHARED (on the login node; no export job)"
        fi
        STAGES=("${STAGES[@]:1}")
    else
        echo "export: $SHARED does not exist, so the export stage stays in the chain and exports the sample on a gg node"
    fi
fi

JOB_PREFIX=dspark-sft-onpolicy   # Slurm job names (and log file names): <prefix>-<stage>

# Dependencies. A stage normally starts only if the previous job succeeded (afterok). A stage that is repeated right after itself
# (regen regen, train train) is a RETRY of the resumable stage: it starts only if the job before it failed or hit its time limit
# (afternotok), and Slurm removes it (--kill-on-invalid-dep=yes) when that job succeeded, so no gb node is requested for nothing. The stage
# after a repeated stage starts when ANY job of the group succeeded (afterok:a?afterok:b). At most two jobs per stage (the retry
# depends on the one job before it). The retry resumes: `regen --resume`, the trainer's checkpoints.
prev="${AFTER:-}"; prev_stage=""; group=(); SUBMITTED=()
for s in "${STAGES[@]}"; do
    p=$(partition "$s") || { echo "unknown stage '$s' (export regen prepare train)" >&2; exit 2; }
    cmd=(sbatch --parsable -p "$p" -t "$(maxtime "$p")" -J "$JOB_PREFIX-$s" --export=ALL)
    retry=0
    if [[ -n "$prev_stage" && "$s" == "$prev_stage" && ( "$s" == regen || "$s" == train ) ]]; then
        retry=1; cmd+=(--dependency="afternotok:$prev" --kill-on-invalid-dep=yes)
    elif [[ ${#group[@]} -gt 0 ]]; then
        dep=""; for g in "${group[@]}"; do dep+="${dep:+?}afterok:$g"; done
        cmd+=(--dependency="$dep")
    elif [[ -n "$prev" ]]; then
        cmd+=(--dependency="afterok:$prev")
    fi
    cmd+=("$HERE/run.sbatch" "$s")
    if [[ $DRY -eq 1 ]]; then
        echo "${cmd[*]}"; id="<job-$s>"; [[ $retry -eq 1 ]] && id="<job-$s#2>"
    else
        id=$(submit "${cmd[@]}") || {
            echo "sbatch failed for stage $s." >&2
            [[ ${#SUBMITTED[@]} -gt 0 ]] && echo "Already submitted: ${SUBMITTED[*]}. Cancel (scancel) or continue with AFTER=<last id> bash submit_chain.sh <remaining stages>." >&2
            exit 1; }
        SUBMITTED+=("$s=$id")
        note=""; [[ $retry -eq 1 ]] && note="  [retry: starts only if the previous $s job failed or timed out]"
        echo "$s: job $id (partition $p)$note"
    fi
    if [[ $retry -eq 1 ]]; then group+=("$id"); else group=("$id"); fi
    prev=$id; prev_stage=$s
done
# The directory the jobs will really use (settings.sh default, or WORK_DIR from this command's environment).
RUN_DIR=$( PROJECT_ROOT="$(cd "$HERE/../../../../../.." && pwd)"; source "$HERE/../settings.sh" >/dev/null 2>&1; echo "${WORK_DIR:-}" )
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; Slurm logs in <workspace>/logs/slurm/${JOB_PREFIX}-<stage>-<jobid>.out, run logs in ${RUN_DIR}/logs/"
