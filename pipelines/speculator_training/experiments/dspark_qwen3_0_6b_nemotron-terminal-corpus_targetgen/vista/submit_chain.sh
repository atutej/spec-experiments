#!/bin/bash
# Submit the pipeline stages as a chain of Slurm jobs (run from a LOGIN node: sbatch is refused on compute nodes).
#   bash submit_chain.sh [--dry-run] [stage ...]     stages: export regen prepare train   (default: all)
# Each argument is ONE Slurm job; join stages with commas to run several in the same job (one after another, one time limit). The CPU stages are on gb
# (CPU_PARTITION, common/slurm.sh), where the QOS allows only 3 submitted jobs, hence the default `export,regen prepare,train`.
# Each submitted stage waits for the previous submitted one (afterok). Submit a subset to rerun or resume one
# stage (e.g. `bash submit_chain.sh train` resumes training from its checkpoints). Overrides pass through the
# environment of this command, e.g. REGEN_LIMIT=1000 WORK_DIR=... bash submit_chain.sh --dry-run
#
# AFTER=<jobid> makes the first submitted stage wait for an already queued job, e.g. when a chain broke after its
# first job: `AFTER=1056558 bash submit_chain.sh regen prepare train`.
#
# Time limits are the maximum the partition's QOS allows (rule of thumb for all jobs: qgb 12:00:00,
# qgg / qgh 2-00:00:00). The qgb QOS also allows only 3 submitted jobs per user: this chain uses 2 gb jobs (export,regen and prepare,train).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../../../../../common/slurm.sh"   # gb_reservation_args: the CGAI-gb reservation for gb jobs
DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(export,regen prepare,train)   # 2 gb jobs: the CPU stages ride along with a GPU stage
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi

partition() {  # the partition of a job: one stage, or comma-joined stages that all need the same one
    local st p="" q
    for st in ${1//,/ }; do
        case "$st" in export|prepare) q=$CPU_PARTITION ;; regen|train) q=gb ;; *) return 1 ;; esac
        [[ -z "$p" || "$p" == "$q" ]] || { echo "stages '$1' need different partitions; make them separate jobs" >&2; return 2; }
        p=$q
    done
    [[ -n "$p" ]] && echo "$p"
}
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

JOB_PREFIX=dspark-targetgen   # Slurm job names (and log file names): <prefix>-<stage>
prev="${AFTER:-}"; SUBMITTED=()
for s in "${STAGES[@]}"; do
    p=$(partition "$s") || { echo "unknown stage '$s' (export regen prepare train)" >&2; exit 2; }
    cmd=(sbatch --parsable -p "$p" -t "$(maxtime "$p")" -J "$JOB_PREFIX-${s//,/-}" --export=ALL)
    if [[ "$p" == gb ]]; then res=$(gb_reservation_args); [[ -z "$res" ]] || cmd+=("$res"); fi
    [[ -n "$prev" ]] && cmd+=(--dependency="afterok:$prev")
    # shellcheck disable=SC2206  # the comma-joined stages become separate arguments of run.sh on purpose
    cmd+=("$HERE/run.sbatch" ${s//,/ })
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$s>"; continue; fi
    prev=$(submit "${cmd[@]}") || {
        echo "sbatch failed for stage $s." >&2
        [[ ${#SUBMITTED[@]} -gt 0 ]] && echo "Already submitted: ${SUBMITTED[*]}. Cancel (scancel) or continue with AFTER=<last id> bash submit_chain.sh <remaining stages>." >&2
        exit 1; }
    SUBMITTED+=("$s=$prev")
    echo "$s: job $prev (partition $p)"
done
# The directory the jobs will really use (settings.sh default, or WORK_DIR from this command's environment).
RUN_DIR=$( PROJECT_ROOT="$(cd "$HERE/../../../../../.." && pwd)"; source "$HERE/../settings.sh" >/dev/null 2>&1; echo "${WORK_DIR:-}" )
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; Slurm logs in <workspace>/logs/slurm/${JOB_PREFIX}-<stage>-<jobid>.out, run logs in ${RUN_DIR}/logs/"
