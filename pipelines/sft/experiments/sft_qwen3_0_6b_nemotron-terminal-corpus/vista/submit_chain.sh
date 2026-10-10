#!/bin/bash
# Submit the pipeline stages as a chain of Slurm jobs (run from a LOGIN node: sbatch is refused on compute nodes).
#   bash submit_chain.sh [--dry-run] [job ...]     stages: export prepare convert train
# Each argument is ONE Slurm job; join stages with commas to run several in the same job (one after another, one time limit). The default is `export,prepare train`:
# one gg job (link or export the sample, then build the dataset on 144 CPU cores) followed by one gb job (train). To use the speculator
# pipeline's prepared data instead of tokenizing (minutes, not hours): `bash submit_chain.sh export,convert train`. The stages of one
# job must use the same partition. Each job waits for the previous one (afterok; afterany between two train jobs, so a second train
# job still starts after the first hit its time limit and resumes from its last checkpoint). Submit a subset to rerun or resume (e.g.
# `bash submit_chain.sh train` resumes training from its checkpoints; `export` and `prepare` skip when their output exists).
# Overrides pass through the environment of this command, e.g. PREBUILT_DATASET=<path> bash submit_chain.sh train
# AFTER=<jobid> makes the first submitted job wait for an already queued job.
# Time limits are the maximum the partition's QOS allows (rule of thumb for all jobs: qgb 12:00:00, qgg / qgh 2-00:00:00). The qgb QOS
# also allows only 3 submitted jobs per user: this chain uses 1 gb job (train).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
STAGES=("$@"); [[ ${#STAGES[@]} -gt 0 ]] || STAGES=(export,prepare train)
if [[ $DRY -eq 0 ]] && ! command -v sbatch >/dev/null; then echo "sbatch not found (login node?)" >&2; exit 1; fi

partition() {  # the partition of a job: one stage, or comma-joined stages that all need the same one
    local st p="" q
    for st in ${1//,/ }; do
        case "$st" in export|prepare|convert) q=gg ;; train) q=gb ;; *) return 1 ;; esac
        [[ -z "$p" || "$p" == "$q" ]] || { echo "stages '$1' need different partitions (gg and gb); make them separate jobs" >&2; return 2; }
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

JOB_PREFIX=sft-qwen3   # Slurm job names (and log file names): <prefix>-<stage>
prev="${AFTER:-}"; prev_stage=""; SUBMITTED=()
for s in "${STAGES[@]}"; do
    p=$(partition "$s") || { echo "bad job '$s' (stages: export prepare convert train; join with commas)" >&2; exit 2; }
    cmd=(sbatch --parsable -p "$p" -t "$(maxtime "$p")" -J "$JOB_PREFIX-${s//,/-}" --export=ALL)
    # afterok: a stage starts only if the previous job succeeded. Between two train jobs it is afterany: a train job that hits its time
    # limit counts as failed, and the next one must still start (it resumes from the last checkpoint). A crash then restarts too.
    if [[ -n "$prev" ]]; then
        if [[ "$s" == train && "$prev_stage" == train ]]; then cmd+=(--dependency="afterany:$prev"); else cmd+=(--dependency="afterok:$prev"); fi
    fi
    # shellcheck disable=SC2206  # the comma-joined stages become separate arguments of run.sh on purpose
    cmd+=("$HERE/run.sbatch" ${s//,/ })
    if [[ $DRY -eq 1 ]]; then echo "${cmd[*]}"; prev="<job-$s>"; prev_stage=$s; continue; fi
    prev=$(submit "${cmd[@]}") || {
        echo "sbatch failed for stage $s." >&2
        [[ ${#SUBMITTED[@]} -gt 0 ]] && echo "Already submitted: ${SUBMITTED[*]}. Cancel (scancel) or continue with AFTER=<last id> bash submit_chain.sh <remaining stages>." >&2
        exit 1; }
    prev_stage=$s
    SUBMITTED+=("$s=$prev")
    echo "$s: job $prev (partition $p)"
done
# The directory the jobs will really use (settings.sh default, or WORK_DIR from this command's environment).
RUN_DIR=$( PROJECT_ROOT="$(cd "$HERE/../../../../../.." && pwd)"; source "$HERE/../settings.sh" >/dev/null 2>&1; echo "${WORK_DIR:-}" )
[[ $DRY -eq 1 ]] || echo "Watch: squeue -u \$USER ; Slurm logs in <workspace>/logs/slurm/${JOB_PREFIX}-<stage>-<jobid>.out, run logs in ${RUN_DIR}/logs/"
