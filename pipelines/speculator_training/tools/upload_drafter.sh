# Sourced by the drafter experiments' vista/run.sh (after settings.sh and use_env): the `upload` stage, which sends the finished drafter
# (best and latest checkpoint, see upload_drafter_to_hf.py) to the Hugging Face Hub. Uses HF_REPO, HF_PRIVATE, HF_DESCRIPTION (settings.sh),
# CKPT_DIR, RUN_NAME, REPO_DIR. HF_DRY_RUN=1 prints the file list and the model card without touching the network.
# Token: HF_TOKEN, or HUG_TOKEN (this account's name for it in ~/.bashrc), or `hf auth login`.
stage_upload() {
    [[ -n "${HF_REPO:-}" ]] || { echo "set HF_REPO=<namespace>/<name> (or auto) to upload" >&2; return 1; }
    [[ -d "$CKPT_DIR" ]] || { echo "no checkpoints in $CKPT_DIR (run the train stage first)" >&2; return 1; }
    echo "=== Upload: $CKPT_DIR -> Hugging Face repo $HF_REPO (the logged-in account if it has no namespace) ==="
    export HF_TOKEN="${HF_TOKEN:-${HUG_TOKEN:-}}"
    [[ -n "$HF_TOKEN" ]] || unset HF_TOKEN   # an empty value would hide a token saved by `hf auth login`
    use_env speculators
    local args=(--checkpoints-dir "$CKPT_DIR" --repo-id "$HF_REPO" --run-name "$RUN_NAME" --description "${HF_DESCRIPTION:-}"
                --commit-message "Drafter $RUN_NAME")
    [[ "${HF_PRIVATE:-false}" == true ]] || args+=(--public)   # public by default; HF_PRIVATE=true makes the repo private
    [[ -z "${HF_DRY_RUN:-}" ]] || args+=(--dry-run)
    python "$REPO_DIR/pipelines/speculator_training/tools/upload_drafter_to_hf.py" "${args[@]}"
}

# After the stages of a job that trained: upload when HF_REPO is set. A failed upload must not fail the job (the drafter is trained and
# saved); it says how to retry. Not reached when training failed or hit its time limit, so a partial drafter is never uploaded.
upload_after_training() {
    case " $* " in *" train "*) ;; *) return 0 ;; esac
    [[ -n "${HF_REPO:-}" ]] || return 0
    stage_upload || echo "WARNING: the upload to $HF_REPO failed; retry with: HF_REPO=$HF_REPO bash vista/run.sh upload" >&2
}
