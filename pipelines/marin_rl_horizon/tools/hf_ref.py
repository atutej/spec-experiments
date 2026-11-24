"""Resolve a model reference to a local folder through the Hugging Face cache.

A reference is ``ORG/REPO@<40-hex commit>`` (a Hub model pinned to a commit, the form Marin uses for immutable model
identities) or an absolute folder path. For a Hub reference the cached snapshot is returned if it is there; otherwise
it is downloaded once, which needs network (run on a login node). The snapshot folder is named by the commit, so the
returned path cannot silently be another revision.
"""

import os
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

import paths

REF = re.compile(r"^([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)@([0-9a-f]{40})$")
# A model needs its weights, config, tokenizer and chat template; repos of finished runs also carry logs, plots and sources.
IGNORE = ["evidence/*", "configs/*", "source/*", "*.png", "*.svg", "*.md", "*.npy", "step*/*"]


def hf_home() -> str:
    return os.environ.get("HF_HOME") or str(paths.HF_HOME)


def resolve(ref: str, what: str) -> tuple[str, str]:
    """Return (local folder, identity): the cached snapshot and its commit, or the folder and its name."""
    if ref.startswith("/"):
        folder = Path(ref)
        if not (folder / "config.json").is_file():
            raise SystemExit(f"{what}: {ref} has no config.json")
        return str(folder), folder.name
    match = REF.match(ref)
    if match is None:
        raise SystemExit(f"{what} takes ORG/REPO@<40-hex commit> or an absolute folder path, got {ref!r}")
    repo, commit = match.groups()
    options = dict(repo_id=repo, revision=commit, cache_dir=str(Path(hf_home()) / "hub"), ignore_patterns=IGNORE)
    try:
        folder = snapshot_download(**options, local_files_only=True)
    except LocalEntryNotFoundError:
        print(f"{what}: {repo}@{commit[:8]} is not in {hf_home()}; downloading it once", flush=True)
        folder = snapshot_download(**options)
    return folder, commit
