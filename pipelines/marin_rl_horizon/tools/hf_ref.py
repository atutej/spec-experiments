"""Resolve a model reference to a local folder through the Hugging Face cache.

A reference is ``ORG/REPO@<40-hex commit>`` (a Hub model pinned to a commit, the form Marin uses for immutable model
identities) or an absolute folder path. For a Hub reference the cached snapshot is returned if it is there; otherwise
it is downloaded once, which needs network (run on a login node). The snapshot folder is named by the commit, so the
returned path cannot silently be another revision.
"""

import os
import re
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

import paths

REF = re.compile(r"^([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)@([0-9a-f]{40})$")
# A model needs its weights, config, tokenizer and chat template; repos of finished runs also carry logs, plots and sources.
IGNORE = ["evidence/*", "configs/*", "source/*", "*.png", "*.svg", "*.md", "*.npy", "step*/*"]


def hf_home() -> str:
    return os.environ.get("HF_HOME") or str(paths.HF_HOME)


def _real_files(snapshot: str, repo: str, commit: str) -> str:
    """Copy a cached snapshot's files into a folder of regular files and return it.

    A Hub cache snapshot holds only symlinks into blobs/, and Marin's check of a local drafter folder lists files through
    fsspec, which reports a symlink as type "other" and so sees the folder as empty ("missing config.json").
    """
    source = Path(snapshot)
    dest = Path(hf_home()) / "materialized" / f"{repo.replace('/', '--')}@{commit[:8]}"
    wanted = sorted(entry for entry in source.iterdir() if entry.is_file())  # is_file follows the symlink
    if dest.is_dir() and all((dest / e.name).is_file() and (dest / e.name).stat().st_size == e.stat().st_size for e in wanted):
        return str(dest)
    staging = dest.with_name(dest.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    for entry in wanted:
        shutil.copyfile(entry, staging / entry.name)  # follows the symlink: a regular file with the same bytes
    shutil.rmtree(dest, ignore_errors=True)
    staging.rename(dest)
    return str(dest)


def resolve(ref: str, what: str, real_files: bool = False) -> tuple[str, str]:
    """Return (local folder, identity): the cached snapshot and its commit, or the folder and its name.

    real_files: return a folder of regular files copied from the snapshot (needed for a drafter, see _real_files).
    """
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
    return (_real_files(folder, repo, commit) if real_files else folder), commit
