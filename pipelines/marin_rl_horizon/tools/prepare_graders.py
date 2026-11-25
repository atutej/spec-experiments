"""Check that the stand-in tasks' grader files exist, and build the Python environments their graders run in.

    prepare_graders.py <run dir>            build what is missing (what launch.sh runs before it submits)
    prepare_graders.py <run dir> --check    only check that the lock files exist (what launch.sh --validate runs)

A Reasoning Gym task is graded by a script that imports reasoning_gym, in a Python environment built from a hash lock the
task names by absolute path. The first grade on a node builds that environment under ~/.cache/marin/task-runtime, which is on
the shared home filesystem, so ten nodes would build it at the same time. Building it here, once, before the job starts avoids
that; a second call finds it built and returns. The lock itself comes from the README's images step.

Run it with the runtime's Python (launch.sh does): it imports taskcompendium and shellbox from the marin checkout the runtime
was built from, so the environment gets the identity the nodes will look for. Packages that are only available as source are
built with the system gcc, because the login shell exports CC=nvc, which cannot compile them.
"""

import os
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from taskcompendium.runtime.local import local_runtime

from build_standin_tasks import referenced_files


def train_files(config: dict) -> list[Path]:
    return [
        Path(entry["uri"]) / entry["relative_path"]
        for entry in config["inputs"]["train_data"]
        if entry["kind"] == "directory"
    ]


def main(run_dir: Path, check_only: bool) -> None:
    config = yaml.safe_load((run_dir / "launch.yaml").read_text())
    locks: set[str] = set()
    for path in train_files(config):
        for spec in pq.read_table(path, columns=["task_spec"]).column("task_spec").to_pylist():
            locks |= referenced_files(spec)
    missing = sorted(lock for lock in locks if not Path(lock).is_file())
    if missing:
        raise SystemExit(
            "the tasks name grader files that do not exist; build them with the images command in the README "
            "and rebuild the tasks:\n  " + "\n  ".join(missing)
        )
    print(f"{len(locks)} grader lock(s) exist")
    if check_only:
        return
    if shutil.which("uv") is None:
        raise SystemExit("uv is not on PATH; it builds the grader environments")
    os.environ.update(CC="gcc", CXX="g++", UV_CACHE_DIR=os.environ.get("UV_CACHE_DIR") or f"/tmp/uv-cache-{os.environ['USER']}")
    for lock in sorted(locks):
        runtime = local_runtime(lock)
        runtime.ensure_built()
        print(f"grader environment ready: {runtime.root}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--check"):
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve(), check_only=len(sys.argv) == 3)
