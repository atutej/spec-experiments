"""Write a Snowball-format train.parquet from locally converted public sources.

Stand-in for snowball-100k, whose build needs Marin object storage. Rows keep the canonical `task_spec` JSON plus the
strata columns the selection step writes, and are drawn with a fixed seed in equal parts from each source.
"""

import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import paths

OUT = paths.DATA / "curation" / "out"  # task_curation driver output, see the README for the command
DEST = paths.STANDIN_TASKS
SEED = 20261019
# source directory -> (family, aai category as in task_curation.selection.AAII_CATEGORIES)
SOURCES = {"openscience": ("stem", "scientific_reasoning"), "reasoning_gym_generated": ("reasoning_gym", "scientific_reasoning")}
SCHEMA = pa.schema(
    [pa.field("task_spec", pa.string(), nullable=False)]
    + [pa.field(name, pa.string()) for name in ("aai_category", "family", "source", "topic", "difficulty")]
    + [pa.field("agentic", pa.bool_())]
)


def referenced_files(spec: str) -> set[str]:
    """Absolute paths a task's graders read at run time: the package locks of its grader environments."""
    found = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            lock = node.get("packages_lock")
            if isinstance(lock, str):
                found.add(lock)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(spec))
    return found


def check_references(rows: list[dict]) -> None:
    """Fail now, not in the middle of a run, when a task points at a file that does not exist.

    A missing grader lock makes every grade of that task fail with a bare FileNotFoundError, which the trainer
    masks, so the run looks healthy while part of its data is thrown away.
    """
    missing = sorted({path for row in rows for path in referenced_files(row["task_spec"]) if not Path(path).is_file()})
    if missing:
        raise SystemExit(
            "tasks reference files that do not exist (build them with the images command in the README):\n  "
            + "\n  ".join(missing)
        )


def main(total: int) -> None:
    per_source = total // len(SOURCES)
    rows = []
    for name, (family, category) in SOURCES.items():
        specs = []
        for part in sorted((OUT / name / "normalize").glob("*.parquet")):
            specs += pq.read_table(part, columns=["task_json"]).column("task_json").to_pylist()
        specs = [spec for spec in specs if spec is not None]
        chosen = random.Random(SEED).sample(specs, per_source)
        print(f"{name}: {len(specs)} usable, kept {len(chosen)}")
        rows += [
            {"task_spec": spec, "aai_category": category, "family": family, "source": name, "topic": "untagged",
             "difficulty": "untagged", "agentic": False}
            for spec in chosen
        ]
    random.Random(SEED).shuffle(rows)
    check_references(rows)
    DEST.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), DEST / "train.parquet")
    print(f"wrote {len(rows)} rows to {DEST / 'train.parquet'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4096)
