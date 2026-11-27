"""Migrate public TaskCompendium task files from schema 0.27 to the schema of the Marin checkout, as selection-format parquet.

    migrate_taskspec.py IN.parquet OUT.parquet

IN has the `lowered_task_spec` column of Open Athena's published splits: one JSON envelope per task with `task`, `runtime`
and `session`. OUT has what the trainer reads: `task_spec` (the migrated TaskSpec) and the label columns of a selection.
Only the task changes; its instructions, tests and environment are carried over as they are. Two edits make a 0.27 task
readable by Marin's current TaskSpec:

1. schema_version 0.27 -> 0.30. The changes in between only add fields that have defaults (normal_submission, empty_output,
   judge, labels). Marin's own upgrade then takes 0.30 to its current version.
2. grader {"kind": "session"} -> {"kind": "none", "reason": ...}. 0.27 had a session grader, a task graded by the harness
   inside the trial; Marin now writes those as NoGrader and selects them by the `harbor` tag.

Every row is validated with Marin's TaskSpec before it is written; one that fails stops the run. Run it with the runtime's
Python, which imports taskcompendium from the marin checkout the runtime was built from.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.models import TaskSpec, upgrade_task_payload

OLD_SCHEMA, MIDDLE_SCHEMA = "0.27", "0.30"
HARNESS_GRADER = {"kind": "none", "reason": "The harness grades the completed trial"}
# The selection's label columns (experiments/post_training/task_curation/selection.py): agentic tasks are family "agentic",
# whose category is "coding"; topic and difficulty stay untagged, as for the stand-in prompts.
FAMILY, AAI_CATEGORY = "agentic", "coding"
SCHEMA = pa.schema(
    [pa.field("task_spec", pa.string(), nullable=False)]
    + [pa.field(name, pa.string()) for name in ("aai_category", "family", "source", "topic", "difficulty")]
    + [pa.field("agentic", pa.bool_())]
)


def migrate(task: dict) -> TaskSpec:
    task = dict(task)
    if task.get("schema_version") == OLD_SCHEMA:
        task["schema_version"] = MIDDLE_SCHEMA
    if task.get("grader", {}).get("kind") == "session":
        task["grader"] = HARNESS_GRADER
    if task.get("schema_version") == MIDDLE_SCHEMA:
        task = upgrade_task_payload(task)
    return TaskSpec.model_validate(task)


def main(source: Path, destination: Path) -> None:
    rows, ids = [], set()
    for position, value in enumerate(pq.read_table(source, columns=["lowered_task_spec"]).column(0).to_pylist()):
        spec = migrate(json.loads(value)["task"])
        assert spec.id not in ids, f"duplicate task id {spec.id}"
        ids.add(spec.id)
        rows.append(
            {
                "task_spec": spec.model_dump_json(),
                "aai_category": AAI_CATEGORY,
                "family": FAMILY,
                "source": spec.source.dataset,
                "topic": "untagged",
                "difficulty": "untagged",
                "agentic": True,
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), destination)
    kinds = Counter(type(TaskSpec.model_validate_json(row["task_spec"]).grader).__name__ for row in rows)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print(json.dumps({"input": source.name, "output": str(destination), "rows": len(rows), "graders": dict(kinds), "sha256": digest}))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
