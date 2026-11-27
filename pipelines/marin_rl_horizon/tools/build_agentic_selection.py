"""Build the agentic selection for train-medium-agentic: half agentic tasks, half non-agentic prompts, fixed seed.

    build_agentic_selection.py POOL_DIR STANDIN_TRAIN OUT_DIR

POOL_DIR holds the migrated pool (migrate_taskspec.py output): swesmith/ and calibforge-shared/ with train and validation
parquet. STANDIN_TRAIN is the stand-in prompts file (build_standin_tasks.py). OUT_DIR gets train.parquet, validation.parquet
and manifest.json.

The recipe composes each batch from half agentic and half non-agentic tasks (rl_agentic_recipe._BATCH_MIX), and a step takes
64 prompts, so 3,200 train rows are 50 steps of one pass. The agentic half is all of CalibForge-shared's train tasks (the only
terminal tasks in the pool) and SWE-smith tasks for the rest. The non-agentic half is OpenScience only: the Reasoning Gym tasks
carry an absolute path to a grader environment on the machine that built them, so they are neither portable nor publishable.
Every task is written at the schema of the Marin checkout. The validation file holds the pool's own validation tasks, all agentic. Nothing has been removed for overlap with evaluation
sets; that is not checked here.
"""

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from taskcompendium.models import TaskSpec, upgrade_task_payload

SEED = 20261127
STEPS, PROMPTS_PER_STEP = 50, 64
TRAIN_ROWS = STEPS * PROMPTS_PER_STEP
AGENTIC_ROWS = NON_AGENTIC_ROWS = TRAIN_ROWS // 2


def read(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def to_current_schema(row: dict) -> dict:
    """Write a row's task at the schema of the Marin checkout (the stand-in prompts are one version behind the migrated pool)."""
    task = json.loads(row["task_spec"])
    if task.get("schema_version") == "0.30":
        task = upgrade_task_payload(task)
    return {**row, "task_spec": TaskSpec.model_validate(task).model_dump_json()}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(pool: Path, standin: Path, out: Path) -> None:
    rng = random.Random(SEED)
    calibforge = read(pool / "calibforge-shared" / "train.parquet")
    swesmith = read(pool / "swesmith" / "train.parquet")
    openscience = [to_current_schema(row) for row in read(standin) if row["source"] == "openscience"]
    swesmith_rows = AGENTIC_ROWS - len(calibforge)
    assert 0 < swesmith_rows <= len(swesmith) and NON_AGENTIC_ROWS <= len(openscience)
    chosen = (
        calibforge
        + rng.sample(swesmith, swesmith_rows)
        + rng.sample(openscience, NON_AGENTIC_ROWS)
    )
    rng.shuffle(chosen)
    validation = read(pool / "swesmith" / "validation.parquet") + read(pool / "calibforge-shared" / "validation.parquet")
    for row in chosen + validation:
        assert "/scratch/" not in row["task_spec"] and "atutej" not in row["task_spec"], "a task names a local path"
    ids = [json.loads(row["task_spec"])["id"] for row in chosen]
    assert len(set(ids)) == len(ids), "duplicate task ids in the selection"
    out.mkdir(parents=True, exist_ok=True)
    schema = pq.read_schema(pool / "swesmith" / "train.parquet")
    for name, rows in (("train", chosen), ("validation", validation)):
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / f"{name}.parquet")
    manifest = {
        "seed": SEED, "steps": STEPS, "prompts_per_step": PROMPTS_PER_STEP,
        "train_rows": len(chosen), "validation_rows": len(validation),
        "train_by_source": dict(Counter(f"{row['family']}:{row['source']}" for row in chosen)),
        "train_agentic_by_pool": {"calibforge-shared": len(calibforge), "swesmith": swesmith_rows},
        "inputs": {str(p.relative_to(pool)) if p.is_relative_to(pool) else p.name: sha256(p) for p in
                   [pool / "swesmith/train.parquet", pool / "swesmith/validation.parquet",
                    pool / "calibforge-shared/train.parquet", pool / "calibforge-shared/validation.parquet", standin]},
        "outputs": {f"{name}.parquet": sha256(out / f"{name}.parquet") for name in ("train", "validation")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("train_rows", "validation_rows", "train_by_source", "train_agentic_by_pool", "outputs")}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
