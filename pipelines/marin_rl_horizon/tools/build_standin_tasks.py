"""Write a Snowball-format train.parquet from locally converted public sources.

Stand-in for snowball-100k, whose build needs Marin object storage. Rows keep the canonical `task_spec` JSON plus the
strata columns the selection step writes, and are drawn with a fixed seed in equal parts from each source.
"""

import random
import sys

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
    DEST.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), DEST / "train.parquet")
    print(f"wrote {len(rows)} rows to {DEST / 'train.parquet'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4096)
