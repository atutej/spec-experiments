#!/usr/bin/env python
"""Export a seeded random sample of a speculators dataset preset to JSONL.

Why: `datasets` cannot read some parquet files (e.g. nvidia/Nemotron-Terminal-Corpus,
whose single row group holds >2 GB of nested text and overflows Arrow's list offsets).
Plain pyarrow reads them fine in small batches, so this script only does the reading;
`Dataset.from_generator` builds a normal HF dataset from those rows, and HF's own
`.shuffle` / `.select` / `.to_json` do the sampling and writing. Rows are copied
verbatim (all columns); feed the output to `speculators regenerate-responses --dataset`.

The preset (hf_path, subset, split) comes from DATASET_CONFIGS. Presets with a
filter_fn or normalize_fn are refused: a local file loses them in regenerate-responses.

Only parquet-backed presets are supported. Run in the `speculators` conda env.
The first run copies the whole subset into the HF cache; later runs reuse it.
"""

import argparse
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, load_dataset_builder
from huggingface_hub import hf_hub_download

from speculators.cli.regenerate_responses import REGEN_DATASETS
from speculators.data_generation.configs import DATASET_CONFIGS

BATCH_ROWS = 256  # small enough that no batch nears Arrow's 2 GB limit
HF_URL = re.compile(r"^hf://datasets/(?P<repo>[^@/]+/[^@/]+)(?:@(?P<rev>[^/]+))?/(?P<path>.+)$")


def resolve_parquet_files(hf_path: str, subset: str | None, split: str) -> list[str]:
    """Local cached paths of the parquet files behind a preset, in a stable order."""
    builder = load_dataset_builder(hf_path, name=subset)
    paths = []
    for url in builder.config.data_files[split]:
        m = HF_URL.match(str(url))
        if m is None or not m["path"].endswith(".parquet"):
            sys.exit(f"unsupported data file (parquet on the Hub only): {url}")
        paths.append(
            hf_hub_download(m["repo"], m["path"], repo_type="dataset", revision=m["rev"])
        )
    return paths


def read_rows(files: list[str], columns: tuple[str, ...]):
    # Files of one subset can differ in columns (code.parquet has `source`, swe does
    # not); every row carries the union, with None where its file lacks a column.
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=BATCH_ROWS):
            for row in batch.to_pylist():
                yield {c: row.get(c) for c in columns}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=REGEN_DATASETS, help="preset name")
    ap.add_argument("--subset", default=None, help="override the preset's subset")
    ap.add_argument("--split", default=None, help="override the preset's split")
    ap.add_argument("--limit", type=int, required=True, help="rows to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    if config.filter_fn is not None or config.normalize_fn is not None:
        sys.exit(f"{args.dataset} has a filter_fn/normalize_fn; use the preset directly")
    subset = args.subset if args.subset is not None else config.subset
    split = args.split if args.split is not None else config.split
    files = resolve_parquet_files(config.hf_path, subset, split)

    schema = pa.unify_schemas([pq.read_schema(f) for f in files])
    ds = Dataset.from_generator(
        read_rows,
        # a list in gen_kwargs is sharded across workers; columns is a tuple so it is not
        gen_kwargs={"files": files, "columns": tuple(schema.names)},
        features=Features.from_arrow_schema(schema),
    )
    print(f"{config.hf_path} [{subset}/{split}]: {len(ds)} rows in {len(files)} files")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.limit, len(ds))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    ds.to_json(tmp)  # JSON Lines
    tmp.replace(out)
    print(f"wrote {len(ds)} rows to {out}")


if __name__ == "__main__":
    main()
