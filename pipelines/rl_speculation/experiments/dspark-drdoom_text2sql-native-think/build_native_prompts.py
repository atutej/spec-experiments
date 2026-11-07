#!/usr/bin/env python3
"""Derive SQL task parquet whose system prompt no longer asks for <think>...</think> (the model thinks in its own template).
Reads the downloaded SkyRL-SQL parquet read-only and writes NEW files into a fresh directory."""
import re, sys
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=False)
BULLET = re.compile(r"- Conduct thinking inside <think>\.\.\.</think> blocks every time you get new observation or information\. ?\n")

def fix(system: str) -> str:
    s, n = BULLET.subn("", system)
    assert n == 1, "format bullet not found"
    s = re.sub(r"<think>(.*?)</think>", r"\1", s, flags=re.S)   # the worked example keeps its reasoning, without tags
    assert "<think>" not in s and "</think>" not in s
    return s

VAL_ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 0   # optional: also write validation_<N>.parquet from the first N rows of validation.parquet
jobs = [("train.parquet", "train.parquet", None)]
jobs += [("validation_64.parquet", "validation_64.parquet", None)] if not VAL_ROWS else [("validation.parquet", f"validation_{VAL_ROWS}.parquet", VAL_ROWS)]
for name, out_name, nrows in jobs:
    t = pq.read_table(src / name)
    if nrows: t = t.slice(0, nrows)
    rows = t.to_pylist()
    for r in rows:
        assert r["prompt"][0]["role"] == "system"
        r["prompt"][0]["content"] = fix(r["prompt"][0]["content"])
        r["env_class"] = "text2sql_native_think"   # the trainer takes the environment from this per-row column, not from environment.env_class
    pq.write_table(pa.Table.from_pylist(rows, schema=t.schema), out / out_name)
    print(out_name, len(rows))
