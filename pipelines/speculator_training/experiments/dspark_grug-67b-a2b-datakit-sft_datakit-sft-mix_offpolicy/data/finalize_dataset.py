"""Finish the dataset directory `build_datakit_sft_mix.py write` produced. Run in the speculators env (needs torch and datasets).

    python data/finalize_dataset.py <dataset dir> [--full-check]

- writes token_freq.pt (the file `speculators.train` builds the reduced draft vocabulary from) from token_freq.json;
- loads the directory exactly as the trainer does (`datasets.load_from_disk`) and checks it against manifest.json: row count,
  columns and dtypes, seq_len == len(input_ids), loss_mask in {0,1}, nothing longer than the manifest's seq_length;
- --full-check also recomputes the supervised-token frequencies from every row and compares them with token_freq.pt.
"""
import argparse, collections, json, sys
from pathlib import Path

import torch
from datasets import load_from_disk

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("--full-check", action="store_true")
a = ap.parse_args()
d = Path(a.dataset)
manifest = json.loads((d / "manifest.json").read_text())
freq = {int(k): v for k, v in json.loads((d / "token_freq.json").read_text()).items()}
torch.save(freq, d / "token_freq.pt")
print(f"token_freq.pt: {len(freq):,} distinct supervised tokens, {sum(freq.values()):,} supervised tokens")

ds = load_from_disk(str(d))
n = manifest["totals"]["rows_kept"]
assert len(ds) == n, f"dataset has {len(ds)} rows, manifest says {n}"
assert set(ds.column_names) == {"input_ids", "loss_mask", "seq_len"}, ds.column_names
assert str(ds.features["input_ids"]) == str(ds.features["loss_mask"]), ds.features
L = manifest["seq_length"]
raw = ds.with_format(None)
check = range(len(raw)) if a.full_check else sorted(set(list(range(min(len(raw), 2000))) + list(range(0, len(raw), max(1, len(raw) // 2000)))))
recount, bad = collections.Counter(), 0
for i in check:
    r = raw[i]
    ids, mask = r["input_ids"], r["loss_mask"]
    ok = len(ids) == len(mask) == r["seq_len"] <= L and set(mask) <= {0, 1} and any(mask)
    bad += not ok
    if a.full_check:
        for t, m in zip(ids, mask):
            if m: recount[t] += 1
assert bad == 0, f"{bad} rows failed the per-row checks"
print(f"loaded {len(ds):,} rows from {len(list(d.glob('data-*.arrow')))} shards; checked {len(check):,} rows: seq_len == len(input_ids) <= {L}, mask in {{0,1}}, every row supervised")
if a.full_check:
    assert dict(recount) == freq, "token_freq.pt differs from the supervised-token counts of the rows"
    print("token_freq.pt == supervised-token counts recomputed from every row")
t = ds[0]; print("torch format row 0:", {k: (tuple(v.shape), str(v.dtype)) if hasattr(v, "shape") else v for k, v in t.items()})
