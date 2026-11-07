"""One trainer dataset from Datakit rows with their ORIGINAL replies (itw part) and Dr. Doom's REGENERATED replies to other Datakit rows (targetgen part), both in the Datakit layout.

    python3 data/build_mix.py --out <new dir> [--itw-tokens 4.85e9] [--tg-rows 105000] [--val-itw-rows 3000] [--val-tg-rows 500] [--seed 0]

itw part   whole Datakit shards, picked at random (seed) until --itw-tokens tokens are reached; they are SYMLINKED, not copied. The Datakit shards are already random samples of the mixture
           (rows are bucketed by id hash). The last shard (the Datakit run's held-out rows, which the eval prompts come from) is never used for training.
targetgen  rows of the warm80k run's regenerated parts (runs/grug_dspark_drdoom_targetgen_warm80k/regen_parts: 250k rows, drawn from rows the itw sampler reaches after step 80,813), in a random
           part order, skipping rows whose source row lies in a chosen itw shard or in the last shard (no prompt appears twice), until --tg-rows (+ --val-tg-rows) rows are collected.
validation the LAST shard of the mixed dataset: --val-itw-rows rows of the Datakit last shard + --val-tg-rows held-out targetgen rows. The trainer's validation split is the last (1 - ratio) of the rows
           by position; manifest.json["train_data_ratio"] is the ratio that makes it exactly this shard (pass it as --train-data-ratio).
Vocabulary: token_freq.pt / .json are the Datakit dataset's own, so the draft vocabulary is the one of the itw runs (itw-80k, warm80k).
Training order does not depend on the order of the shards (the trainer's sampler permutes rows), only the validation split does.
"""
import argparse, glob, json, random, shutil, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from datasets import Dataset, Features, concatenate_datasets

DATAKIT = Path("/scratch/09749/atutej/marin_datakit/dataset_seq32768")
TG_PARTS = Path("/scratch/09749/atutej/marin_speculator/runs/grug_dspark_drdoom_targetgen_warm80k/regen_parts")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--itw-tokens", type=float, default=4.85e9)
ap.add_argument("--tg-rows", type=int, default=105_000)
ap.add_argument("--val-itw-rows", type=int, default=3000)
ap.add_argument("--val-tg-rows", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--datakit", default=str(DATAKIT)); ap.add_argument("--tg-parts", default=str(TG_PARTS))
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"refusing: {out} exists")
dk = Path(a.datakit)
info = json.loads((dk / "dataset_info.json").read_text())
features = Features.from_dict(info["features"])
shards = sorted(dk.glob("data-*.arrow"))
assert [f["filename"] for f in json.loads((dk / "state.json").read_text())["_data_files"]] == [s.name for s in shards], "state.json order differs from the sorted shard names"
rng = random.Random(a.seed)

def table(path):
    return ipc.open_stream(pa.memory_map(str(path))).read_all()

# global row offsets of the shards (the regen tool indexes rows over the sorted shards)
sizes = [table(s).num_rows for s in shards]
offsets = np.concatenate([[0], np.cumsum(sizes)])
assert offsets[-1] == json.loads((dk / "manifest.json").read_text())["totals"]["rows_kept"]

# itw part: random whole shards (never the last one) until the token budget is reached
order = list(range(len(shards) - 1)); rng.shuffle(order)
chosen, tokens = [], 0
for k in order:
    if tokens >= a.itw_tokens: break
    chosen.append(k); tokens += int(np.asarray(table(shards[k])["seq_len"]).sum())
chosen.sort()
itw_rows = int(sum(sizes[k] for k in chosen))
print(f"itw: {len(chosen)} shards, {itw_rows:,} rows, {tokens/1e9:.3f}B tokens", flush=True)
excluded = [(int(offsets[k]), int(offsets[k + 1])) for k in chosen] + [(int(offsets[-2]), int(offsets[-1]))]
def is_excluded(row):
    return any(lo <= row < hi for lo, hi in excluded)

# targetgen part
files = sorted(glob.glob(str(Path(a.tg_parts) / "part-*.parquet"))); rng.shuffle(files)
need = a.tg_rows + a.val_tg_rows
tg, skipped_overlap, skipped_empty = [], 0, 0
for f in files:
    t = pq.read_table(f, columns=["row", "input_ids", "spans"]).to_pydict()
    for row, ids, spans in zip(t["row"], t["input_ids"], t["spans"]):
        if len(spans) == 0: skipped_empty += 1; continue
        if is_excluded(int(row)): skipped_overlap += 1; continue
        m = np.zeros(len(ids), dtype=np.int64)
        for s, e in zip(spans[0::2], spans[1::2]): m[s:e] = 1
        tg.append({"input_ids": np.asarray(ids, dtype=np.int64), "loss_mask": m, "seq_len": len(ids)})
        if len(tg) == need: break
    if len(tg) == need: break
assert len(tg) == need, f"only {len(tg)} targetgen rows available (need {need})"
print(f"targetgen: {len(tg):,} rows ({skipped_overlap:,} skipped because their prompt is in an itw shard or the Datakit validation shard, {skipped_empty} empty)", flush=True)
tg_ds = Dataset.from_list(tg, features=features).shuffle(seed=a.seed); del tg
tg_val, tg_train = tg_ds.select(range(a.val_tg_rows)), tg_ds.select(range(a.val_tg_rows, len(tg_ds)))
tg_tokens = int(np.asarray(tg_train["seq_len"]).sum())

out.mkdir(parents=True)
tmp = out / ".tmp"
tg_train.save_to_disk(str(tmp / "tg"), num_shards=max(1, len(tg_train) // 30000))
last = Dataset.from_file(str(shards[-1]))
val = concatenate_datasets([last.select(range(a.val_itw_rows)), tg_val]).shuffle(seed=a.seed)
val.save_to_disk(str(tmp / "val"), num_shards=1)
files_out = []
for k in chosen:                                   # symlinks: the itw part is not copied
    name = f"data-itw-{k:05d}.arrow"; (out / name).symlink_to(shards[k].resolve()); files_out.append(name)
for i, f in enumerate(sorted((tmp / "tg").glob("data-*.arrow"))):
    name = f"data-tg-{i:05d}.arrow"; shutil.move(str(f), out / name); files_out.append(name)
name = "data-zval-00000.arrow"; shutil.move(str(next((tmp / "val").glob("data-*.arrow"))), out / name); files_out.append(name)   # LAST: the trainer's validation split
shutil.rmtree(tmp)
n_train = itw_rows + len(tg_train); n_total = n_train + len(val)
ratio = (n_train + 0.5) / n_total                  # int(n_total * ratio) == n_train
state = json.loads((dk / "state.json").read_text()); state["_data_files"] = [{"filename": f} for f in files_out]
(out / "state.json").write_text(json.dumps(state, indent=2))
info["splits"]["train"]["num_examples"] = n_total; info["splits"]["train"]["num_bytes"] = 0
(out / "dataset_info.json").write_text(json.dumps(info, indent=2))
for f in ("token_freq.pt", "token_freq.json"): shutil.copy2(dk / f, out / f)
(out / "manifest.json").write_text(json.dumps({"rows_total": n_total, "rows_train": n_train, "rows_val": len(val), "train_data_ratio": ratio, "itw_shards": [shards[k].name for k in chosen], "itw_rows": itw_rows,
    "itw_tokens": tokens, "tg_rows": len(tg_train), "tg_tokens": tg_tokens, "val_itw_rows": a.val_itw_rows, "val_tg_rows": a.val_tg_rows, "seed": a.seed, "datakit": str(dk), "tg_parts": a.tg_parts}, indent=1))
print(f"wrote {n_total:,} rows ({n_train:,} train: {itw_rows:,} itw + {len(tg_train):,} targetgen, {tokens/1e9:.2f}B + {tg_tokens/1e9:.2f}B tokens; {len(val):,} validation) to {out}; train_data_ratio {ratio:.9f}")
