"""One trainer dataset from Datakit ORIGINAL rows (free) and BigMix TARGETGEN rows (Dr. Doom's own replies, paid for), both in the Datakit layout, at their natural proportions.

    python data/build_mix.py --out <new dir> [--val-orig-rows 3000] [--val-tg-rows 300] [--seed 0] [--max-orig-shards N --max-tg-shards N]   (speculators env: needs datasets + pyarrow)

orig   the Datakit shards of spec_data/datakit_orig (215 shards, 46.0B tokens), SYMLINKED, not copied. The LAST shard is the held-out one (the eval prompts come from it): never trained on.
tg     the 9 shards of spec_data/bigmix_tg (296,000 rows, 3.22B tokens) REWRITTEN into the Datakit layout (int32 -> int64 input_ids, `source_row` dropped); the validation rows are taken out first.
valid  the LAST shard of the output: --val-orig-rows random rows of the Datakit last shard + --val-tg-rows random targetgen rows, shuffled. The trainer's validation split is the last
       (1 - ratio) of the rows BY POSITION; manifest.json["train_data_ratio"] is the ratio that makes it exactly this shard (the launcher passes it as --train-data-ratio).
Order    the on-disk order does not matter for training: the trainer permutes all training rows every epoch (distributed_batch_sampler.py), so the mix is uniform. It matters only for validation, hence "last".
Vocab    token_freq.pt / .json = Datakit's + BigMix-TG's counts, summed, so the 32k draft vocabulary follows the combined data. (The few validation rows stay in the counts: negligible.)
The --max-*-shards options build a small dataset for smoke tests.
"""
import argparse, json, random, shutil, sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from datasets import Dataset, Features, concatenate_datasets

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--datakit", default="/scratch/09749/atutej/spec_data/datakit_orig")
ap.add_argument("--tg", default="/scratch/09749/atutej/spec_data/bigmix_tg")
ap.add_argument("--val-orig-rows", type=int, default=3000)
ap.add_argument("--val-tg-rows", type=int, default=300)
ap.add_argument("--max-orig-shards", type=int, default=0, help="smoke: use only the first N training shards of Datakit")
ap.add_argument("--max-tg-shards", type=int, default=0, help="smoke: use only the first N shards of BigMix-TG")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"refusing: {out} exists")
dk, tg = Path(a.datakit), Path(a.tg)
rng = random.Random(a.seed)

def shard_names(d):
    names = [f["filename"] for f in json.loads((d / "state.json").read_text())["_data_files"]]
    assert names == sorted(p.name for p in d.glob("data-*.arrow")), f"{d}: state.json order differs from the sorted shard names"
    return [d / n for n in names]

dk_info = json.loads((dk / "dataset_info.json").read_text())
features = Features.from_dict(dk_info["features"])          # input_ids int64, loss_mask int64, seq_len int64
dk_shards, tg_shards = shard_names(dk), shard_names(tg)
orig_train = dk_shards[:-1][: a.max_orig_shards or None]      # never the last (held-out) shard
dk_last = dk_shards[-1]
tg_use = tg_shards[: a.max_tg_shards or None]

def seq_total(path):
    return int(np.asarray(ipc.open_stream(pa.memory_map(str(path))).read_all()["seq_len"]).sum())

def n_rows(path):
    return ipc.open_stream(pa.memory_map(str(path))).read_all().num_rows

def to_datakit_layout(ds):
    return ds.remove_columns([c for c in ds.column_names if c not in features]).cast(features)

# ---- targetgen: per-shard validation rows (random, spread over all shards), the rest rewritten ----
tg_sizes = [n_rows(p) for p in tg_use]
tg_total = sum(tg_sizes)
val_idx = {}
quota = [round(a.val_tg_rows * s / tg_total) for s in tg_sizes]
for i, (s, q) in enumerate(zip(tg_sizes, quota)):
    val_idx[i] = sorted(rng.sample(range(s), q))
print(f"targetgen: {len(tg_use)} shards, {tg_total:,} rows, {sum(quota)} held out for validation", flush=True)

out.mkdir(parents=True)
tmp = out / ".tmp"; tmp.mkdir()
files_out, tg_train_rows, tg_train_tokens, tg_val_parts = [], 0, 0, []
for i, p in enumerate(tg_use):
    ds = Dataset.from_file(str(p))
    held = set(val_idx[i])
    keep = [r for r in range(ds.num_rows) if r not in held]
    tg_val_parts.append(to_datakit_layout(ds.select(sorted(held))) if held else None)
    tr = to_datakit_layout(ds.select(keep))
    tg_train_rows += tr.num_rows; tg_train_tokens += int(np.asarray(tr["seq_len"]).sum())
    tr.save_to_disk(str(tmp / f"tg{i}"), num_shards=1)
    name = f"data-tg-{i:05d}.arrow"; shutil.move(str(next((tmp / f"tg{i}").glob("data-*.arrow"))), out / name); files_out.append(name)
    print(f"  tg shard {i}: {tr.num_rows:,} rows written", flush=True)
    del ds, tr

# ---- validation shard ----
last = Dataset.from_file(str(dk_last))
vo = sorted(rng.sample(range(last.num_rows), a.val_orig_rows))
val = concatenate_datasets([last.select(vo)] + [x for x in tg_val_parts if x is not None]).shuffle(seed=a.seed)
val.save_to_disk(str(tmp / "val"), num_shards=1)
val_rows = val.num_rows
shutil.move(str(next((tmp / "val").glob("data-*.arrow"))), out / "data-zval-00000.arrow")
shutil.rmtree(tmp)

# ---- originals: symlinks ----
orig_files, orig_rows, orig_tokens = [], 0, 0
for k, p in enumerate(orig_train):
    name = f"data-orig-{k:05d}.arrow"; (out / name).symlink_to(p.resolve()); orig_files.append(name)
    t = ipc.open_stream(pa.memory_map(str(p))).read_all(); orig_rows += t.num_rows; orig_tokens += int(np.asarray(t["seq_len"]).sum())
# on-disk order: originals, targetgen, validation LAST
all_files = orig_files + files_out + ["data-zval-00000.arrow"]
n_train = orig_rows + tg_train_rows
n_total = n_train + val_rows
ratio = (n_train + 0.5) / n_total                       # int(n_total * ratio) == n_train
state = json.loads((dk / "state.json").read_text()); state["_data_files"] = [{"filename": f} for f in all_files]
(out / "state.json").write_text(json.dumps(state, indent=2))
dk_info["splits"]["train"]["num_examples"] = n_total; dk_info["splits"]["train"]["num_bytes"] = 0; dk_info["download_size"] = 0; dk_info["dataset_size"] = 0
(out / "dataset_info.json").write_text(json.dumps(dk_info, indent=2))

# ---- summed token frequencies (supervised tokens; both files are {token id: count}) ----
freq = {}
for d in (dk, tg):
    f = torch.load(d / "token_freq.pt", weights_only=False)
    for k, v in f.items():
        freq[int(k)] = freq.get(int(k), 0) + int(v)
torch.save(freq, out / "token_freq.pt")
(out / "token_freq.json").write_text(json.dumps({str(k): v for k, v in sorted(freq.items())}))
manifest = {"rows_total": n_total, "rows_train": n_train, "rows_val": val_rows, "train_data_ratio": ratio,
            "orig_shards": len(orig_files), "orig_rows": orig_rows, "orig_tokens": orig_tokens,
            "tg_shards": len(tg_use), "tg_rows": tg_train_rows, "tg_tokens": tg_train_tokens,
            "tg_token_share": tg_train_tokens / (orig_tokens + tg_train_tokens),
            "val_orig_rows": a.val_orig_rows, "val_tg_rows": val_rows - a.val_orig_rows, "seed": a.seed,
            "datakit": str(dk), "tg": str(tg), "files": all_files}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"wrote {n_total:,} rows to {out}: {n_train:,} train ({orig_rows:,} orig = {orig_tokens/1e9:.2f}B tokens, {tg_train_rows:,} targetgen = {tg_train_tokens/1e9:.2f}B tokens, "
      f"targetgen share {manifest['tg_token_share']:.2%}) + {val_rows:,} validation; train_data_ratio {ratio:.9f}")
