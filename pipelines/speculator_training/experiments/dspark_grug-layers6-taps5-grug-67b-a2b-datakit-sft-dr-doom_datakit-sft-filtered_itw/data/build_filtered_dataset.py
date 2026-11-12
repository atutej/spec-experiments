#!/usr/bin/env python
"""Copy of the Datakit SFT dataset WITHOUT the finance_v1 rows (nemotron_sft_v3/finance_v1/train, 326,695 rows), same format, same shard names, same row order.

  python build_filtered_dataset.py --scan <scan dir with finance_rows.npz> --out <NEW dir> [--workers 64] [--only-shard K ...]

SRC is READ ONLY. Each shard is rewritten without its finance rows (the list comes from scan_finance.py), to <out>/data-XXXXX-of-00215.arrow via a temporary name, so a rerun skips finished shards.
Per shard the script also records rows kept, tokens kept, supervised tokens kept and the supervised-token frequencies (.shard_stats/). When all shards exist it writes state.json, dataset_info.json,
manifest.json (the finance leaf removed, totals recomputed) and token_freq.json; then run data/finalize_dataset.py on <out> (writes token_freq.pt and checks the directory like the trainer loads it).
Checks (the run stops if one fails): rows removed == manifest finance rows; tokens removed == the finance leaf's tokens_kept; supervised tokens kept == total supervised - the leaf's supervised.
The validation split of the trainer is the LAST 0.1% of the rows, by position: the old validation rows without finance stay at the end, so the held-out eval prompts (taken from the old validation rows) remain held out.
"""
import argparse, json, os, shutil, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pyarrow as pa

SRC = Path("/scratch/09749/atutej/marin_datakit/dataset_seq32768"); V = 128256
LEAF = "nemotron_sft_v3/finance_v1/train"


def build_shard(args):
    k, out_dir, drop_local = args
    src_f = sorted(SRC.glob("data-*.arrow"))[k]; out_f = Path(out_dir) / src_f.name; stats_f = Path(out_dir) / ".shard_stats" / f"{k:05d}.json"
    if out_f.exists() and stats_f.exists(): return k, json.loads(stats_f.read_text()), True
    tmp = Path(str(out_f) + ".tmp"); drop = set(int(x) for x in drop_local); freq = np.zeros(V, np.int64); kept = removed = tok_kept = sup_kept = tok_removed = 0; base = 0
    with pa.memory_map(str(src_f)) as src:
        reader = pa.ipc.open_stream(src)
        with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_stream(sink, reader.schema) as w:
            for batch in reader:
                n = batch.num_rows; keep = np.ones(n, bool)
                for i in [d - base for d in drop if base <= d < base + n]: keep[i] = False
                lens = batch.column(2).to_numpy(zero_copy_only=False)
                tok_removed += int(lens[~keep].sum()); removed += int((~keep).sum())
                if keep.all(): fb = batch
                else: fb = batch.filter(pa.array(keep))
                if fb.num_rows:
                    ids = fb.column(0).flatten().to_numpy(zero_copy_only=False); m = fb.column(1).flatten().to_numpy(zero_copy_only=False)
                    sup = ids[m == 1]; freq += np.bincount(sup, minlength=V)[:V]; sup_kept += len(sup); tok_kept += int(fb.column(2).to_numpy(zero_copy_only=False).sum()); kept += fb.num_rows
                    w.write_batch(fb)
                base += n
    assert removed == len(drop), (k, removed, len(drop))
    os.replace(tmp, out_f); stats_f.parent.mkdir(exist_ok=True)
    np.save(Path(out_dir) / ".shard_stats" / f"{k:05d}.npy", freq)
    st = {"rows_kept": kept, "rows_removed": removed, "tokens_kept": tok_kept, "tokens_removed": tok_removed, "sup_kept": sup_kept}; stats_f.write_text(json.dumps(st))
    return k, st, False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scan", required=True); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=64); ap.add_argument("--only-shard", type=int, nargs="*")
    a = ap.parse_args(); out = Path(a.out); marker = out / ".filtered_dataset"
    if out.exists() and not marker.exists(): sys.exit(f"refusing: {out} exists and was not made by this script")
    out.mkdir(parents=True, exist_ok=True); marker.write_text(f"Datakit SFT without {LEAF}; source {SRC}; made {time.ctime()}\n")
    z = np.load(Path(a.scan) / "finance_rows.npz"); nshard = len(sorted(SRC.glob("data-*.arrow")))
    drops = {k: z["local"][z["shard"] == k] for k in range(nshard)}
    todo = [(k, str(out), drops[k]) for k in (a.only_shard if a.only_shard else range(nshard))]
    t0 = time.time(); res = {}
    with Pool(a.workers) as p:
        for i, (k, st, skipped) in enumerate(p.imap_unordered(build_shard, todo)):
            res[k] = st
            if i % 10 == 0 or i == len(todo) - 1: print(f"[{i + 1}/{len(todo)}] shard {k}{' (skipped, done)' if skipped else ''} kept {st['rows_kept']:,} removed {st['rows_removed']:,}  elapsed {time.time() - t0:.0f}s", flush=True)
    if a.only_shard: print("partial run (--only-shard): metadata not written"); return
    man = json.loads((SRC / "manifest.json").read_text()); leaf = man["per_source"][LEAF]; T = man["totals"]
    rows_kept = sum(s["rows_kept"] for s in res.values()); tok_kept = sum(s["tokens_kept"] for s in res.values()); sup_kept = sum(s["sup_kept"] for s in res.values()); removed = sum(s["rows_removed"] for s in res.values()); tok_removed = sum(s["tokens_removed"] for s in res.values())
    assert removed == leaf["rows_kept"], (removed, leaf["rows_kept"]); assert tok_removed == leaf["tokens_kept"], (tok_removed, leaf["tokens_kept"])
    assert rows_kept == T["rows_kept"] - leaf["rows_kept"] and tok_kept == T["tokens_kept"] - leaf["tokens_kept"], "totals do not add up"
    assert sup_kept == T["tokens_supervised"] - leaf["tokens_supervised"], (sup_kept, T["tokens_supervised"] - leaf["tokens_supervised"])
    freq = sum(np.load(f) for f in sorted((out / ".shard_stats").glob("*.npy")))
    (out / "token_freq.json").write_text(json.dumps({str(i): int(c) for i, c in enumerate(freq) if c}))
    man["per_source"] = {k: v for k, v in man["per_source"].items() if k != LEAF}
    for key in ("rows_kept", "tokens_kept", "tokens_supervised", "rows_clipped", "tokens_clipped", "dropped_unsupervised"): T[key] = T[key] - leaf.get(key, 0)
    man["filter"] = {"removed_source": LEAF, "removed_rows": removed, "removed_tokens": tok_removed, "source_dataset": str(SRC)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    info = json.loads((SRC / "dataset_info.json").read_text()); size = sum(f.stat().st_size for f in out.glob("data-*.arrow"))
    info["dataset_size"] = info["download_size"] = info["size_in_bytes"] = size; info["splits"]["train"]["num_bytes"] = size; info["splits"]["train"]["num_examples"] = rows_kept
    (out / "dataset_info.json").write_text(json.dumps(info, indent=1)); shutil.copy(SRC / "state.json", out / "state.json")
    print(f"done: {rows_kept:,} rows, {tok_kept / 1e9:.2f}B tokens, {sup_kept / 1e9:.2f}B supervised; removed {removed:,} rows / {tok_removed / 1e9:.2f}B tokens; {size / 1e9:.0f} GB; now run data/finalize_dataset.py on {out}", flush=True)


if __name__ == "__main__":
    main()
