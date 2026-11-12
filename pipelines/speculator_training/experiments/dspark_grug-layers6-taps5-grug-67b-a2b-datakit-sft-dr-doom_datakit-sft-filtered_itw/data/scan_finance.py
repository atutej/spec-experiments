#!/usr/bin/env python
"""Find the finance_v1 rows of the Datakit dataset (token-id match of the task's opening instruction) and record their lengths. READ ONLY: writes only <out-dir>/finance_rows.npz.

  python scan_finance.py --out-dir <NEW dir> [--workers 64]

Match: the task prompt starts "<|start_header_id|>user<|end_header_id|>\nYou are given a financial text extracted from 10-K or 10-Q..."; the first KEY_LEN token ids from the user header on are taken from a
reference row (shard 7, row 22) and searched in the first 96 tokens of every row. Output arrays: shard (index), local (row inside the shard), seq_len (tokens in the row, replies included).
The result is compared with manifest.json (nemotron_sft_v3/finance_v1/train rows_kept).
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pyarrow as pa

D = Path("/scratch/09749/atutej/marin_datakit/dataset_seq32768"); HDR, KEY_LEN, HEAD = 128006, 22, 96
SHARDS = sorted(D.glob("data-*.arrow"))


def reference_key():
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom", trust_remote_code=True)
    ids = np.asarray(Dataset.from_file(str(SHARDS[7]))[22]["input_ids"]); user = tok.encode("user", add_special_tokens=False)[0]
    p = next(i for i in range(len(ids) - 1) if ids[i] == HDR and ids[i + 1] == user)
    key = ids[p : p + KEY_LEN]; text = tok.decode(list(key)); assert "financial text" in text or "You are given" in text, text
    print("key:", repr(text), flush=True); return key


def scan(args):
    k, key = args; out_local, out_len = [], []; base = 0
    with pa.memory_map(str(SHARDS[k])) as src:
        for batch in pa.ipc.open_stream(src):
            arr = batch.column(0); off = arr.offsets.to_numpy(); vals = arr.values.to_numpy(zero_copy_only=False); n = len(arr)
            lens = np.diff(off); idx = np.minimum(off[:-1, None] + np.arange(HEAD), np.maximum(off[1:, None] - 1, 0)); M = vals[idx]
            hit = np.zeros(n, bool)
            for o in range(HEAD - KEY_LEN + 1): hit |= (M[:, o : o + KEY_LEN] == key).all(1)
            hit &= lens >= HEAD
            out_local += (np.flatnonzero(hit) + base).tolist(); out_len += lens[hit].tolist(); base += n
    return k, base, out_local, out_len


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True); ap.add_argument("--workers", type=int, default=64); a = ap.parse_args()
    out = Path(a.out_dir)
    if (out / "finance_rows.npz").exists(): sys.exit("exists: old outputs are never overwritten")
    out.mkdir(parents=True, exist_ok=True); key = reference_key(); t0 = time.time(); sh, lo, ln, rows = [], [], [], 0
    with Pool(a.workers) as p:
        for k, n, l, s in p.imap_unordered(scan, [(k, key) for k in range(len(SHARDS))]):
            sh += [k] * len(l); lo += l; ln += s; rows += n
    o = np.lexsort((lo, sh)); sh, lo, ln = np.array(sh)[o], np.array(lo)[o], np.array(ln)[o]
    np.savez(out / "finance_rows.npz", shard=sh, local=lo, seq_len=ln)
    want = json.loads((D / "manifest.json").read_text())["per_source"]["nemotron_sft_v3/finance_v1/train"]["rows_kept"]
    print(f"scanned {rows:,} rows in {time.time() - t0:.0f}s; finance rows matched {len(ln):,} (manifest finance_v1: {want:,})")
    print("finance row length quantiles (tokens, replies included): " + ", ".join(f"p{q}={int(np.percentile(ln, q)):,}" for q in (1, 5, 10, 25, 50, 75, 90, 99)))
    print(f"{'threshold':>10}{'rows >= thr':>14}{'share':>8}{'tokens removed':>16}{'share of dataset tokens':>25}")
    tot = 45_991_968_601
    for thr in (8000, 16000, 20000, 24000, 28000, 30000, 32000, 32700):
        m = ln >= thr; print(f"{thr:10,}{int(m.sum()):14,}{m.mean() * 100:7.1f}%{int(ln[m].sum()):16,}{ln[m].sum() / tot * 100:24.1f}%")


if __name__ == "__main__":
    main()
