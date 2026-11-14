#!/usr/bin/env python
"""Do the training prompts contain the eval problems?  13-token n-grams of each eval problem's text (chat template and the fixed instruction suffix stripped) are
looked up in every non-supervised window of every training row.  A row 'contains' a problem when at least half of the problem's n-grams occur in the row's prompt side.
  python contam.py build          -> results/contam_index.npz   (eval problems, hashes)
  python contam.py scan [--only name] [--limit-shards N] -> results/contam_<dataset>.npz"""
import argparse, glob, json, os, re, sys, time
import numpy as np
W = "/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114"; R = "/scratch/09749/atutej/marin_speculator/runs"
TOK = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
SETS = {"GSM8K": f"{R}/paper_prompts_20261107/gsm8k", "MATH-500": f"{R}/paper_prompts_20261107/math500", "HumanEval": f"{R}/paper_prompts_20261107/humaneval",
        "MBPP": f"{R}/paper_prompts_20261107/mbpp", "MT-Bench": f"{R}/paper_prompts_20261107/mtbench", "AIME": f"{R}/aime_prompts_20261112/aime"}
DATA = {"datakit_orig": "/scratch/09749/atutej/marin_datakit/dataset_seq32768", "bigmix_tg": f"{R}/grug_dspark_drdoom_grug6t5_bigmix_targetgen296k/data",
        "datakit_tg": f"{R}/grug_dspark_drdoom_grug6t5_datakit_targetgen281k/data", "bigmix_orig": "/scratch/09749/atutej/marin_big_pool/dataset_itw_seq32768"}
NG = int(os.environ.get("CONTAM_NG", "13")); SUF = "" if NG == 13 else f"_n{NG}"; P = np.uint64(1000003); MIN_TOK = 30   # CONTAM_NG=6 finds reworded variants, not only copies
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
def ngram_hash(a, n=NG):
    a = a.astype(np.uint64); k = len(a) - n + 1; h = np.zeros(k, np.uint64)
    for j in range(n): h = h * P + a[j:j + k]
    return h
def build():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True); names, texts, hs, ids, setid = [], [], [], [], []
    for si, (name, d) in enumerate(SETS.items()):
        seen = set()
        for line in open(f"{d}/first_turn.jsonl"):
            r = json.loads(line); key = r.get("index", len(seen))
            if key in seen: continue
            seen.add(key); s = tok.decode(r["prompt"], skip_special_tokens=False)
            m = re.search(r"user<\|end_header_id\|>\n(.*)<\|eot_id\|>\n<\|start_header_id\|>assistant", s, re.S)
            if not m: continue
            c = m.group(1)
            if c.endswith(SUFFIX): c = c[:-len(SUFFIX)]
            if name == "HumanEval":
                c = re.sub(r"^Write a solution to the following problem and make sure that it passes the tests:\n```python\n", "", c); c = re.sub(r"\n```$", "", c)
            t = np.array(tok.encode(c, add_special_tokens=False), dtype=np.int64)
            if t.size < MIN_TOK: continue
            h = np.unique(ngram_hash(t)); names.append(name); texts.append(c[:200]); hs.append(h); setid.append(si)
    allh = np.concatenate(hs); alli = np.concatenate([np.full(h.size, i, np.int32) for i, h in enumerate(hs)]); o = np.argsort(allh, kind="stable"); allh, alli = allh[o], alli[o]
    keep = np.concatenate([[True], allh[1:] != allh[:-1]]); allh, alli = allh[keep], alli[keep]   # an n-gram shared by two problems keeps the first
    np.savez(f"{W}/results/contam_index{SUF}.npz", hashes=allh, owner=alli, n_ngrams=np.array([h.size for h in hs]), set_of=np.array(names), text=np.array(texts))
    print("eval problems:", {n: names.count(n) for n in SETS}, "| distinct n-grams", allh.size)
G = {}
def _init():
    z = np.load(f"{W}/results/contam_index{SUF}.npz", allow_pickle=True); G.update(h=z["hashes"], o=z["owner"], n=z["n_ngrams"])
def scan_shard(args):
    path, si = args
    import pyarrow as pa, pyarrow.compute as pc
    H, O, NN = G["h"], G["o"], G["n"]; E = len(NN)
    best = np.zeros(E); n50 = np.zeros(E, np.int64); n90 = np.zeros(E, np.int64); ex = np.full((E, 2), -1, np.int64); rows = 0; t0 = time.time()
    for batch in pa.ipc.open_stream(pa.memory_map(path)):
        ic, mc = batch.column(0), batch.column(1)
        lens = pc.list_value_length(ic).to_numpy(zero_copy_only=False).astype(np.int64); offs = np.concatenate([[0], np.cumsum(lens)])
        flat = pc.list_flatten(ic).to_numpy(zero_copy_only=False).astype(np.int64); mf = (pc.list_flatten(mc).to_numpy(zero_copy_only=False) != 0).astype(np.int64)
        for r in range(len(lens)):
            L = int(lens[r]); rows += 1
            if L < NG: continue
            o = int(offs[r]); ids = flat[o:o + L]; m = mf[o:o + L]; h = ngram_hash(ids); cs = np.concatenate([[0], np.cumsum(m)]); valid = (cs[NG:] - cs[:-NG]) == 0
            hv = h[valid]
            if hv.size == 0: continue
            pos = np.searchsorted(H, hv); pos[pos >= H.size] = H.size - 1; hit = H[pos] == hv
            if not hit.any(): continue
            u = np.unique(hv[hit]); own, cnt = np.unique(O[np.searchsorted(H, u)], return_counts=True)   # distinct eval n-grams found, per eval problem
            for e, c in zip(own, cnt):
                cov = c / NN[e]
                if cov > best[e]: best[e] = cov; ex[e] = (si, rows - 1)
                n50[e] += int(cov >= 0.5); n90[e] += int(cov >= 0.9)
    return best, n50, n90, ex, rows, time.time() - t0
def scan(only, limit_shards, workers):
    from multiprocessing import Pool
    for name, d in DATA.items():
        if only and name != only: continue
        shards = sorted(glob.glob(f"{d}/data-*.arrow"))[:limit_shards or None]; t0 = time.time()
        with Pool(min(workers, len(shards)), initializer=_init) as pool: res = pool.map(scan_shard, [(s, i) for i, s in enumerate(shards)], chunksize=1)
        best = np.max([r[0] for r in res], axis=0); n50 = np.sum([r[1] for r in res], axis=0); n90 = np.sum([r[2] for r in res], axis=0)
        k = np.argmax([r[0] for r in res], axis=0); ex = np.stack([res[k[e]][3][e] for e in range(len(best))])
        np.savez(f"{W}/results/contam{SUF}_{name}{'_test' if limit_shards else ''}.npz", best=best, n50=n50, n90=n90, ex=ex, shards=np.array(shards), rows=sum(r[4] for r in res))
        print(f"{name}: {len(shards)} shards, {sum(r[4] for r in res)} rows, problems with a row covering >=50%: {int((n50>0).sum())}, >=90%: {int((n90>0).sum())}  ({time.time()-t0:.0f}s)", flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "scan"]); ap.add_argument("--only"); ap.add_argument("--limit-shards", type=int, default=0); ap.add_argument("--workers", type=int, default=100)
    a = ap.parse_args(); build() if a.cmd == "build" else scan(a.only, a.limit_shards, a.workers)
