#!/usr/bin/env python
"""Compare training-data statistics of Datakit-original, BigMix-targetgen (and two controls), prompts and completions.
  python datacmp.py resources          # token classes, think ids, the five draft-vocabulary masks -> results/resources.npz
  python datacmp.py stats [--limit-rows N] [--only name]   # one pass over shards in parallel -> results/stats_<dataset>.npz
Sub-commands read only; the datasets are never written."""
import argparse, glob, json, os, re, struct, sys, time, zlib
import numpy as np
W = "/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114"; R = "/scratch/09749/atutej/marin_speculator/runs"
TOK = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
CKPT = {"Datakit orig 24h": f"{R}/eval_ckpts/drdoom_grug6_taps5_itw_step75834", "NoFin orig 24h": f"{R}/eval_ckpts/datakit_nofinance6_taps5_itw_step73117",
        "BigMix orig 24h": f"{R}/eval_ckpts/bigmix6_taps5_itw_step95146", "BigMix TG 28h": f"{R}/eval_ckpts/bigmix6_taps5_targetgen_epoch0_end",
        "Datakit TG 25h": f"{R}/eval_ckpts/datakit6_taps5_targetgen_epoch0_end"}
# dataset name -> (directory, shard stride, text-sample rate)
DATASETS = {"datakit_orig": ("/scratch/09749/atutej/marin_datakit/dataset_seq32768", 3, 0.012),
            "bigmix_tg": (f"{R}/grug_dspark_drdoom_grug6t5_bigmix_targetgen296k/data", 1, 0.05),
            "datakit_tg": (f"{R}/grug_dspark_drdoom_grug6t5_datakit_targetgen281k/data", 1, 0.05),
            "bigmix_orig": ("/scratch/09749/atutej/marin_big_pool/dataset_itw_seq32768", 1, 0.012)}
CLASSES = ["special", "CJK", "other non-ASCII", "digits", "whitespace/newline", "punctuation/operators", "latex/code symbols", "English word pieces", "mixed other"]
MARKERS = [r"\bWait\b", r"\bHmm\b", r"\bAlternatively\b", r"\bBut wait\b", r"\bLet me\b", r"\bActually\b", r"\bOkay\b"]
NCAT = 6   # mode (think, nothink, none) x terminal-agent prompt (no, yes) : cat = mode*2 + terminal
V = 128256
def read_tensor(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); m = hdr[name]; a, b = m["data_offsets"]; f.seek(8 + n + a); raw = f.read(b - a)
    return np.frombuffer(raw, dtype={"BOOL": np.bool_, "I64": np.int64}[m["dtype"]]).reshape(m["shape"])

def build_resources():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    cls = np.zeros(V, np.int8)
    for t in range(V):
        s = tok.decode([t], skip_special_tokens=False)
        if s.startswith("<|") and s.endswith("|>"): c = 0
        elif re.search(r"[぀-ヿ㐀-鿿가-힯]", s): c = 1
        elif re.search(r"[^\x00-\x7f]", s): c = 2
        elif re.fullmatch(r"\s*\d+\s*", s): c = 3
        elif s.strip() == "": c = 4
        elif re.search(r"[\\{}^_$]", s): c = 6
        elif re.search(r"[A-Za-z]", s) and not re.search(r"\d", s): c = 7
        elif re.search(r"[A-Za-z0-9]", s): c = 8
        else: c = 5
        cls[t] = c
    masks = np.stack([read_tensor(f"{d}/model.safetensors", "t2d").astype(bool) for d in CKPT.values()])
    phrase = np.array(tok.encode(" tasked with solving command-line tasks", add_special_tokens=False), dtype=np.int64)
    np.savez(f"{W}/results/resources.npz", cls=cls, st=tok.convert_tokens_to_ids("<|start_think|>"), en=tok.convert_tokens_to_ids("<|end_think|>"), masks=masks,
             mask_labels=np.array(list(CKPT)), class_names=np.array(CLASSES), terminal_phrase=phrase)
    print("classes:", {CLASSES[i]: int((cls == i).sum()) for i in range(len(CLASSES))}); print("vocab sizes:", masks.sum(1).tolist(), "| terminal phrase ids:", phrase.tolist())

# ---------------------------------------------------------------- worker
G = {}
def _init():
    from transformers import AutoTokenizer
    z = np.load(f"{W}/results/resources.npz", allow_pickle=True)
    G.update(tok=AutoTokenizer.from_pretrained(TOK, trust_remote_code=True), cls=z["cls"].astype(np.int64), st=int(z["st"]), en=int(z["en"]), masks=z["masks"], phrase=z["terminal_phrase"], hdr={})
P = np.uint64(1000003)
def ngram_hash(a, n=8):
    a = a.astype(np.uint64); k = len(a) - n + 1; h = np.zeros(k, np.uint64)
    for j in range(n): h = h * P + a[j:j + k]
    return h
def lb(x): return int(min(19, np.floor(np.log2(x + 1))))
def new_acc(nmask):
    z = lambda *s: np.zeros(s, np.int64)
    return dict(rows=z(NCAT), empty=z(NCAT), tokens=z(NCAT), sup=z(NCAT), prompt=z(NCAT), turns=z(NCAT), window=z(NCAT), thinkrows=z(NCAT), think_tok=z(NCAT), unclosed=z(NCAT),
                cp_ctx=z(NCAT), cp_self=z(NCAT), cls_hist=z(NCAT, len(CLASSES)), cov=z(NCAT, nmask), sup_hist=z(NCAT, 20), prompt_hist=z(NCAT, 20), think_hist=z(NCAT, 20),
                uncov_cls=z(nmask, len(CLASSES)), uni=z(3, V), txt_rows=z(NCAT), txt_bytes=z(NCAT), txt_zbytes=z(NCAT), txt_sup=z(NCAT), markers=z(NCAT, len(MARKERS)),
                f_boxed=z(NCAT), f_step=z(NCAT), f_fence=z(NCAT), f_latex=z(NCAT), f_nonascii=z(NCAT), f_prompt600=z(NCAT))
def process(args):
    path, rate, limit, seed = args
    import pyarrow as pa, pyarrow.compute as pc
    tok, cls, ST, EN, masks, phrase, hdr = G["tok"], G["cls"], G["st"], G["en"], G["masks"], G["phrase"], G["hdr"]
    rng = np.random.default_rng(seed); A = new_acc(len(masks)); t0 = time.time(); nrows = 0
    mk = [re.compile(p) for p in MARKERS]
    for batch in pa.ipc.open_stream(pa.memory_map(path)):
        ic, mc = batch.column(0), batch.column(1)
        lens = pc.list_value_length(ic).to_numpy(zero_copy_only=False).astype(np.int64); offs = np.concatenate([[0], np.cumsum(lens)])
        flat = pc.list_flatten(ic).to_numpy(zero_copy_only=False).astype(np.int64); mflat = pc.list_flatten(mc).to_numpy(zero_copy_only=False) != 0
        sups, modes = [], []
        for r in range(len(lens)):
            L = int(lens[r]); o = int(offs[r]); ids = flat[o:o + L]; m = mflat[o:o + L]
            # category: reasoning mode from the header, terminal-agent prompt from its fixed phrase
            key = tuple(ids[:20].tolist()); mode = hdr.get(key)
            if mode is None:
                s = tok.decode(list(key), skip_special_tokens=False); mode = 1 if "/nothink" in s else 0 if "/think" in s else 2
                if len(hdr) < 200000: hdr[key] = mode
            term = 0
            if phrase.size:
                cand = np.flatnonzero(ids[:400] == phrase[0])
                term = int(any(c + phrase.size <= L and np.array_equal(ids[c:c + phrase.size], phrase) for c in cand))
            c = mode * 2 + term; A["rows"][c] += 1; A["tokens"][c] += L; sup = ids[m]
            if sup.size == 0: A["empty"][c] += 1; continue
            d = np.diff(m.astype(np.int8), prepend=0); starts = np.flatnonzero(d == 1); fs = int(starts[0])
            A["sup"][c] += sup.size; A["prompt"][c] += L - sup.size; A["turns"][c] += starts.size; A["window"][c] += int(L >= 32700)
            A["sup_hist"][c, lb(sup.size)] += 1; A["prompt_hist"][c, lb(L - sup.size)] += 1
            A["cls_hist"][c] += np.bincount(cls[sup], minlength=len(CLASSES))
            for k in range(len(masks)): A["cov"][c, k] += int(masks[k][sup].sum())
            sups.append(sup); modes.append(mode)
            sm = ids == ST
            if sm.any():
                em = ids == EN; depth = np.cumsum(sm.astype(np.int32) - em.astype(np.int32)); A["think_tok"][c] += int(((depth > 0) & m).sum()); A["thinkrows"][c] += 1; A["unclosed"][c] += int(depth[-1] > 0)
                i0 = int(np.argmax(sm)); j = np.flatnonzero(em[i0:]); A["think_hist"][c, lb(int(j[0]) if j.size else L - i0)] += 1
            if L >= 16:   # context copying: supervised positions covered by an 8-gram that occurred earlier in the row
                h = ngram_hash(ids); u, fi, inv = np.unique(h, return_index=True, return_inverse=True); first = fi[inv]; rep = first != np.arange(h.size)
                if rep.any():
                    last_sup = m[first + 7]; rep_ctx = rep & ~last_sup; rep_self = rep & last_sup
                    def cover(x):
                        cs = np.concatenate([[0], np.cumsum(x)]); p = np.arange(L); lo = np.clip(p - 7, 0, h.size - 1); hi = np.clip(p, 0, h.size - 1); return (cs[hi + 1] - cs[lo]) > 0
                    cc = cover(rep_ctx); cs_ = cover(rep_self) & ~cc; A["cp_ctx"][c] += int((cc & m).sum()); A["cp_self"][c] += int((cs_ & m).sum())
            if rng.random() < rate:   # text-level features on a sample of rows
                try:
                    pre = tok.decode(ids[:min(fs, 600)].tolist(), skip_special_tokens=False); suf = tok.decode(ids[max(0, fs - 70):fs].tolist(), skip_special_tokens=False)
                    txt = tok.decode(sup[:20000].tolist(), skip_special_tokens=False); raw = txt.encode("utf-8", "ignore")
                    A["txt_rows"][c] += 1; A["txt_bytes"][c] += len(raw); A["txt_zbytes"][c] += len(zlib.compress(raw, 6)); A["txt_sup"][c] += min(sup.size, 20000)
                    for q, rx in enumerate(mk): A["markers"][c, q] += len(rx.findall(txt))
                    A["f_boxed"][c] += int("boxed" in suf); A["f_step"][c] += int("step by step" in suf.lower() or "step-by-step" in suf.lower()); A["f_fence"][c] += int("```" in pre)
                    A["f_latex"][c] += int(("$" in pre) or ("\\(" in pre) or ("\\[" in pre)); A["f_prompt600"][c] += 1
                    A["f_nonascii"][c] += int(len(pre) > 0 and sum(ord(ch) > 127 for ch in pre) / len(pre) > 0.3)
                except Exception: pass
            nrows += 1
            if limit and nrows >= limit: break
        if sups:
            sa = np.concatenate(sups); ma = np.concatenate([np.full(s.size, md, np.int8) for s, md in zip(sups, modes)])
            for md in range(3): A["uni"][md] += np.bincount(sa[ma == md], minlength=V)
            for k in range(len(masks)): A["uncov_cls"][k] += np.bincount(cls[sa[~masks[k][sa]]], minlength=len(CLASSES))
        if limit and nrows >= limit: break
    A["seconds"] = np.array([time.time() - t0]); return A
def run_stats(only, limit, workers):
    from multiprocessing import Pool
    for name, (d, stride, rate) in DATASETS.items():
        if only and name != only: continue
        shards = sorted(glob.glob(f"{d}/data-*.arrow"))[::stride]; t0 = time.time()
        with Pool(min(workers, len(shards)), initializer=_init) as pool: parts = pool.map(process, [(s, rate, limit, i) for i, s in enumerate(shards)], chunksize=1)
        tot = {k: sum(p[k] for p in parts) for k in parts[0]}; np.savez(f"{W}/results/stats_{name}{'_test' if limit else ''}.npz", shards=len(shards), **tot)
        n = int(tot["rows"].sum()); print(f"{name}: {len(shards)} shards, {n} rows, tokens/row {tot['tokens'].sum()/n:.0f}, supervised/row {tot['sup'].sum()/n:.0f}, {time.time()-t0:.0f}s", flush=True)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["resources", "stats"]); ap.add_argument("--limit-rows", type=int, default=0); ap.add_argument("--only"); ap.add_argument("--workers", type=int, default=120)
    a = ap.parse_args()
    build_resources() if a.cmd == "resources" else run_stats(a.only, a.limit_rows, a.workers)
