"""Stage 2: nearest Datakit rows per RL prompt (exact, on one GPU), then the round-robin selection. Run in any env with torch + numpy.

  python select_rows.py --emb EMB_DIR --out SEL_DIR [--topk 64] [--token-budget 9.7e9] [--seed 0]

Reads rl.vec.npy and every finished shard-*.vec.npy / .seq.npy of EMB_DIR (stage 1), as a universe of rows numbered in shard order.
1. knn: for every RL prompt the TOPK most similar rows (cosine = dot product of the normalised vectors; brute force in chunks, so it is exact).
2. selection: round r takes, for every RL prompt, its r-th nearest row, until the sum of the rows' packed lengths (seq_len) reaches --token-budget.
   The round the budget runs out in is filled in a seeded random order of the RL prompts. No similarity cutoff. No deduplication unless --dedup
   (then a row already taken is skipped and that prompt's next-nearest row is used); the duplicates in the selection are counted either way.
SEL_DIR must be new. It gets knn.ids.npy (int32 global row, queries x TOPK), knn.scores.npy (float16), selection.npz (global_row, rank, rl_index,
score, seq_len; in selection order), universe.json (shard row ranges) and summary.json.
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import torch


def load_universe(emb: Path):
    done = sorted(emb.glob("shard-*.done.json"))
    if not done:
        sys.exit(f"no finished shards in {emb}")
    vecs, seqs, ranges, start = [], [], [], 0
    for d in done:
        stem = d.name.removesuffix(".done.json")
        meta = json.loads(d.read_text())
        v, s = np.load(emb / f"{stem}.vec.npy", mmap_mode="r"), np.load(emb / f"{stem}.seq.npy")
        assert len(v) == len(s) == meta["rows"], f"{stem}: {len(v)} vectors, {len(s)} lengths, meta says {meta['rows']}"
        vecs.append(v); seqs.append(s)
        ranges.append({"shard": int(stem.split("-")[1]), "shard_file": meta["shard_file"], "start": start, "rows": meta["rows"]})
        start += meta["rows"]
    return vecs, np.concatenate(seqs), ranges


@torch.no_grad()
def knn(q: np.ndarray, vecs, topk: int, dev="cuda", qchunk=2048, dchunk=262144):
    """Exact top-k of every query row against all document rows (cosine on normalised fp16 vectors)."""
    n = len(q)
    best_s = torch.full((n, topk), -2.0, dtype=torch.float32, device=dev)
    best_i = torch.zeros((n, topk), dtype=torch.int64, device=dev)
    qt = torch.from_numpy(q).to(dev)
    base = 0
    for v in vecs:
        for lo in range(0, len(v), dchunk):
            d = torch.from_numpy(np.ascontiguousarray(v[lo:lo + dchunk])).to(dev)
            for a in range(0, n, qchunk):
                s = (qt[a:a + qchunk] @ d.T).float()
                top_s, top_i = s.topk(min(topk, s.shape[1]), dim=1)
                cs = torch.cat([best_s[a:a + qchunk], top_s], 1)
                ci = torch.cat([best_i[a:a + qchunk], top_i + base + lo], 1)
                bs, bi = cs.topk(topk, dim=1)
                best_s[a:a + qchunk], best_i[a:a + qchunk] = bs, ci.gather(1, bi)
        base += len(v)
        print(f"  searched {base:,} rows", flush=True)
    return best_i.cpu().numpy().astype(np.int32), best_s.cpu().numpy().astype(np.float16)


def select(ids, scores, seq, budget, seed, dedup):
    n, k = ids.shape
    rng = np.random.default_rng(seed)
    taken, total = [], 0
    seen = set()
    chosen = {"global_row": [], "rank": [], "rl_index": [], "score": []}
    used = np.zeros(n, dtype=np.int32)                          # per prompt: next neighbour column to use (only moves with --dedup)
    full_rounds, stopped = 0, False
    for r in range(k):
        order = np.arange(n)
        pending = []
        for i in order:
            c = r if not dedup else used[i]
            if dedup:
                while c < k and int(ids[i, c]) in seen:
                    c += 1
                if c >= k:
                    continue
                used[i] = c + 1
            pending.append((int(i), int(c)))
        if total + sum(int(seq[ids[i, c]]) for i, c in pending) <= budget:
            part, stopped_here = pending, False
        else:
            perm = rng.permutation(len(pending)); part = []
            for p in perm:
                i, c = pending[p]; L = int(seq[ids[i, c]])
                if total + L > budget:
                    continue                                   # skip rows that no longer fit; a shorter one later in the shuffle may
                part.append((i, c)); total += L
            stopped_here = True
        if not stopped_here:
            total += sum(int(seq[ids[i, c]]) for i, c in part)
            full_rounds += 1
        for i, c in part:
            g = int(ids[i, c]); seen.add(g)
            chosen["global_row"].append(g); chosen["rank"].append(c + 1); chosen["rl_index"].append(i); chosen["score"].append(float(scores[i, c]))
        if stopped_here:
            stopped = True; break
    if not stopped:
        print(f"WARNING: top-{k} was exhausted at {total:,} tokens, below the budget {budget:,}; raise --topk", flush=True)
    return {k_: np.array(v) for k_, v in chosen.items()}, total, full_rounds, stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=64); ap.add_argument("--token-budget", type=float, default=9.7e9)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dedup", action="store_true")
    a = ap.parse_args()
    emb, out = Path(a.emb), Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing: {out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    print(f"selection goes to {out.resolve()}", flush=True)
    q = np.load(emb / "rl.vec.npy")
    vecs, seq, ranges = load_universe(emb)
    print(f"{len(q):,} RL prompts, {len(seq):,} candidate rows in {len(ranges)} shards, {int(seq.sum()):,} tokens", flush=True)
    if len(seq) < a.topk:
        sys.exit("fewer candidate rows than --topk")
    ids, scores = knn(q, vecs, a.topk)
    np.save(out / "knn.ids.npy", ids); np.save(out / "knn.scores.npy", scores)
    chosen, total, full_rounds, stopped = select(ids, scores, seq, int(a.token_budget), a.seed, a.dedup)
    chosen["seq_len"] = seq[chosen["global_row"]]
    np.savez(out / "selection.npz", **chosen)
    (out / "universe.json").write_text(json.dumps(ranges))
    uniq = len(np.unique(chosen["global_row"]))
    summary = {"rl_prompts": len(q), "universe_rows": len(seq), "universe_tokens": int(seq.sum()), "topk": a.topk, "token_budget": int(a.token_budget), "seed": a.seed, "dedup": a.dedup,
               "selected_rows": len(chosen["global_row"]), "unique_rows": int(uniq), "duplicate_rows": len(chosen["global_row"]) - int(uniq), "selected_tokens": int(total),
               "complete_rounds": full_rounds, "last_round_partial": stopped, "mean_score_by_rank": {int(r): float(chosen["score"][chosen["rank"] == r].mean()) for r in np.unique(chosen["rank"])[:: max(1, len(np.unique(chosen["rank"])) // 8)]},
               "selected_seq_len_percentiles": {p: int(np.percentile(chosen["seq_len"], p)) for p in (10, 50, 90, 99)}, "universe_seq_len_percentiles": {p: int(np.percentile(seq, p)) for p in (10, 50, 90, 99)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
