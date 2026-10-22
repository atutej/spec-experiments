#!/usr/bin/env python
"""Acceptance length against request number for runs of the same request stream (Aurora's figures: x = request number, y = speculative accept length).
Each run dir holds rollouts.jsonl (one line per rollout batch, from tools/rollout_client.py). Prints the mean accept length per bin of requests, token-weighted by
the number of drafts is not available per batch, so the batch means are averaged with equal weight (all batches have the same size), and writes a PNG if matplotlib is present.
  compare_accept_curves.py --run online=<dir> --run static=<dir> [--bin 480] [--png out.png]"""
import argparse, json

p = argparse.ArgumentParser()
p.add_argument("--run", action="append", required=True, help="label=<run dir>")
p.add_argument("--bin", type=int, default=480, help="requests per bin")
p.add_argument("--png", default="")
a = p.parse_args()
runs = {}
for spec in a.run:
    label, d = spec.split("=", 1)
    rows = [json.loads(l) for l in open(f"{d}/rollouts.jsonl")]
    runs[label] = [(r["requests_done"], r["accept_length"], r["gen_tokens_per_s"]) for r in rows if r.get("status") == 200 and r.get("accept_length")]
labels = list(runs)
print(f"{'requests':>10s} " + " ".join(f"{l:>14s}" for l in labels))
top = max(x[0] for v in runs.values() for x in v)
for lo in range(0, top, a.bin):
    cells = []
    for l in labels:
        vals = [al for done, al, _ in runs[l] if lo < done <= lo + a.bin]
        cells.append(f"{sum(vals) / len(vals):14.3f}" if vals else f"{'-':>14s}")
    print(f"{lo + 1:>5d}-{lo + a.bin:<5d} " + " ".join(cells))
for l in labels:
    v = runs[l]; print(f"{l}: {len(v)} batches, mean accept length {sum(x[1] for x in v) / len(v):.3f}, mean generation {sum(x[2] for x in v) / len(v):,.0f} tokens/s")
if a.png:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for l in labels:
            xs = [x[0] for x in runs[l]]; ys = [x[1] for x in runs[l]]
            w = max(1, len(ys) // 40); sm = [sum(ys[max(0, i - w):i + 1]) / len(ys[max(0, i - w):i + 1]) for i in range(len(ys))]
            plt.plot(xs, sm, label=l)
        plt.xlabel("Request Number"); plt.ylabel("Speculative Accept Length"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(a.png, dpi=150); print("wrote", a.png)
    except ImportError:
        print("matplotlib not available: no PNG")
