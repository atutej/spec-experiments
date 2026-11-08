#!/usr/bin/env python3
"""Speedup of each drafter over its same-job plain (no-draft) arm, from an eval_matrix output base run with --plain.

  python speedup_report.py <out base> [--csv FILE]

Layout read: <out base>/<benchmark>/<label>/... and <out base>/<benchmark>/plain-<label>/... (the plain arm eval_matrix --plain runs right before the drafter's).
  datakit, bigmix : <arm>/<set>/c<N>/results.json  results[0].output_tokens_per_s  (aggregate output tokens per second at client concurrency N), plus acceptance.csv for the acceptance length
  speedbench      : <arm>/speed_bench_<category>/profile_export_aiperf.json  output_token_throughput.avg (tokens/s), inter_token_latency.avg (ms); overall = total output tokens / total benchmark seconds
Speedup = drafter tokens/s divided by plain tokens/s at the same client concurrency (16 for the held-out sets, 32 for SPEED-Bench). It depends on the concurrency and on the node: only same-job pairs are comparable.
"""
import argparse, csv, glob, json, os, sys


def num(x):
    if isinstance(x, dict):
        x = x.get("avg", x.get("value"))
    return float(x) if x is not None else None


def heldout(arm_dir, bench):
    out = {}
    for rj in sorted(glob.glob(f"{arm_dir}/{bench}/c*/results.json")):
        level = os.path.basename(os.path.dirname(rj))
        r = json.load(open(rj))["results"][0]
        entry = {"tok_s": float(r["output_tokens_per_s"]), "wall_s": float(r["wall_s"]), "tokens": float(r["completion_tokens"])}
        ac = os.path.join(os.path.dirname(rj), "acceptance.csv")
        if os.path.exists(ac):
            rows = list(csv.DictReader(open(ac)))
            pick = [r for r in rows if r["subset"] == "all"] or rows[:1]   # a single-subset run has no pooled "all" row
            if pick:
                entry["al"] = float(pick[0]["acceptance_length"])
        out[level] = entry
    return out


def speedbench(arm_dir):
    out = {}; tot_tok = tot_s = 0.0
    for pj in sorted(glob.glob(f"{arm_dir}/speed_bench_*/profile_export_aiperf.json")):
        cat = os.path.basename(os.path.dirname(pj))[len("speed_bench_"):]
        d = json.load(open(pj))
        tok, dur = num(d.get("total_output_tokens")), num(d.get("benchmark_duration"))
        out[cat] = {"tok_s": num(d["output_token_throughput"]), "itl_ms": num(d.get("inter_token_latency"))}
        if tok and dur:
            tot_tok += tok; tot_s += dur
    if tot_s:
        out["ALL"] = {"tok_s": tot_tok / tot_s, "itl_ms": None}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("--csv", default="")
    a = ap.parse_args()
    rows = []
    for bdir in sorted(glob.glob(f"{a.base}/*/")):
        bench = os.path.basename(bdir.rstrip("/"))
        arms = sorted(d for d in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, d)) and not d.endswith(".hung_20261107"))
        for lab in arms:
            if lab.startswith("plain-"):
                continue
            plain = f"plain-{lab}"
            if plain not in arms:
                continue
            if bench == "speedbench":
                d, p = speedbench(f"{bdir}{lab}"), speedbench(f"{bdir}{plain}")
                for cat in d:
                    if cat in p and p[cat]["tok_s"]:
                        rows.append((bench, lab, cat, p[cat]["tok_s"], d[cat]["tok_s"], d[cat]["tok_s"] / p[cat]["tok_s"], None))
            else:
                if bench == "paper":   # several prompt sets in one folder: one row per set and level
                    sets = sorted(x for x in os.listdir(f"{bdir}{lab}") if glob.glob(f"{bdir}{lab}/{x}/c*/results.json"))
                else:
                    sets = [bench]
                for st in sets:
                    d, p = heldout(f"{bdir}{lab}", st), heldout(f"{bdir}{plain}", st)
                    for lvl in d:
                        if lvl in p:
                            rows.append((bench if st == bench else f"{bench}:{st}", lab, lvl, p[lvl]["tok_s"], d[lvl]["tok_s"], d[lvl]["tok_s"] / p[lvl]["tok_s"], d[lvl].get("al")))
    if not rows:
        print("no drafter/plain pairs found under", a.base, "(run eval_matrix with --plain)", file=sys.stderr); return 1
    print(f"{'benchmark':10} {'drafter':28} {'level/category':14} {'plain tok/s':>12} {'drafter tok/s':>14} {'speedup':>8} {'AL':>6}")
    for b, l, c, pt, dt, sp, al in rows:
        print(f"{b:10} {l[:28]:28} {c:14} {pt:12.0f} {dt:14.0f} {sp:7.2f}x {('-' if al is None else format(al, '.3f')):>6}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["benchmark", "drafter", "level_or_category", "plain_tok_s", "drafter_tok_s", "speedup", "acceptance_length"]); w.writerows(rows)
        print("wrote", a.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
