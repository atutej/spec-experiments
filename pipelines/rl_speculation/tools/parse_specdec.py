#!/usr/bin/env python
"""Pool vLLM `SpecDecoding metrics` log windows into one accept length.

vLLM logs a window every ~5 s per engine: "Mean acceptance length: L, ... Accepted: A tokens, Drafted: D tokens, Per-position acceptance rate: p0, p1, ...".
With k = number of speculative tokens, drafts = D / k and the pooled accept length is 1 + sum(A) / sum(drafts) (the official definition used by
speculator_training/tools/eval_acceptance). Windows are pooled by token counts, not averaged, so idle windows do not count.

  python parse_specdec.py <ray session logs dir or file> [more ...] [--k 8]
"""
import argparse
import re
import sys
from pathlib import Path

PAT = re.compile(r"SpecDecoding metrics: Mean acceptance length: ([\d.]+).*?Accepted: (\d+) tokens, Drafted: (\d+) tokens, Per-position acceptance rate: ([\d., ]+),")

ap = argparse.ArgumentParser()
ap.add_argument("paths", nargs="+")
ap.add_argument("--k", type=int, default=None, help="speculative tokens per draft (default: number of per-position rates)")
args = ap.parse_args()
files = []
for p in map(Path, args.paths):
    files += sorted(p.glob("worker-*.out")) + sorted(p.glob("*.log")) if p.is_dir() else [p]
acc = dra = 0
windows = 0
pos_num: list[float] = []
for f in files:
    for line in f.read_text(errors="ignore").splitlines():
        m = PAT.search(line)
        if not m:
            continue
        a, d = int(m.group(2)), int(m.group(3))
        rates = [float(x) for x in m.group(4).split(",")]
        k = args.k or len(rates)
        drafts = d / k
        if drafts <= 0:
            continue
        windows += 1
        acc += a
        dra += drafts
        pos_num = [x + r * drafts for x, r in zip(pos_num + [0.0] * (len(rates) - len(pos_num)), rates)]
if not dra:
    sys.exit("no SpecDecoding windows found")
print(f"windows={windows} drafts={dra:.0f} accepted={acc} pooled accept length={1 + acc / dra:.3f}")
print("per-position acceptance:", ", ".join(f"{x / dra:.3f}" for x in pos_num))
