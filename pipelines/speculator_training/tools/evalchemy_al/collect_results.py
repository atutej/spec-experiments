#!/usr/bin/env python
"""Collect acceptance lengths from run_evalchemy_al.sbatch result dirs into a markdown table and a CSV. Reads only spec_metrics.json and status files.

  python collect_results.py <results_dir> --out-md RESULTS.md --out-csv results.csv

Result dirs are named <drafter label>_<task>[_<rerun tag>]; a run counts only if its status is exit=0 (failed runs are skipped), and a later rerun of the same
drafter/task/seed replaces nothing: if two successful runs exist for one cell the script stops.
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

NOTHINK = ["gsm8k-0shot", "humanevalplus", "mbppplus", "ifbench", "cruxeval", "triviaqa"]
THINK = ["math500", "aime24", "gpqa-diamond"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_dir")
    p.add_argument("--out-md", required=True)
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    cells = {}   # (drafter, task, seed) -> (AL, generation tokens)
    for status in sorted(Path(a.results_dir).glob("*/seed*/*/status")):
        if status.read_text().strip() != "exit=0":
            continue
        run = status.parent
        seed, task = run.parent.name[len("seed"):], run.name
        drafter = run.parent.parent.name.split("_", 1)[0]
        m = json.loads((run / "spec_metrics.json").read_text())
        key = (drafter, task, seed)
        if key in cells:
            raise SystemExit(f"two successful runs for {key}")
        cells[key] = (m["acceptance_length"], int(m["generation_tokens"]))
    drafters = sorted({k[0] for k in cells})
    assert len(drafters) == 2, drafters
    d1, d2 = drafters
    seeds = lambda d, t: sorted(s for (dd, tt, s) in cells if dd == d and tt == t)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["drafter", "task", "seed", "acceptance_length", "generation_tokens"])
        for (d, t, s), (al, n) in sorted(cells.items()):
            w.writerow([d, t, s, f"{al:.4f}", n])

    def stats(d, t):
        v = [cells[(d, t, s)][0] for s in seeds(d, t)]
        return (statistics.mean(v), max(v) - min(v), len(v)) if v else None

    lines = [f"| Task | Thinking | {d1} mean | {d2} mean | {d1} - {d2} | run spread {d1} / {d2} | runs {d1} / {d2} |", "|---|---|---|---|---|---|---|"]
    gaps = defaultdict(list)
    for mode, tasks in (("off", NOTHINK), ("on", THINK)):
        for t in tasks:
            x, y = stats(d1, t), stats(d2, t)
            if x is None or y is None:
                lines.append(f"| {t} | {mode} | {'-' if x is None else f'{x[0]:.3f}'} | {'-' if y is None else f'{y[0]:.3f}'} | - | - | - |")
                continue
            gaps[mode].append(x[0] - y[0])
            lines.append(f"| {t} | {mode} | {x[0]:.3f} | {y[0]:.3f} | {x[0]-y[0]:+.3f} | {x[1]:.3f} / {y[1]:.3f} | {x[2]} / {y[2]} |")
    lines += ["", "Mean gap over tasks with both drafters:"]
    for mode in ("off", "on"):
        if gaps[mode]:
            lines.append(f"- thinking {mode}: {statistics.mean(gaps[mode]):+.3f} over {len(gaps[mode])} tasks")
    Path(a.out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
