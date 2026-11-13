#!/usr/bin/env python
"""Speculative-decoding metrics of a vLLM server over a time window, from two scrapes of its /metrics endpoint.

  python spec_metrics.py <metrics_before.prom> <metrics_after.prom> [--out summary.json]

Sums every label set (one per data-parallel engine) and subtracts the first scrape from the second. Prints and writes:
  acceptance_length          1 + accepted draft tokens / drafts (the bonus token counts; the maximum is 1 + the drafted tokens per step)
  acceptance_per_position    accepted / drafts at each draft position
  drafts, draft_tokens, accepted_tokens, generation_tokens, prompt_tokens
Without a drafter the speculative counters are absent and only the token counts are reported.
"""
import argparse
import json
import re
import sys
from collections import defaultdict

LINE = re.compile(r'^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$')
WANT = {
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted_tokens",
    "vllm:generation_tokens": "generation_tokens",
    "vllm:prompt_tokens": "prompt_tokens",
}
PER_POS = "vllm:spec_decode_num_accepted_tokens_per_pos"


def scrape(path):
    totals, per_pos = defaultdict(float), defaultdict(float)
    for raw in open(path):
        m = LINE.match(raw.strip())
        if not m or raw.startswith("#"):
            continue
        name = m["name"][:-6] if m["name"].endswith("_total") else m["name"]
        value = float(m["value"])
        if name in WANT:
            totals[WANT[name]] += value
        elif name == PER_POS:
            pos = re.search(r'position="(\d+)"', m["labels"] or "")
            if pos:
                per_pos[int(pos[1])] += value
    return totals, per_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--out")
    a = ap.parse_args()
    (t0, p0), (t1, p1) = scrape(a.before), scrape(a.after)
    d = {k: t1[k] - t0[k] for k in set(t0) | set(t1)}
    out = {k: int(round(v)) for k, v in d.items()}
    if d.get("drafts", 0) > 0:
        out["acceptance_length"] = round(1 + d["accepted_tokens"] / d["drafts"], 4)
        out["acceptance_per_position"] = {str(p): round((p1[p] - p0.get(p, 0.0)) / d["drafts"], 4) for p in sorted(p1)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
