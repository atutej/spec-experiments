#!/usr/bin/env python
"""Prompts of the finance_v1 task (Nemotron-SFT v3 finance: "You are given a financial text extracted from 10-K or 10-Q files...") from the HELD-OUT validation rows of the Datakit dataset.

  python build_finance_val_prompts.py --output-dir <NEW dir> [--max-prompts 256]

The trainer's validation split is the last (1 - 0.999) of the rows (the end of the last shard, as in build_val_prompts.py). The arrow rows carry no source label, so finance rows are recognised by
the task's opening instruction in the user turn. Each prompt = everything before the first supervised token, as token ids (first_turn.jsonl + manifest.json, the layout run_eval.py reads).
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True); ap.add_argument("--max-prompts", type=int, default=256)
ap.add_argument("--dataset", default="/scratch/09749/atutej/marin_datakit/dataset_seq32768"); ap.add_argument("--train-ratio", type=float, default=0.999); ap.add_argument("--max-prompt-tokens", type=int, default=31000)
ap.add_argument("--model", default="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom")
a = ap.parse_args(); out = Path(a.output_dir)
assert not (out / "first_turn.jsonl").exists(), f"{out} already holds prompts: old outputs are never overwritten"
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
total = json.loads((Path(a.dataset) / "manifest.json").read_text())["totals"]["rows_kept"]
split_idx = int(total * a.train_ratio); n_val = total - split_idx
last = Dataset.from_file(str(sorted(Path(a.dataset).glob("data-*.arrow"))[-1])); first_val = len(last) - n_val; assert first_val >= 0
MARK = "financial text extracted from 10-K or 10-Q"
rows, seen = [], 0
for i in range(first_val, len(last)):
    r = last[i]; m = np.asarray(r["loss_mask"]); nz = np.flatnonzero(m); seen += 1
    if len(nz) == 0: continue
    j = int(nz[0]); ids = list(r["input_ids"])[:j]
    if MARK not in tok.decode(ids[:200], skip_special_tokens=False) or j > a.max_prompt_tokens: continue
    e = j
    while e < len(m) and m[e] == 1: e += 1
    rows.append({"prompt": ids, "prompt_tokens": j, "val_row": split_idx + (i - first_val), "reference_reply_tokens": e - j, "turn": 0, "turns_in_row": 1})
print(f"{len(rows)} finance rows among {seen} validation rows ({n_val} expected)")
rows = rows[: a.max_prompts]
out.mkdir(parents=True, exist_ok=True)
with open(out / "first_turn.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
json.dump({"files": {"first_turn.jsonl": {"rows": len(rows), "sha256": hashlib.sha256((out / "first_turn.jsonl").read_bytes()).hexdigest()}},
           "source": f"validation rows [{split_idx}, {total}) of {a.dataset} whose user turn contains {MARK!r}", "max_prompt_tokens": a.max_prompt_tokens}, open(out / "manifest.json", "w"), indent=1)
pt = sorted(r["prompt_tokens"] for r in rows); rt = sorted(r["reference_reply_tokens"] for r in rows)
print(f"wrote {len(rows)} prompts to {out}; prompt tokens median {pt[len(pt)//2]} (min {pt[0]}, max {pt[-1]}); dataset reply tokens median {rt[len(rt)//2]} (max {rt[-1]})")
