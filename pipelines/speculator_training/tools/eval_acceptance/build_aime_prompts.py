#!/usr/bin/env python
"""Prompt folder for AIME 2024 + AIME 2025 (60 problems), for run_eval.py (acceptance length on long competition-math reasoning).

Sources: Maxwell-Jia/AIME_2024 (train, 30, field Problem) and opencompass/AIME2025 (configs AIME2025-I and AIME2025-II, test, 15 each, field question).
Template as for MATH-500 in build_paper_prompts.py: "{problem}\\nPlease reason step by step, and put your final answer within \\boxed{}.", one user message, no system prompt,
the target's chat template with its default (`Reasoning: /think`), add_generation_prompt=True, stored as token ids.
Each problem is repeated --repeats times (default 4) so the acceptance length rests on more than 60 replies at temperature 1: repeat r is the full problem list shuffled with
random.Random(seed + r), concatenated, so concurrent requests are spread over different problems.

  python build_aime_prompts.py --out-root <NEW dir> [--repeats 4] [--model open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom]
Output: <out-root>/aime/first_turn.jsonl (prompt = token ids, prompt_tokens, dataset, index, year, repeat) and manifest.json.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

BOXED = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SOURCES = [("aime2024", ("Maxwell-Jia/AIME_2024",), {"split": "train"}, "Problem"),
           ("aime2025-I", ("opencompass/AIME2025", "AIME2025-I"), {"split": "test"}, "question"),
           ("aime2025-II", ("opencompass/AIME2025", "AIME2025-II"), {"split": "test"}, "question")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="a NEW directory")
    ap.add_argument("--model", default="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom")
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    root = Path(a.out_root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} exists and is not empty: old prompt sets are never overwritten")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    problems, src_info = [], {}
    for name, args, kw, field in SOURCES:
        data = load_dataset(*args, **kw)
        src_info[name] = {"dataset": args[0], "config": args[1] if len(args) > 1 else None, "split": kw["split"], "size": len(data), "field": field}
        for i in range(len(data)):
            text = data[i][field].strip() + BOXED
            ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            problems.append({"prompt": list(ids), "prompt_tokens": len(ids), "dataset": "aime", "index": len(problems), "year": name})
    rows = []
    for r in range(a.repeats):
        order = list(range(len(problems)))
        random.Random(a.seed + r).shuffle(order)
        rows += [{**problems[i], "repeat": r} for i in order]
    out = root / "aime"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "first_turn.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    mx = max(p["prompt_tokens"] for p in problems)
    (out / "manifest.json").write_text(json.dumps({
        "files": {"first_turn.jsonl": {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        "problems": len(problems), "repeats": a.repeats, "max_prompt_tokens": mx, "sources": src_info,
        "template": "{problem}" + BOXED, "seed": a.seed, "model_tokenizer": a.model}, indent=1))
    print(f"{len(problems)} problems x {a.repeats} repeats = {len(rows)} rows -> {path} (longest prompt {mx} tokens)")


if __name__ == "__main__":
    main()
