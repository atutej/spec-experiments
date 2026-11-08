#!/usr/bin/env python
"""Prompt folders for the five datasets the DFlash / DSpark papers report on (GSM8K, MATH-500, HumanEval, MBPP, MT-Bench), for run_eval.py.

Templates and order follow z-lab's `dflash benchmark` (dflash/benchmark.py, DATASETS and _select_dataset), so numbers are comparable in kind:
  gsm8k     openai/gsm8k main, test (1,319):                  "{question}\\nPlease reason step by step, and put your final answer within \\boxed{}."
  math500   HuggingFaceH4/MATH-500, test (500):               "{problem}\\nPlease reason step by step, and put your final answer within \\boxed{}."
  humaneval openai/openai_humaneval, test (164):              "Write a solution to the following problem and make sure that it passes the tests:\\n```python\\n{prompt}\\n```"
  mbpp      google-research-datasets/mbpp sanitized, test (257): the dataset's `prompt` field
  mtbench   HuggingFaceH4/mt_bench_prompts, train (80):        the first turn only (the server backend of `dflash benchmark` also sends turn 1 only)
Each prompt is one user message, no system prompt, rendered with the target's chat template exactly as the RL runs send it (no chat_template_kwargs: the template
default, `Reasoning: /think`; add_generation_prompt=True) and stored as token ids. Order: random.Random(42).shuffle over the dataset indices, all rows, no repeats
(`dflash benchmark openai` repeats small sets up to 1,024 prompts; this does not).

  python build_paper_prompts.py --out-root <NEW dir> [--model open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom] [--datasets gsm8k math500 ...]
Output: <out-root>/<dataset>/first_turn.jsonl (prompt = token ids, prompt_tokens, dataset, index) and manifest.json (sources, templates, seed, counts, sha256).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

BOXED = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DATASETS = {
    "gsm8k": {"load": (("openai/gsm8k", "main"), {"split": "test"}), "format": lambda x: x["question"] + BOXED, "template": "{question}" + BOXED},
    "math500": {"load": (("HuggingFaceH4/MATH-500",), {"split": "test"}), "format": lambda x: x["problem"] + BOXED, "template": "{problem}" + BOXED},
    "humaneval": {"load": (("openai/openai_humaneval",), {"split": "test"}),
                  "format": lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n" + x["prompt"] + "\n```",
                  "template": "Write a solution to the following problem and make sure that it passes the tests:\\n```python\\n{prompt}\\n```"},
    "mbpp": {"load": (("google-research-datasets/mbpp", "sanitized"), {"split": "test"}), "format": lambda x: x["prompt"], "template": "{prompt}"},
    "mtbench": {"load": (("HuggingFaceH4/mt_bench_prompts",), {"split": "train"}), "format": lambda x: x["prompt"][0], "template": "{prompt[0]} (turn 1 only)"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="a NEW directory")
    ap.add_argument("--model", default="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom", help="target: its tokenizer and chat template render the prompts")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-prompt-tokens", type=int, default=28000, help="drop longer prompts (none of these datasets comes close)")
    a = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    root = Path(a.out_root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} exists and is not empty: old prompt sets are never overwritten")
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    for name in a.datasets:
        cfg = DATASETS[name]
        args, kw = cfg["load"]
        data = load_dataset(*args, **kw)
        order = list(range(len(data)))
        random.Random(a.seed).shuffle(order)
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        rows, skipped = [], 0
        for i in order:
            text = cfg["format"](data[i])
            ids = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            if len(ids) > a.max_prompt_tokens:
                skipped += 1
                continue
            rows.append({"prompt": list(ids), "prompt_tokens": len(ids), "dataset": name, "index": i})
        path = out / "first_turn.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        manifest = {"files": {"first_turn.jsonl": {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
                    "source": {"dataset": args[0], "config": args[1] if len(args) > 1 else None, "split": kw["split"], "size": len(data)},
                    "template": cfg["template"], "reference": "z-lab/dflash dflash/benchmark.py DATASETS (read 2026-11-07)",
                    "seed": a.seed, "shuffle": "random.Random(seed).shuffle(range(n)), all rows, no repeats",
                    "chat_template": "target default (Reasoning: /think), one user message, no system prompt; add_generation_prompt=True",
                    "model": a.model, "skipped_over_max_prompt_tokens": skipped,
                    "prompt_tokens": {"min": min(r["prompt_tokens"] for r in rows), "max": max(r["prompt_tokens"] for r in rows),
                                      "mean": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1)}}
        json.dump(manifest, open(out / "manifest.json", "w"), indent=1)
        print(f"{name}: {len(rows)} prompts (of {len(data)}), tokens min/mean/max {manifest['prompt_tokens']['min']}/{manifest['prompt_tokens']['mean']}/{manifest['prompt_tokens']['max']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
