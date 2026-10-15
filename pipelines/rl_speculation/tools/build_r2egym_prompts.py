#!/usr/bin/env python
"""R2E-Gym task instructions rendered with the target's chat template, in run_eval.py's prompt format (one JSONL per subset).

A proxy for the first turn of an RL rollout: the real agent (terminus-2) wraps the instruction in its own system prompt and keeps
calling the model for many turns, so contexts here are shorter than in RL.

  python build_r2egym_prompts.py --tasks-dir <dir of r2egym-v1-* tasks> --model open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 \
      --output-dir runs/r2egym_prompts_128 [--num 128] [--seed 0]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", required=True)
    ap.add_argument("--model", required=True, help="tokenizer / chat template source")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompt-tokens", type=int, default=24000, help="drop longer prompts (the server's max-model-len leaves room to generate)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tasks = sorted(p for p in Path(args.tasks_dir).iterdir() if (p / "instruction.md").is_file())
    random.Random(args.seed).shuffle(tasks)
    rows = []
    for task in tasks:
        text = (task / "instruction.md").read_text()
        prompt = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        n = len(tok(prompt, add_special_tokens=False)["input_ids"])
        if n <= args.max_prompt_tokens:
            rows.append({"prompt": prompt, "task": task.name, "prompt_tokens": n})
        if len(rows) == args.num:
            break
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "first_turn.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts = sorted(r["prompt_tokens"] for r in rows)
    # run_eval.py copies manifest["files"] into its results.json
    manifest = {"tasks_dir": args.tasks_dir, "model": args.model, "seed": args.seed,
                "files": {"first_turn": {"prompts": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                         "prompt_tokens_median": counts[len(counts) // 2] if counts else 0,
                                         "prompt_tokens_max": counts[-1] if counts else 0}}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(manifest)


if __name__ == "__main__":
    main()
