#!/usr/bin/env python
"""R2E-Gym TRAIN prompts for the Aurora rollout client: {"token_ids": [...]} per line. Same rendering as the eval prompts (tools/build_r2egym_prompts.py: the task's instruction.md as ONE user turn through the target's chat template,
generation prompt on, so a first-turn proxy of an RL rollout), over ALL tasks in the tasks dir EXCEPT the tasks of the eval prompt set (runs/r2egym_prompts_1024, field "task"), so the eval prompts are never trained on.
  make_prompts.py --tasks-dir <dir of r2egym-v1-* tasks> --eval <runs/r2egym_prompts_1024> --model <target repo for the tokenizer> --out <NEW dir> [--max-prompt-tokens 4096] [--seed 0]"""
import argparse, json, os, random
from pathlib import Path
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--tasks-dir", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-prompt-tokens", type=int, default=4096); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
out = Path(a.out)
assert not out.exists(), f"{out} exists"
eval_tasks = {json.loads(l)["task"] for l in open(f"{a.eval}/first_turn.jsonl")}
tok = AutoTokenizer.from_pretrained(a.model)
tasks = sorted(p for p in Path(a.tasks_dir).iterdir() if (p / "instruction.md").is_file())
random.Random(a.seed).shuffle(tasks)
rows, excluded, too_long, lens = [], 0, 0, []
for t in tasks:
    if t.name in eval_tasks: excluded += 1; continue
    prompt = tok.apply_chat_template([{"role": "user", "content": (t / "instruction.md").read_text()}], tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > a.max_prompt_tokens: too_long += 1; continue
    rows.append({"token_ids": ids, "task": t.name}); lens.append(len(ids))
out.mkdir(parents=True)
with open(out / "prompts_token_ids.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"token_ids": r["token_ids"]}) + "\n")
json.dump({"tasks_dir": a.tasks_dir, "tasks_total": len(tasks), "eval_tasks_excluded": excluded, "dropped_too_long": too_long, "kept": len(rows), "tasks": [r["task"] for r in rows], "max_prompt_tokens": a.max_prompt_tokens, "seed": a.seed}, open(out / "manifest.json", "w"))
lens.sort(); print(f"kept {len(rows):,} of {len(tasks):,} tasks (excluded {excluded} eval tasks, dropped {too_long} over {a.max_prompt_tokens} tokens); prompt tokens median {lens[len(lens)//2]}, p90 {lens[int(len(lens)*.9)]}, max {lens[-1]}")
print("first ids start:", rows[0]["token_ids"][:6], "| BOS count in first prompt:", rows[0]["token_ids"].count(tok.bos_token_id) if tok.bos_token_id is not None else "n/a")
