#!/usr/bin/env python
"""RLVR TRAIN prompts for the Aurora rollout client: {"token_ids": [...]} per line, from the eval builder's output (tools/eval_acceptance/build_nemotron_rlvr_prompts.py over ALL train rows, same seed and order, prompts <= 4,096 tokens),
WITHOUT the rows of the eval prompt sets (the 1,024-row file the 256-prompt RLVR eval takes its first rows from), so the eval prompts are never trained on.
  make_prompts.py --all <dir with first_turn.jsonl of all rows> --eval <runs/nemotron_rlvr_prompts_1024> --out <NEW file.jsonl>"""
import argparse, collections, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--all", required=True); ap.add_argument("--eval", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert not os.path.exists(a.out), f"{a.out} exists"
eval_rows = {json.loads(l)["row"] for l in open(f"{a.eval}/first_turn.jsonl")}
kept = excluded = 0; stats = collections.Counter(); lens = []
with open(a.all + "/first_turn.jsonl") as f, open(a.out, "w") as out:
    for line in f:
        r = json.loads(line)
        if r["row"] in eval_rows: excluded += 1; continue
        out.write(json.dumps({"token_ids": r["prompt"]}) + "\n"); kept += 1; lens.append(r["prompt_tokens"])
        stats["with_tools"] += bool(r.get("has_tools")); stats["with_history_turns"] += len(r.get("roles", [])) > 2 or r.get("tool_calls_in_history", 0) > 0
lens.sort()
print(f"kept {kept:,} prompts (excluded {excluded} eval rows of {len(eval_rows)}); prompt tokens median {lens[len(lens)//2]}, p90 {lens[int(len(lens)*.9)]}, max {lens[-1]}; with tools {stats['with_tools']:,}, with earlier turns in the prompt {stats['with_history_turns']:,}")
