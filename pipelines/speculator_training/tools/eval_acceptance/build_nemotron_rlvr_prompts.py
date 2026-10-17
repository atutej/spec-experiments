"""Prompts from the Snowball broad-domain RLVR training set (open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30, issue marin-community/marin#9735), rendered with the
target's own chat template exactly as the RL runs send them (no chat_template_kwargs -> the template default, `Reasoning: /think`), as token ids in run_eval.py's layout.

The sample is uniform over the train split's rows (so the environment mix follows the RL data) and shuffled, so the first N rows are themselves a uniform sample
(run_eval.py --max-prompts N). Each row records its environment (`agent`).
  python build_nemotron_rlvr_prompts.py --output-dir runs/nemotron_rlvr_prompts_1024 --num 1024
"""
import argparse, collections, hashlib, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", required=True); ap.add_argument("--num", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--repo", default="open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30"); ap.add_argument("--model", default="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21")
ap.add_argument("--max-prompt-tokens", type=int, default=50000, help="the RL context window is 102,400 tokens with up to 49,152 new tokens per turn")
a = ap.parse_args()
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

path = hf_hub_download(a.repo, "data/train.parquet", repo_type="dataset")
df = pd.read_parquet(path); tok = AutoTokenizer.from_pretrained(a.model)


def clean(o):   # parquet -> plain python: drop null fields, numpy arrays -> lists
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items() if v is not None}
    if isinstance(o, (list, tuple, np.ndarray)): return [clean(x) for x in o]
    return o


def render(r):
    """Messages (with tool_calls / tool_call_id as stored) and the row's tool schemas (request_json), as the RL chat-completions request carries them."""
    msgs = [clean(dict(m)) for m in r["prompt"]]
    rj = json.loads((r["extra_info"].get("nemotron_ultra") or {}).get("request_json") or "{}")
    tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}} for t in rj.get("tools") or []] or None
    ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=True)
    return (ids["input_ids"] if hasattr(ids, "keys") else ids), msgs, tools


rng = random.Random(a.seed); order = list(range(len(df))); rng.shuffle(order)
rows, skipped, failed = [], 0, collections.Counter()
for i in order:
    r = df.iloc[i]
    try: ids, msgs, tools = render(r)
    except Exception as e:  # noqa: BLE001
        failed[type(e).__name__ + ": " + str(e)[:80]] += 1; continue
    if len(ids) > a.max_prompt_tokens: skipped += 1; continue
    agent = (r["extra_info"].get("nemotron_ultra") or {}).get("agent")
    rows.append({"prompt": [int(t) for t in ids], "prompt_tokens": len(ids), "row": int(i), "agent": agent, "roles": [m["role"] for m in msgs], "has_tools": tools is not None, "tool_calls_in_history": sum("tool_calls" in m for m in msgs)})
    if len(rows) == a.num: break
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / "first_turn.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
json.dump({"files": {"first_turn.jsonl": {"rows": len(rows), "sha256": hashlib.sha256((out / "first_turn.jsonl").read_bytes()).hexdigest()}}, "source": f"{a.repo} data/train.parquet ({len(df)} rows)",
           "seed": a.seed, "chat_template": "model default (Reasoning: /think); add_generation_prompt=True", "skipped_over_max_prompt_tokens": skipped, "render_failures": dict(failed), "tools": "from extra_info.nemotron_ultra.request_json, passed as tools="}, open(out / "manifest.json", "w"), indent=1)
pt = sorted(r["prompt_tokens"] for r in rows)
print(f"wrote {len(rows)} prompts to {out}; prompt tokens median {pt[len(pt)//2]}, p90 {pt[int(len(pt)*.9)]}, max {pt[-1]}; skipped {skipped}")
print("environments:", dict(collections.Counter(r["agent"] for r in rows).most_common(12)))
print("with tools:", sum(r["has_tools"] for r in rows), "| with tool calls in history:", sum(r["tool_calls_in_history"] > 0 for r in rows), "| render failures:", dict(failed))
print("role patterns:", dict(collections.Counter(tuple(r["roles"]) for r in rows).most_common(4)))
