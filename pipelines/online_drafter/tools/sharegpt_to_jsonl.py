#!/usr/bin/env python
"""ShareGPT (a json array of {"conversations": [{"from", "value"}...]}) -> the one file the online-drafter runs use for BOTH the Aurora trainer (dataset.train_data_path) and
tools/rollout_client.py: jsonl, one {"conversations": [{"role", "content"}...]} per line, in dataset order. Keeps the first N conversations that start with a user turn and
contain one. No tokenizer and no chat template here: the client renders the prompts.
  sharegpt_to_jsonl.py --input ShareGPT_V4.3_unfiltered_cleaned_split.json --out-dir <NEW dir> --n 5000"""
import argparse, json, os

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--n", type=int, default=5000)
a = p.parse_args()
assert not os.path.exists(a.out_dir), f"{a.out_dir} exists (use a new directory)"
os.makedirs(a.out_dir)
ROLE = {"human": "user", "gpt": "assistant"}
kept = 0
with open(f"{a.out_dir}/dataset.jsonl", "w") as f:
    for item in json.load(open(a.input)):
        conv = item.get("conversations") or []
        if conv and "from" in conv[0]:
            conv = [{"role": ROLE.get(m["from"], m["from"]), "content": m["value"]} for m in conv]
        if not conv or conv[0]["role"] != "user" or not any(m["role"] == "user" for m in conv):
            continue
        f.write(json.dumps({"conversations": conv}) + "\n")
        kept += 1
        if kept >= a.n:
            break
print(f"{kept} conversations -> {a.out_dir}/dataset.jsonl")
