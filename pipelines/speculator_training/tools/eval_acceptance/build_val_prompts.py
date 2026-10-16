"""Prompts from the trainer's held-out validation rows (the last (1 - train_data_ratio) of the dataset, by position), as token ids.

The trainer's validation split is rows [int(N * ratio), N) of the dataset (speculators/train/data.py). The dataset is shuffled into shards, so those rows are a
sample of the whole mixture, and the last shard holds them. For each chosen row the prompt is everything before the first supervised token (the system and
user turns up to the start of the first assistant reply), so the target generates the reply itself, as in run_eval.py. Written as first_turn.jsonl with the
prompt as a list of token ids (the completions API accepts them; nothing is retokenized) plus a manifest.json, the layout run_eval.py reads.

  python build_val_prompts.py --dataset /scratch/.../dataset_seq32768 --output-dir runs/val_prompts_128 --num 128
"""
import argparse, hashlib, json, random
from pathlib import Path
from datasets import Dataset

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--output-dir", required=True)
ap.add_argument("--num", type=int, default=128)
ap.add_argument("--train-ratio", type=float, default=0.999)
ap.add_argument("--total-rows", type=int, default=6471159, help="rows of the whole dataset (manifest.json totals.rows_kept)")
ap.add_argument("--max-prompt-tokens", type=int, default=8000)
ap.add_argument("--min-prompt-tokens", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

shards = sorted(Path(a.dataset).glob("data-*.arrow"))
last = Dataset.from_file(str(shards[-1]))
split_idx = int(a.total_rows * a.train_ratio)
n_val = a.total_rows - split_idx
assert len(last) >= n_val, f"the last shard has {len(last)} rows, fewer than the {n_val} validation rows"
first_val = len(last) - n_val                       # index of the first validation row inside the last shard
rng = random.Random(a.seed)
order = list(range(first_val, len(last))); rng.shuffle(order)
rows, skipped = [], 0
for i in order:
    r = last[i]
    mask = r["loss_mask"]
    ids = r["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    m = mask.tolist() if hasattr(mask, "tolist") else list(mask)
    if 1 not in m: skipped += 1; continue
    start = m.index(1)
    end = start + (m[start:].index(0) if 0 in m[start:] else len(m) - start)   # end of the first supervised span
    if not (a.min_prompt_tokens <= start <= a.max_prompt_tokens): skipped += 1; continue
    rows.append({"prompt": ids[:start], "prompt_tokens": start, "val_row": split_idx + (i - first_val), "reference_reply_tokens": end - start})
    if len(rows) == a.num: break
assert len(rows) == a.num, f"only {len(rows)} usable rows"
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / "first_turn.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
sha = hashlib.sha256((out / "first_turn.jsonl").read_bytes()).hexdigest()
json.dump({"files": {"first_turn.jsonl": {"rows": len(rows), "sha256": sha}}, "source": f"validation rows [{split_idx}, {a.total_rows}) of {a.dataset}",
           "seed": a.seed, "max_prompt_tokens": a.max_prompt_tokens, "skipped_while_sampling": skipped}, open(out / "manifest.json", "w"), indent=1)
pt = sorted(r["prompt_tokens"] for r in rows); rt = sorted(r["reference_reply_tokens"] for r in rows)
print(f"wrote {len(rows)} prompts to {out}; prompt tokens median {pt[len(pt)//2]} (min {pt[0]}, max {pt[-1]}); dataset reply length median {rt[len(rt)//2]} (max {rt[-1]}); skipped {skipped}")
