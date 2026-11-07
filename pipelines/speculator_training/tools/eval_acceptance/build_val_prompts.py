"""Prompts from the trainer's held-out validation rows (the last (1 - train_data_ratio) of the dataset, by position), as token ids.
With `--split train` they come instead from the TRAINING rows (positions [0, int(N * train_ratio))), sampled uniformly over rows, so a row that appears several
times in the dataset (the rlsimilar one repeats rows) is drawn as often as the drafter saw it; N is read from the dataset's manifest.json (totals.rows_kept).

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
ap.add_argument("--split", choices=["val", "train"], default="val", help="val: the held-out validation rows (the last shard); train: random rows of the training part of the dataset")
ap.add_argument("--rows-file", default="", help="with --split train: a .npy of row indices; sample only from these (e.g. the rows a run has seen, from itw_seen_rows.py)")
ap.add_argument("--turns", choices=["first", "random"], default="first", help="first: the first assistant reply of each row (first_turn.jsonl only); random: a uniformly random assistant turn of each row, context = everything before it (the dataset's earlier turns), written as first_turn.jsonl (turn 0) and later_turn.jsonl (turn > 0)")
a = ap.parse_args()

import bisect
shards = sorted(Path(a.dataset).glob("data-*.arrow"))
rng = random.Random(a.seed)
if a.split == "train":
    manifest_path = Path(a.dataset) / "manifest.json"
    if manifest_path.exists():
        a.total_rows = json.loads(manifest_path.read_text())["totals"]["rows_kept"]
    split_idx = int(a.total_rows * a.train_ratio)
    opened = {}
    sizes = [len(Dataset.from_file(str(f))) for f in shards]
    offsets = [0]
    for n in sizes: offsets.append(offsets[-1] + n)
    assert offsets[-1] == a.total_rows, f"the shards hold {offsets[-1]} rows, the manifest says {a.total_rows}"
    allowed = None
    if a.rows_file:
        import numpy as np
        allowed = np.load(a.rows_file); assert allowed.max() < split_idx, "--rows-file has rows outside the training split"
    order = [int(allowed[rng.randrange(len(allowed))]) if allowed is not None else rng.randrange(split_idx) for _ in range(a.num * 20)]   # uniform over the allowed rows, with replacement (rows that repeat in the data repeat here)
    def get_row(i):
        k = bisect.bisect_right(offsets, i) - 1
        if k not in opened: opened[k] = Dataset.from_file(str(shards[k]))
        return opened[k][i - offsets[k]]
    label = lambda i: i
else:
    last = Dataset.from_file(str(shards[-1]))
    split_idx = int(a.total_rows * a.train_ratio)
    n_val = a.total_rows - split_idx
    assert len(last) >= n_val, f"the last shard has {len(last)} rows, fewer than the {n_val} validation rows"
    first_val = len(last) - n_val                       # index of the first validation row inside the last shard
    order = list(range(first_val, len(last))); rng.shuffle(order)
    get_row = lambda i: last[i]
    label = lambda i: split_idx + (i - first_val)
rows, skipped = [], 0
for i in order:
    r = get_row(i)
    mask = r["loss_mask"]
    ids = r["input_ids"]
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    m = mask.tolist() if hasattr(mask, "tolist") else list(mask)
    if 1 not in m: skipped += 1; continue
    spans, j = [], 0
    while j < len(m):
        if m[j] == 1:
            e = j
            while e < len(m) and m[e] == 1: e += 1
            spans.append((j, e)); j = e
        else: j += 1
    k = rng.randrange(len(spans)) if a.turns == "random" else 0
    start, end = spans[k]
    if not (a.min_prompt_tokens <= start <= a.max_prompt_tokens): skipped += 1; continue
    rows.append({"prompt": ids[:start], "prompt_tokens": start, ("train_row" if a.split == "train" else "val_row"): label(i), "reference_reply_tokens": end - start, "turn": k, "turns_in_row": len(spans)})
    if len(rows) == a.num: break
assert len(rows) == a.num, f"only {len(rows)} usable rows"
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
files = {"first_turn": [r for r in rows if r["turn"] == 0]}
if a.turns == "random": files["later_turn"] = [r for r in rows if r["turn"] > 0]
manifest = {}
for name, rs in files.items():
    with open(out / f"{name}.jsonl", "w") as f:
        for r in rs: f.write(json.dumps(r) + "\n")
    manifest[f"{name}.jsonl"] = {"rows": len(rs), "sha256": hashlib.sha256((out / f"{name}.jsonl").read_bytes()).hexdigest()}
json.dump({"files": manifest, "source": ((f"{len(allowed):,} rows from {a.rows_file}" if allowed is not None else f"training rows [0, {split_idx})") + f" of {a.dataset}, sampled uniformly" if a.split == "train" else f"validation rows [{split_idx}, {a.total_rows}) of {a.dataset}"), "turns": a.turns,
           "seed": a.seed, "max_prompt_tokens": a.max_prompt_tokens, "skipped_while_sampling": skipped}, open(out / "manifest.json", "w"), indent=1)
pt = sorted(r["prompt_tokens"] for r in rows); rt = sorted(r["reference_reply_tokens"] for r in rows)
print({n: len(v) for n, v in files.items()}); print(f"wrote {len(rows)} prompts to {out}; prompt tokens median {pt[len(pt)//2]} (min {pt[0]}, max {pt[-1]}); dataset reply length median {rt[len(rt)//2]} (max {rt[-1]}); skipped {skipped}")
