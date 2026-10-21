"""The Datakit rows a run has trained on after STEPS steps: rebuild the trainer's packed batches (speculators' MultipackDistributedBatchSamplerV2 with the run's
seed, batch length and world size, over the training split, epoch 0) and take the first STEPS batches of every rank. Run in the `speculators` env on a compute node.

  python itw_seen_rows.py --dataset /scratch/.../dataset_seq32768 --steps 80813 --out runs/itw_seen_rows/rows_step80813.npy

Writes the sorted unique row indices (positions in the dataset, usable with build_val_prompts.py --split train --rows-file) and prints the token count, which
should match the run's own count (the itw run: 120,043 tokens per step, 9.70B tokens at step 80,813). This is a reconstruction of the sampler, not a log of the run.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
from datasets import Dataset
from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--steps", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--start-steps", type=int, default=0, help="first step to include: the output holds the rows of steps [start-steps, steps) (e.g. the rows a run sees AFTER an earlier checkpoint)")
ap.add_argument("--ranks", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--batch-max-length", type=int, default=32768); ap.add_argument("--train-ratio", type=float, default=0.999)
a = ap.parse_args()
out = Path(a.out)
if out.exists():
    sys.exit(f"refusing: {out} exists")
out.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time()
lens = np.concatenate([np.array(Dataset.from_file(str(f)).with_format(None)["seq_len"], dtype=np.int64) for f in sorted(Path(a.dataset).glob("data-*.arrow"))])
n_train = int(len(lens) * a.train_ratio); train = lens[:n_train]
print(f"rows {len(lens):,}, training split {n_train:,} ({time.time() - t0:.0f} s)", flush=True)
seen, tokens = [], 0
for r in range(a.ranks):
    s = MultipackDistributedBatchSamplerV2(batch_max_length=a.batch_max_length, lengths=train, num_replicas=a.ranks, rank=r, seed=a.seed)
    batches = s._generate_batches(0)
    assert len(batches) >= a.steps, f"rank {r} has only {len(batches)} batches"
    for b in batches[a.start_steps : a.steps]:
        seen.append(np.asarray(b, dtype=np.int64)); tokens += int(np.minimum(train[b], a.batch_max_length).sum())
    print(f"rank {r}: {len(batches):,} batches in the epoch ({time.time() - t0:.0f} s)", flush=True)
rows = np.concatenate(seen); uniq = np.unique(rows)
assert len(uniq) == len(rows), "a row appears in two batches of one epoch"
np.save(out, uniq)
print(f"steps [{a.start_steps}, {a.steps}): {len(uniq):,} rows, {tokens:,} tokens ({tokens / (a.steps - a.start_steps) / a.ranks:,.0f} per rank-batch, {tokens / (a.steps - a.start_steps):,.0f} per step), {len(uniq) / n_train:.1%} of the training rows; wrote {out}")
