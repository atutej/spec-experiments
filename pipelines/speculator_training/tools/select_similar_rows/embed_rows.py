"""Stage 1: embed "system + first user" text with google/embeddinggemma-2 (see common.py). Run in the `embed` env, one process per GPU.

  python embed_rows.py rl      --out EMB_DIR [--rl-dataset REPO] [--limit N]
  python embed_rows.py datakit --out EMB_DIR --dataset DATAKIT_DIR --worker 0 --num-workers 4 [--limit-shards N] [--limit-rows N]

EMB_DIR must be new (or an earlier run of this same command: finished shards are skipped, so a killed job is resumed by submitting it again).
It never writes into the dataset directory. Per Datakit shard it writes shard-00007.vec.npy (float16, rows x 768, L2-normalised),
shard-00007.seq.npy (int32, the rows' packed lengths) and, last, shard-00007.done.json; the RL side writes rl.vec.npy + rl.done.json.
Shards are dealt out to workers round-robin (shard i goes to worker i % num-workers).
"""
import argparse, glob, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common as C


def check_out(out: Path, source: str | None):
    out = out.resolve()
    if source and (out == Path(source).resolve() or Path(source).resolve() in out.parents):
        sys.exit(f"refusing: --out {out} is inside the source {source}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"embeddings go to {out}", flush=True)
    return out


def encode(model, texts, et, batch_size):
    clipped, ncut = C.clip_left(texts, et)
    vec = model.encode(clipped, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    return vec.astype(np.float16), ncut


def run_rl(a):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    out = check_out(Path(a.out), None)
    if (out / "rl.done.json").exists():
        print("rl.done.json exists, nothing to do"); return
    _, et = C.load_tokenizers()
    df = pd.read_parquet(hf_hub_download(a.rl_dataset, "data/train.parquet", repo_type="dataset"))
    if a.limit:
        df = df.iloc[: a.limit]
    texts = [C.join_text(*C.rl_first_turn(m)) for m in df.prompt]
    model = C.load_embedder("cuda")
    t0 = time.time()
    vec, ncut = encode(model, texts, et, a.batch_size)
    np.save(out / "rl.vec.npy", vec)
    (out / "rl.done.json").write_text(json.dumps({"rows": len(vec), "dim": vec.shape[1], "source": a.rl_dataset, "truncated_rows": ncut, "seconds": round(time.time() - t0, 1),
                                                   "model": C.MODEL, "model_revision": C.model_revision(), "prefix": C.PREFIX, "window": C.WINDOW}, indent=1))
    print(f"embedded {len(vec)} RL prompts in {time.time() - t0:.0f} s ({ncut} left-truncated)")


def run_datakit(a):
    import pyarrow as pa, pyarrow.ipc as ipc
    out = check_out(Path(a.out), a.dataset)
    gt, et = C.load_tokenizers()
    sp = C.special_ids(gt)
    shards = sorted(glob.glob(f"{a.dataset}/data-*.arrow"))
    if a.limit_shards:
        shards = shards[: a.limit_shards]
    model = C.load_embedder("cuda")
    for si in range(a.worker, len(shards), a.num_workers):
        name = f"shard-{si:05d}"
        if (out / f"{name}.done.json").exists():
            print(f"{name}: done, skipping", flush=True); continue
        t0 = time.time()
        table = ipc.open_stream(pa.memory_map(shards[si])).read_all()
        col = table.column("input_ids").combine_chunks()
        values, offsets = col.values.to_numpy(zero_copy_only=False), col.offsets.to_numpy()
        n = len(col) if not a.limit_rows else min(len(col), a.limit_rows)
        texts, nofirst = [], []
        for k in range(n):
            r = C.datakit_first_turn(values[offsets[k]:offsets[k + 1]], gt, sp)
            if r is None:
                nofirst.append(k); r = ("", "")
            texts.append(C.join_text(*r))
        vec, ncut = encode(model, texts, et, a.batch_size)
        seq = table.column("seq_len").to_numpy()[:n].astype(np.int32)
        np.save(out / f"{name}.vec.npy", vec); np.save(out / f"{name}.seq.npy", seq)
        (out / f"{name}.done.json").write_text(json.dumps({"shard_file": os.path.basename(shards[si]), "rows": n, "no_first_user_turn": nofirst, "truncated_rows": ncut,
                                                            "seconds": round(time.time() - t0, 1), "model": C.MODEL, "prefix": C.PREFIX, "window": C.WINDOW}))
        print(f"{name}: {n} rows in {time.time() - t0:.0f} s ({ncut} left-truncated, {len(nofirst)} without a first user turn)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="which", required=True)
    r = sub.add_parser("rl"); r.add_argument("--out", required=True); r.add_argument("--rl-dataset", default="open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30")
    r.add_argument("--limit", type=int, default=0); r.add_argument("--batch-size", type=int, default=32)
    d = sub.add_parser("datakit"); d.add_argument("--out", required=True); d.add_argument("--dataset", required=True)
    d.add_argument("--worker", type=int, default=0); d.add_argument("--num-workers", type=int, default=1)
    d.add_argument("--limit-shards", type=int, default=0); d.add_argument("--limit-rows", type=int, default=0); d.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    {"rl": run_rl, "datakit": run_datakit}[a.which](a)
