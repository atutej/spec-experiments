"""Stage 3: write the selected rows as a training dataset in the Datakit format. Run in the `speculators` env (needs datasets, pyarrow, torch).

  python write_subset.py --dataset DATAKIT_DIR --sel SEL_DIR --out OUT_DIR [--seed 0] [--num-shards 215]

Takes selection.npz + universe.json from stage 2, maps each selected row back to its row in the Datakit dataset (rows may repeat if the selection
has duplicates), shuffles them with the seed, and saves them with `Dataset.save_to_disk` (data-xxxxx-of-yyyyy.arrow, state.json, dataset_info.json,
exactly what `datasets.load_from_disk` in the trainer reads). Then it writes token_freq.json and manifest.json (the supervised-token counts of the
SELECTED rows, so the reduced draft vocabulary is built from the training data) and runs the experiment's finalize_dataset.py on the result.
OUT_DIR must be new, and is never inside the source dataset.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
import pyarrow as pa, pyarrow.ipc as ipc

FINALIZE = Path(__file__).resolve().parents[2] / "experiments" / "dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw" / "data" / "finalize_dataset.py"
VOCAB_CAP = 1 << 19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--sel", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--num-shards", type=int, default=215)
    a = ap.parse_args()
    src, sel, out = Path(a.dataset).resolve(), Path(a.sel), Path(a.out).resolve()
    if out == src or src in out.parents:
        sys.exit(f"refusing: --out {out} is inside the source dataset {src}")
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing: {out} exists and is not empty")
    print(f"dataset goes to {out}", flush=True)
    from datasets import load_from_disk

    s = np.load(sel / "selection.npz")
    ranges = json.loads((sel / "universe.json").read_text())
    starts = np.array([r["start"] for r in ranges])
    shard_files = sorted(src.glob("data-*.arrow"))
    shard_rows = [ipc.open_stream(pa.memory_map(str(f))).read_all().num_rows for f in shard_files]
    offsets = np.concatenate([[0], np.cumsum(shard_rows)])
    g = s["global_row"].astype(np.int64)
    ri = np.searchsorted(starts, g, side="right") - 1
    shard = np.array([ranges[i]["shard"] for i in ri]); row = g - starts[ri]
    for i in set(ri.tolist()):                                  # the universe's shard numbers are positions in the sorted shard list
        assert shard_files[ranges[i]["shard"]].name == ranges[i]["shard_file"], (ranges[i], shard_files[ranges[i]["shard"]].name)
    idx = offsets[shard] + row
    idx = idx[np.random.default_rng(a.seed).permutation(len(idx))]
    ds = load_from_disk(str(src))
    assert len(ds) == int(offsets[-1])
    sub = ds.select(idx.tolist())
    print(f"saving {len(sub):,} rows ({len(np.unique(idx)):,} unique)", flush=True)
    sub.save_to_disk(str(out), num_shards=min(a.num_shards, max(1, len(sub) // 1000)))

    freq = np.zeros(VOCAB_CAP, dtype=np.int64)
    n_tok = n_sup = 0
    for f in sorted(out.glob("data-*.arrow")):
        t = ipc.open_stream(pa.memory_map(str(f))).read_all()
        ids = t.column("input_ids").combine_chunks().values.to_numpy(zero_copy_only=False)
        mask = t.column("loss_mask").combine_chunks().values.to_numpy(zero_copy_only=False)
        assert ids.max() < VOCAB_CAP
        freq += np.bincount(ids[mask == 1], minlength=VOCAB_CAP)
        n_tok += len(ids); n_sup += int(mask.sum())
    (out / "token_freq.json").write_text(json.dumps({str(int(k)): int(freq[k]) for k in np.flatnonzero(freq)}))
    src_manifest = json.loads((src / "manifest.json").read_text())
    summary = json.loads((sel / "summary.json").read_text())
    (out / "manifest.json").write_text(json.dumps({
        "seq_length": src_manifest["seq_length"], "seed": a.seed, "shards": len(list(out.glob("data-*.arrow"))),
        "totals": {"rows_in": len(sub), "rows_kept": len(sub), "tokens_kept": n_tok, "tokens_supervised": n_sup},
        "built_from": {"dataset": str(src), "selection": str(sel.resolve()), "selection_summary": summary, "tool": "tools/select_similar_rows/write_subset.py"}}, indent=1))
    print(f"{n_tok:,} tokens, {n_sup:,} supervised; running finalize_dataset.py", flush=True)
    subprocess.run([sys.executable, str(FINALIZE), str(out)], check=True)


if __name__ == "__main__":
    main()
