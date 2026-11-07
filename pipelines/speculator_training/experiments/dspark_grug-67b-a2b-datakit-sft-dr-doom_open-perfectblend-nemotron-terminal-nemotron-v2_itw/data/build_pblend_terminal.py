"""mlabonne/open-perfectblend + nvidia/Nemotron-Terminal-Corpus -> one trainer dataset directory (the "big dataset" the PerfectBlend + Terminal experiments sample from).

Run in the Datakit env (it imports Marin and the Datakit tokenizer; run `finalize_dataset.py` afterwards in the speculators env):

    $ROOT/venv/bin/python -I data/build_pblend_terminal.py --out <new dir> --thinking nothink|think [--workers 64]

- PerfectBlend: rendered by `build_perfectblend.py` (Marin chat template, human -> user, gpt -> assistant) with the system header `Reasoning: /<thinking>`.
- Terminal corpus: the Datakit pipeline's own normalized rows (`normalized/sft/nemotron-terminal_*/outputs/main`, all 174,064 rows; the `dups` folder is not used), already rendered by Marin's template with the header `Reasoning: /nothink`
  on every row; with --thinking think that header is changed to `Reasoning: /think` (the only change to the text).
- Both sources are cut into small parts so that the tokenizing (the Datakit `_write_job`: [BOS] + text + " " + eos, assistant-only mask, clip at --seq-length) runs on many processes, then written as shards by the Datakit `_write_shard`.
Rows of the two sources are shuffled together in the shards (the bucket is the id hash), like the Datakit mix; the per-source counts are in manifest.json.
"""
import argparse
import collections
import glob
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
TERMINAL_MAIN = "/scratch/09749/atutej/marin_datakit/out/normalized/sft/nemotron-terminal_f87003a5/outputs/main"
PB_ROWS_PER_PART = 30_000
TERMINAL_ROWS_PER_PART = 1_800
OLD_HEADER = "Reasoning: /nothink"
NEW_HEADER = {"nothink": "Reasoning: /nothink", "think": "Reasoning: /think"}


def load_pb():
    spec = importlib.util.spec_from_file_location("build_perfectblend", HERE / "build_perfectblend.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def split_parts(src_path, out_dir, prefix, rows_per_part, header_swap):
    """One normalized parquet {id, text} -> several smaller parts; returns their paths and row counts."""
    table = pq.read_table(src_path, columns=["id", "text"])
    out = []
    for k, start in enumerate(range(0, table.num_rows, rows_per_part)):
        part = table.slice(start, rows_per_part)
        if header_swap:
            texts = part["text"].to_pylist()
            head = f"<|start_header_id|>system<|end_header_id|>{OLD_HEADER}<|eot_id|>"
            bad = [i for i, t in enumerate(texts) if not t.startswith(head)]
            if bad:
                raise SystemExit(f"{src_path}: {len(bad)} rows do not start with the expected system header, e.g. row {bad[0]}")
            new_head = f"<|start_header_id|>system<|end_header_id|>{header_swap}<|eot_id|>"
            part = pa.table({"id": part["id"], "text": pa.array([new_head + t[len(head):] for t in texts], pa.string())})
        path = Path(out_dir) / f"{prefix}-{k:03d}.parquet"
        pq.write_table(part, path)
        out.append((str(path), part.num_rows))
    return out


def _split_job(job):
    return split_parts(*job)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="a NEW dataset directory (refused if it exists)")
    parser.add_argument("--thinking", choices=sorted(NEW_HEADER), required=True)
    parser.add_argument("--seq-length", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--shard-workers", type=int, default=6)
    parser.add_argument("--shard-tokens", type=int, default=250_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    work = Path(str(out) + ".work")
    if out.exists():
        raise SystemExit(f"refusing: {out} exists")
    pb = load_pb()
    datakit = pb.load_datakit()
    datakit.verify_tokenizer()

    pb_files = sorted(glob.glob(pb.SOURCE_GLOB))
    terminal_files = sorted(glob.glob(TERMINAL_MAIN + "/part-*.parquet"))
    assert len(pb_files) == 6 and len(terminal_files) == 4, (len(pb_files), len(terminal_files))
    same, compared = pb.verify_masks(datakit, pb_files, 300, args.thinking)
    print(f"mask check (PerfectBlend): {same}/{compared} rows", flush=True)
    assert compared >= 100 and same >= 0.99 * compared, "token-stream mask differs from the template's generation spans"

    for d in ("render", "normalized", "ids"):
        (work / d).mkdir(parents=True, exist_ok=True)
    marker = work / "normalized" / "_complete.json"
    if marker.exists():
        parts = [tuple(x) for x in json.loads(marker.read_text())["parts"]]
        print(f"resuming: {len(parts)} normalized parts already written", flush=True)
    else:
        render_jobs = [(f, str(work / "render" / f"pb-{k:02d}.parquet"), k, args.thinking) for k, f in enumerate(pb_files)]
        render_stats = collections.Counter()
        with ProcessPoolExecutor(len(render_jobs)) as pool:
            for path, n_rows, stats in pool.map(pb.render_shard, render_jobs):
                render_stats.update(stats)
        print("PerfectBlend render stats:", dict(render_stats), flush=True)
        split_jobs = [(p, str(work / "normalized"), f"open-perfectblend-{k:02d}", PB_ROWS_PER_PART, None) for k, p in enumerate(sorted(glob.glob(str(work / "render" / "pb-*.parquet"))))]
        split_jobs += [(p, str(work / "normalized"), f"nemotron-terminal-{k:02d}", TERMINAL_ROWS_PER_PART, NEW_HEADER[args.thinking] if args.thinking != "nothink" else None)
                       for k, p in enumerate(terminal_files)]
        parts = []
        with ProcessPoolExecutor(len(split_jobs)) as pool:
            for res in pool.map(_split_job, split_jobs):
                parts.extend(res)
        marker.write_text(json.dumps({"parts": parts, "render_stats": dict(render_stats)}))
    n_pb = sum(1 for p, _ in parts if "/open-perfectblend-" in p)
    print(f"{len(parts)} normalized parts ({n_pb} PerfectBlend, {len(parts) - n_pb} Terminal), {sum(n for _, n in parts):,} rows", flush=True)

    all_ids = pa.concat_arrays([pq.read_table(p, columns=["id"])["id"].combine_chunks() for p, _ in parts])
    assert len(set(all_ids.to_pylist())) == len(all_ids), "duplicate ids across the two sources"
    ids_file = work / "ids" / "all.parquet"
    pq.write_table(pa.table({"id": all_ids}), ids_file)

    expected_tokens = 2_500_000_000
    n_buckets = max(1, -(-expected_tokens // args.shard_tokens))
    jobs = [{"work": str(work), "source": "open-perfectblend" if "/open-perfectblend-" in p else "nemotron-terminal", "path": p, "ids_file": str(ids_file),
             "seq_length": args.seq_length, "n_buckets": n_buckets, "name": Path(p).stem} for p, _ in parts]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(datakit._write_job, jobs, chunksize=1), 1):
            results.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  tokenized {i}/{len(jobs)} parts", flush=True)
    out.mkdir(parents=True)
    shard_jobs = [{"work": str(work), "bucket": b, "seed": args.seed, "shard_path": str(out / f"data-{b:05d}-of-{n_buckets:05d}.arrow")} for b in range(n_buckets)]
    with ProcessPoolExecutor(args.shard_workers) as pool:
        shards = list(pool.map(datakit._write_shard, shard_jobs, chunksize=1))

    total, per_source, freq = collections.Counter(), collections.defaultdict(collections.Counter), collections.Counter()
    for r in results:
        total.update(r["stats"])
        per_source[r["source"]].update(r["stats"])
        freq.update({int(k): v for k, v in r["token_freq"].items()})
    files_out = sorted(p.name for p in out.glob("data-*.arrow"))
    rows = sum(s.get("rows") or 0 for s in shards)
    if rows != total["rows_kept"]:
        raise SystemExit(f"row count mismatch: shards hold {rows}, tokenizing kept {total['rows_kept']}")
    (out / "token_freq.json").write_text(json.dumps({str(k): v for k, v in sorted(freq.items())}))
    size = sum((out / f).stat().st_size for f in files_out)
    (out / "state.json").write_text(json.dumps({"_data_files": [{"filename": f} for f in files_out], "_fingerprint": "0" * 16,
        "_format_columns": ["input_ids", "loss_mask", "seq_len"], "_format_kwargs": {}, "_format_type": "torch", "_output_all_columns": False, "_split": None}, indent=2))
    (out / "dataset_info.json").write_text(json.dumps({"builder_name": "json", "citation": "", "config_name": "default", "dataset_name": "json",
        "dataset_size": size, "description": "", "download_size": size, "features": datakit.HF_FEATURES, "homepage": "", "license": "", "size_in_bytes": size,
        "splits": {"train": {"name": "train", "num_bytes": size, "num_examples": total["rows_kept"], "dataset_name": "json"}},
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0}}, indent=2))
    (out / "manifest.json").write_text(json.dumps({"seq_length": args.seq_length, "seed": args.seed, "shards": n_buckets, "totals": dict(total),
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())}, "thinking": args.thinking,
        "sources": ["mlabonne/open-perfectblend", "nvidia/Nemotron-Terminal-Corpus (Datakit normalized rows, all 174,064)"], "partial": False}, indent=1))
    for name, c in sorted(per_source.items()):
        print(f"  {name}: {c['rows_kept']:,} rows, {c['tokens_kept']/1e9:.3f}B tokens, {c['tokens_supervised']/1e9:.3f}B supervised, {c['rows_clipped']:,} clipped")
    print(f"wrote {total['rows_kept']:,} rows ({total['tokens_kept']/1e9:.3f}B tokens, {total['tokens_supervised']/1e9:.3f}B supervised) in {len(files_out)} shards to {out}")
    print(f"next: speculators env, `python data/finalize_dataset.py {out}`; the work dir {work} can then be deleted")


if __name__ == "__main__":
    main()
