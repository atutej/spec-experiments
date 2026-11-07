"""The "big dataset" the PerfectBlend + Terminal + Nemotron experiments sample from: open-perfectblend + Nemotron-Terminal-Corpus + Nemotron-Post-Training-Dataset-v2 (the English splits chat, code, stem, math), as is.

Run in the Datakit env (it imports Marin and the Datakit tokenizer; run `finalize_dataset.py` afterwards in the speculators env):

    $ROOT/venv/bin/python -I data/build_big_pool.py --out <new dir> --mode itw|think [--workers 64]

--mode itw   existing text, each row with its own native reasoning mode: PerfectBlend and the Terminal rows `Reasoning: /nothink`; v2 rows `/think` when the row's `reasoning` flag is "on" (its `<think>...</think>` becomes
             Marin's `<|start_think|>...<|end_think|>` block) and `/nothink` when it is "off" (the empty `<think></think>` prefix is dropped).
--mode think every row `Reasoning: /think` (the model's serving default): the dataset the targetgen regeneration replays; the original replies are replaced there, so only the prompts matter.

PerfectBlend and the Terminal rows are built by `build_pblend_terminal.py` / `build_perfectblend.py` (same code). v2 rows are single user/assistant exchanges with an empty system message (dropped; the header comes from the reasoning mode).
Tokenizing, assistant-only masking, clipping at --seq-length and sharding are the Datakit `_write_job` / `_write_shard`.
"""
import argparse
import collections
import glob
import hashlib
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
V2_DIR = "/scratch/09749/atutej/marin_speculator/cache/hf/hub/datasets--nvidia--Nemotron-Post-Training-Dataset-v2/snapshots/*/data"
V2_SPLITS = ("chat", "code", "stem", "math")
V2_ROWS_PER_PART = 15_000


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def v2_messages(row, mode, stats):
    """(chat messages, enable_thinking) for one v2 row, or None (counted in stats) when it is not user + assistant."""
    roles = [m["role"] for m in row["messages"]]
    if roles != ["system", "user", "assistant"] or row["messages"][0]["content"].strip():
        stats["dropped_shape"] += 1
        return None
    user, content = row["messages"][1]["content"], row["messages"][2]["content"]
    reasoning_on = row["reasoning"] == "on"
    assistant = {"role": "assistant", "tool_calls": []}
    if content.startswith("<think>"):
        head, closed, tail = content[len("<think>"):].partition("</think>")
        reasoning, answer = head.strip(), (tail.strip() if closed else "")
    else:
        reasoning, answer = "", content.strip()
    if reasoning and reasoning_on:
        assistant["reasoning_content"] = reasoning
    elif reasoning:
        stats["off_row_with_reasoning_dropped_block"] += 1   # does not occur in the splits seen (off rows have an empty <think></think>)
    assistant["content"] = answer
    stats[f"reasoning_{row['reasoning']}"] += 1
    return [{"role": "user", "content": user}, assistant], (True if mode == "think" else reasoning_on)


def render_v2_file(job):
    from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
    from transformers.utils.chat_template_utils import render_jinja_template

    path, out_path, split, file_idx, mode = job
    stats, ids, texts = collections.Counter(), [], []
    for i, row in enumerate(pq.read_table(path, columns=["reasoning", "messages"]).to_pylist()):
        stats["rows_in"] += 1
        prepared = v2_messages(row, mode, stats)
        if prepared is None:
            continue
        messages, thinking = prepared
        rendered, _ = render_jinja_template(conversations=[messages], chat_template=MARIN_CHAT_TEMPLATE, bos_token="", enable_thinking=thinking)
        ids.append(hashlib.blake2b(f"nemotron-v2:{split}:{file_idx:02d}:{i:07d}".encode(), digest_size=16).hexdigest())
        texts.append(rendered[0])
    pq.write_table(pa.table({"id": pa.array(ids, pa.string()), "text": pa.array(texts, pa.string())}), out_path)
    return out_path, len(ids), dict(stats)


def verify_v2_masks(datakit, files, mode, n=300):
    """The token-stream mask must decode to the template's own generation spans (reasoning block included)."""
    from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
    from transformers.utils.chat_template_utils import render_jinja_template

    tok = datakit._load()["tokenizer"]
    same = compared = 0
    stats = collections.Counter()
    for path in files:
        for row in pq.read_table(path).slice(0, 4000).to_pylist()[:: max(1, 4000 // (n // len(files) + 1))]:
            prepared = v2_messages(row, mode, stats)
            if prepared is None:
                continue
            messages, thinking = prepared
            rendered, spans = render_jinja_template(conversations=[messages], chat_template=MARIN_CHAT_TEMPLATE, bos_token="", enable_thinking=thinking, return_assistant_tokens_mask=True)
            ids = datakit.tokenize_texts([rendered[0]])[0]
            mask, _ = datakit.assistant_mask(ids)
            compared += 1
            same += tok.decode([i for i, m in zip(ids, mask) if m]) == "".join(rendered[0][a:b] for a, b in spans[0])
    return same, compared


def _split_job(job):
    return PT.split_parts(*job)


def main():
    global PT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="a NEW dataset directory (refused if it exists)")
    parser.add_argument("--mode", choices=("itw", "think"), required=True)
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
    PT = load("build_pblend_terminal")
    pb = PT.load_pb()
    datakit = pb.load_datakit()
    datakit.verify_tokenizer()
    pb_think = "think" if args.mode == "think" else "nothink"    # PerfectBlend's header; the Terminal rows are /nothink natively and are switched to /think in think mode

    pb_files = sorted(glob.glob(pb.SOURCE_GLOB))
    terminal_files = sorted(glob.glob(PT.TERMINAL_MAIN + "/part-*.parquet"))
    v2_files = {s: sorted(glob.glob(f"{V2_DIR}/{s}-*.parquet")) for s in V2_SPLITS}
    assert len(pb_files) == 6 and len(terminal_files) == 4 and all(v2_files.values()), (len(pb_files), len(terminal_files), {k: len(v) for k, v in v2_files.items()})
    same, compared = pb.verify_masks(datakit, pb_files, 300, pb_think)
    print(f"mask check (PerfectBlend): {same}/{compared} rows", flush=True)
    assert compared >= 100 and same >= 0.99 * compared
    same, compared = verify_v2_masks(datakit, [v2_files["chat"][0], v2_files["chat"][-1], v2_files["math"][0]], args.mode)
    print(f"mask check (Nemotron v2): {same}/{compared} rows", flush=True)
    assert compared >= 100 and same >= 0.99 * compared

    for d in ("render", "normalized", "ids"):
        (work / d).mkdir(parents=True, exist_ok=True)
    marker = work / "normalized" / "_complete.json"
    if marker.exists():
        parts = [tuple(x) for x in json.loads(marker.read_text())["parts"]]
        render_stats = {}
        print(f"resuming: {len(parts)} normalized parts already written", flush=True)
    else:
        render_stats = collections.Counter()
        jobs_pb = [(f, str(work / "render" / f"pb-{k:02d}.parquet"), k, pb_think) for k, f in enumerate(pb_files)]
        with ProcessPoolExecutor(6) as pool:
            for path, n_rows, stats in pool.map(pb.render_shard, jobs_pb):
                render_stats.update({f"pb_{k}": v for k, v in stats.items()})
        v2_jobs = [(f, str(work / "render" / f"v2-{s}-{k:02d}.parquet"), s, k, args.mode) for s, fs in v2_files.items() for k, f in enumerate(fs)]
        with ProcessPoolExecutor(len(v2_jobs)) as pool:
            for (path, n_rows, stats), job in zip(pool.map(render_v2_file, v2_jobs), v2_jobs):
                render_stats.update({f"v2_{job[2]}_{k}": v for k, v in stats.items()})
        print("render stats:", dict(render_stats), flush=True)
        split_jobs = [(p, str(work / "normalized"), f"open-perfectblend-{k:02d}", PT.PB_ROWS_PER_PART, None) for k, p in enumerate(sorted(glob.glob(str(work / "render" / "pb-*.parquet"))))]
        split_jobs += [(p, str(work / "normalized"), f"nemotron-terminal-{k:02d}", PT.TERMINAL_ROWS_PER_PART, PT.NEW_HEADER["think"] if args.mode == "think" else None) for k, p in enumerate(terminal_files)]
        split_jobs += [(p, str(work / "normalized"), Path(p).stem.replace("v2-", "nemotron-v2-"), V2_ROWS_PER_PART, None) for p in sorted(glob.glob(str(work / "render" / "v2-*.parquet")))]
        parts = []
        with ProcessPoolExecutor(min(64, len(split_jobs))) as pool:
            for res in pool.map(_split_job, split_jobs):
                parts.extend(res)
        marker.write_text(json.dumps({"parts": parts, "render_stats": dict(render_stats)}))

    def source_of(path):
        name = Path(path).name
        return "open-perfectblend" if name.startswith("open-perfectblend-") else "nemotron-terminal" if name.startswith("nemotron-terminal-") else "nemotron-v2-" + name.split("-")[2]
    counts = collections.Counter()
    for p, n in parts:
        counts[source_of(p)] += n
    print(f"{len(parts)} normalized parts, {sum(counts.values()):,} rows: " + ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())), flush=True)

    all_ids = pa.concat_arrays([pq.read_table(p, columns=["id"])["id"].combine_chunks() for p, _ in parts])
    assert len(set(all_ids.to_pylist())) == len(all_ids), "duplicate ids across the sources"
    ids_file = work / "ids" / "all.parquet"
    pq.write_table(pa.table({"id": all_ids}), ids_file)

    n_buckets = max(1, -(-4_800_000_000 // args.shard_tokens))
    jobs = [{"work": str(work), "source": source_of(p), "path": p, "ids_file": str(ids_file), "seq_length": args.seq_length, "n_buckets": n_buckets, "name": Path(p).stem} for p, _ in parts]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(datakit._write_job, jobs, chunksize=1), 1):
            results.append(r)
            if i % 40 == 0 or i == len(jobs):
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
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())}, "mode": args.mode, "render_stats": dict(render_stats),
        "sources": ["mlabonne/open-perfectblend", "nvidia/Nemotron-Terminal-Corpus (Datakit normalized rows, all 174,064)", "nvidia/Nemotron-Post-Training-Dataset-v2 (chat, code, stem, math)"], "partial": False}, indent=1))
    for name, c in sorted(per_source.items()):
        print(f"  {name:22s} {c['rows_kept']:>10,} rows {c['tokens_kept']/1e9:7.3f}B tokens {c['tokens_supervised']/1e9:7.3f}B supervised {c['rows_clipped']:>6,} clipped")
    print(f"wrote {total['rows_kept']:,} rows ({total['tokens_kept']/1e9:.3f}B tokens, {total['tokens_supervised']/1e9:.3f}B supervised) in {len(files_out)} shards to {out}")
    print(f"next: speculators env, `python data/finalize_dataset.py {out}`; the work dir {work} can then be deleted")


PT = None

if __name__ == "__main__":
    main()
