"""mlabonne/open-perfectblend -> the trainer's dataset directory, rendered, tokenized and masked exactly like the Datakit SFT mix.

Run in the Datakit env (it imports Marin for the chat template and the Datakit tokenizer; run `finalize_dataset.py` afterwards in the speculators env):

    $ROOT/venv/bin/python -I data/build_perfectblend.py --out <new dir> [--seq-length 32768] [--workers 32]

1. Render every conversation (human -> user, gpt -> assistant) with MARIN_CHAT_TEMPLATE, no BOS (the tokenizer step adds it) and no reasoning
   header, into normalized parquet files of {id, text}: the form the Datakit stores are in.
2. Check on a sample that the token-stream mask equals the template's own {% generation %} spans (the check `build_datakit_sft_mix.py stats --verify-masks` makes).
3. Tokenize, mask, clip and bucket with the Datakit `_write_job`, and write the shards with its `_write_shard`; then write the metadata files as `write` does.
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

DATAKIT_BUILD = (
    "/scratch/09749/atutej/marin_speculator/spec-experiments/pipelines/speculator_training/experiments/"
    "dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data/build_datakit_sft_mix.py"
)
SOURCE_GLOB = "/scratch/09749/atutej/marin_speculator/cache/hf/hub/datasets--mlabonne--open-perfectblend/snapshots/*/data/train-*.parquet"
ROLES = {"human": "user", "gpt": "assistant"}
SOURCE_NAME = "open-perfectblend"


def load_datakit():
    spec = importlib.util.spec_from_file_location("build_datakit_sft_mix", DATAKIT_BUILD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # worker processes are forked: they find the module here
    spec.loader.exec_module(module)
    return module


def conversation_messages(conversation, stats):
    """Chat messages for one row, or None (counted in stats) when it does not start with a user turn, alternate, and end with an assistant turn."""
    messages = []
    for turn in conversation:
        role = ROLES.get(turn["from"])
        if role is None:
            stats[f"dropped_role_{turn['from']}"] += 1
            return None
        message = {"role": role, "content": turn["value"]}
        if role == "assistant":
            message["tool_calls"] = []  # as Marin's _inference_messages yields assistant turns
        messages.append(message)
    if not messages or messages[0]["role"] != "user" or messages[-1]["role"] != "assistant" or any(a["role"] == b["role"] for a, b in zip(messages, messages[1:])):
        stats["dropped_shape"] += 1
        return None
    return messages


THINKING = {"none": {}, "think": {"enable_thinking": True}, "nothink": {"enable_thinking": False}}   # the template renders "Reasoning: /think" or "/nothink" as a system header; none: no header


def render_shard(job):
    from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
    from transformers.utils.chat_template_utils import render_jinja_template

    source, out_path, shard, thinking = job
    stats, ids, texts = collections.Counter(), [], []
    for i, conversation in enumerate(pq.read_table(source, columns=["conversations"])["conversations"].to_pylist()):
        stats["rows_in"] += 1
        messages = conversation_messages(conversation, stats)
        if messages is None:
            continue
        rendered, _ = render_jinja_template(conversations=[messages], chat_template=MARIN_CHAT_TEMPLATE, bos_token="", **THINKING[thinking])
        ids.append(hashlib.blake2b(f"{SOURCE_NAME}:{shard:02d}:{i:07d}".encode(), digest_size=16).hexdigest())  # hex: the bucketing reads the first 8 hex digits
        texts.append(rendered[0])
    pq.write_table(pa.table({"id": pa.array(ids, pa.string()), "text": pa.array(texts, pa.string())}), out_path)
    return out_path, len(ids), dict(stats)


def verify_masks(datakit, files, n, thinking):
    """The token-stream mask must decode to the template's own generation spans."""
    from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
    from transformers.utils.chat_template_utils import render_jinja_template

    tok = datakit._load()["tokenizer"]
    same = compared = 0
    stats = collections.Counter()
    for source in files[:: max(1, len(files) // 3)]:
        for conversation in pq.read_table(source, columns=["conversations"]).slice(0, n // 3 * 40)["conversations"].to_pylist()[:: 40]:
            messages = conversation_messages(conversation, stats)
            if messages is None:
                continue
            rendered, spans = render_jinja_template(conversations=[messages], chat_template=MARIN_CHAT_TEMPLATE, bos_token="", return_assistant_tokens_mask=True, **THINKING[thinking])
            ids = datakit.tokenize_texts([rendered[0]])[0]
            mask, _ = datakit.assistant_mask(ids)
            body = tok.decode([i for i, m in zip(ids, mask) if m])
            want = "".join(rendered[0][a:b] for a, b in spans[0])
            compared += 1
            same += body == want
    return same, compared


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="a NEW dataset directory (refused if it exists)")
    parser.add_argument("--seq-length", type=int, default=32768)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--shard-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thinking", choices=sorted(THINKING), default="none", help="system header: none (no header), think (Reasoning: /think, the model's serving default) or nothink")
    args = parser.parse_args()

    out = Path(args.out)
    work = Path(str(out) + ".work")
    if out.exists():
        raise SystemExit(f"refusing: {out} exists")
    datakit = load_datakit()
    datakit.verify_tokenizer()
    files = sorted(glob.glob(SOURCE_GLOB))
    assert len(files) == 6, f"expected 6 parquet shards of open-perfectblend, found {len(files)}: {files}"
    print(f"{len(files)} source shards; work dir {work}", flush=True)

    same, compared = verify_masks(datakit, files, 300, args.thinking)
    print(f"mask check: {same}/{compared} rows: decoded supervised tokens == template generation spans", flush=True)
    assert compared >= 100 and same >= 0.99 * compared, "token-stream mask differs from the template's generation spans"

    (work / "normalized").mkdir(parents=True, exist_ok=True)
    (work / "ids").mkdir(exist_ok=True)
    render_jobs = [(f, str(work / "normalized" / f"part-{k:02d}.parquet"), k, args.thinking) for k, f in enumerate(files)]
    render_stats, parts = collections.Counter(), []
    with ProcessPoolExecutor(len(render_jobs)) as pool:
        for path, n_rows, stats in pool.map(render_shard, render_jobs):
            parts.append(path)
            render_stats.update(stats)
            print(f"  rendered {path}: {n_rows:,} rows", flush=True)
    print("render stats:", dict(render_stats), flush=True)

    all_ids = pa.concat_arrays([pq.read_table(p, columns=["id"])["id"].combine_chunks() for p in parts])
    ids_file = work / "ids" / f"{SOURCE_NAME}.parquet"
    pq.write_table(pa.table({"id": all_ids}), ids_file)

    jobs = [{"work": str(work), "source": SOURCE_NAME, "path": p, "ids_file": str(ids_file), "seq_length": args.seq_length, "n_buckets": args.shards,
             "name": f"{SOURCE_NAME}--{Path(p).stem}"} for p in parts]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(datakit._write_job, jobs, chunksize=1), 1):
            results.append(r)
            print(f"  tokenized {i}/{len(jobs)} files", flush=True)
    out.mkdir(parents=True)
    shard_jobs = [{"work": str(work), "bucket": b, "seed": args.seed, "shard_path": str(out / f"data-{b:05d}-of-{args.shards:05d}.arrow")} for b in range(args.shards)]
    with ProcessPoolExecutor(args.shard_workers) as pool:
        shards = list(pool.map(datakit._write_shard, shard_jobs, chunksize=1))

    total, freq = collections.Counter(), collections.Counter()
    for r in results:
        total.update(r["stats"])
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
    (out / "manifest.json").write_text(json.dumps({"seq_length": args.seq_length, "seed": args.seed, "shards": args.shards, "totals": dict(total),
        "per_source": {SOURCE_NAME: dict(total)}, "render_stats": dict(render_stats), "source": "mlabonne/open-perfectblend", "thinking": args.thinking, "partial": False}, indent=1))
    print(f"wrote {total['rows_kept']:,} rows ({total['tokens_kept']/1e9:.3f}B tokens, {total['tokens_supervised']/1e9:.3f}B supervised) in {len(files_out)} shards to {out}; "
          f"clipped {total['rows_clipped']:,} rows, dropped {total['dropped_unsupervised']:,} unsupervised")
    print(f"next: speculators env, `python data/finalize_dataset.py {out}`; the work dir {work} can then be deleted")


if __name__ == "__main__":
    main()
