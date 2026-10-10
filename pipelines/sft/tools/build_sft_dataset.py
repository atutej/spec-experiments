#!/usr/bin/env python
"""Per-assistant-turn SFT dataset for LLaMA-Factory, tokenized with LLaMA-Factory's own encoder.

Input: a JSONL file of conversations (`conversations`: [{role, content}, ...], user first, strictly alternating user/assistant), e.g. the
100k sample of pipelines/speculator_training. Output: a `datasets.DatasetDict` ({train, validation}) with columns
input_ids / attention_mask / labels, saved with `save_to_disk`, which LLaMA-Factory loads with `tokenized_path: <out>` (it then skips
its own tokenization).

One example per assistant turn j of a conversation: the messages up to and including turn j, encoded by
`SupervisedDatasetProcessor._encode_data_example` with template `qwen3` and `mask_history=True`:
  - loss only on turn j (its thinking and response, through <|im_end|>\\n); all earlier turns, user and tool turns are context;
  - earlier assistant turns have their thinking stripped, no empty think tags added (upstream `discarding_history_cot`),
    which is what Qwen3's chat template shows the model at inference. Checked equal to HF `apply_chat_template` on 4,356 of 4,364 examples
    (the rest differ by one `\\n` in the last turn's own thinking), see pipelines/sft/README.md.
Examples longer than --max-len tokens are DROPPED, not truncated (a truncated turn would lose its end and its <|im_end|>).

The validation split is by conversation (a hash of trial_name|episode), so no conversation has turns on both sides.
Rows are written in a fixed order (source order, turns in order); the trainer shuffles. Needs the `sft` env.

  python build_sft_dataset.py --source sample.jsonl --out DIR [--workers 64] [--max-len 16384] [--limit-conversations N]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value

FEATURES = Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")),
                     "labels": Sequence(Value("int32"))})


def byte_ranges(path: str, n_ranges: int, limit_lines: int | None) -> list[list[int]]:
    """Split the file into n_ranges byte ranges that start and end at line starts (binary, so U+2028 inside records is harmless)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if limit_lines:
            for _ in range(limit_lines):
                if not f.readline():
                    break
            size = f.tell()
        starts = [0]
        for i in range(1, n_ranges):
            f.seek(size * i // n_ranges)
            f.readline()  # skip to the next line start
            pos = f.tell()
            if pos > starts[-1] and pos < size:
                starts.append(pos)
    return [[s, e] for s, e in zip(starts, starts[1:] + [size])]


def tokens(ds: Dataset) -> int:
    import pyarrow.compute as pc

    return int(pc.sum(pc.list_value_length(ds.data.column("input_ids"))).as_py() or 0) if len(ds) else 0


def is_val(key: str, fraction: float) -> bool:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64 < fraction


def generate(ranges, source, split, model, template, max_len, val_fraction, stats_dir):
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import DataArguments
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    # cutoff_len huge: this encoder must never truncate; over-long examples are dropped below.
    da = DataArguments(template=template, cutoff_len=1_000_000, mask_history=True)
    tpl = get_template_and_fix_tokenizer(tok, da)
    assert tpl.enable_thinking and not tpl.preserve_thinking, "history thinking must be stripped and the last turn's kept"
    proc = SupervisedDatasetProcessor(template=tpl, tokenizer=tok, processor=None, data_args=da)
    st = {"conversations": 0, "skipped_conversations": 0, "examples": 0, "dropped_too_long": 0, "no_loss": 0}
    with open(source, "rb") as f:
        for start, end in ranges:
            f.seek(start)
            while f.tell() < end:
                line = f.readline()
                if not line:
                    break
                rec = json.loads(line)
                key = f"{rec.get('trial_name')}|{rec.get('episode')}"
                if is_val(key, val_fraction) != (split == "validation"):
                    continue
                msgs = [{"role": m["role"], "content": m["content"]} for m in rec["conversations"]]
                ok = len(msgs) >= 2 and all(m["role"] == ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(msgs))
                if not ok:
                    st["skipped_conversations"] += 1
                    continue
                st["conversations"] += 1
                for j in range(1, len(msgs), 2):
                    sub = msgs[: j + 1]
                    ids, labels = proc._encode_data_example(sub[:-1], [sub[-1]], None, None, [], [], [])
                    if len(ids) > max_len:
                        st["dropped_too_long"] += 1
                        continue
                    if all(t == -100 for t in labels):
                        st["no_loss"] += 1
                        continue
                    st["examples"] += 1
                    yield {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(stats_dir) / f"{split}-{os.getpid()}-{ranges[0][0]}.json", "w") as sf:
        json.dump(st, sf)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--template", default="qwen3")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--val-fraction", type=float, default=0.002, help="fraction of conversations held out for validation")
    ap.add_argument("--limit-conversations", type=int, default=None, help="only the first N conversations (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    ranges = byte_ranges(args.source, args.workers * 8, args.limit_conversations)
    print(f"{args.source}: {len(ranges)} shards, {args.workers} workers", flush=True)
    stats_dir = tmp / "stats"
    parts = {}
    for split in ("train", "validation"):
        parts[split] = Dataset.from_generator(
            generate, features=FEATURES, num_proc=min(args.workers, len(ranges)), cache_dir=str(tmp / "cache"),
            gen_kwargs={"ranges": ranges, "source": args.source, "split": split, "model": args.model, "template": args.template,
                        "max_len": args.max_len, "val_fraction": args.val_fraction, "stats_dir": str(stats_dir)})
        print(f"{split}: {len(parts[split])} examples", flush=True)
    DatasetDict(parts).save_to_disk(str(tmp / "dataset"))

    totals = {"train": {}, "validation": {}}
    for p in stats_dir.glob("*.json"):
        split = p.name.split("-")[0]
        for k, v in json.load(open(p)).items():
            totals[split][k] = totals[split].get(k, 0) + v
    manifest = {"source": args.source, "model": args.model, "template": args.template, "max_len": args.max_len,
                "mask_history": True, "val_fraction": args.val_fraction, "limit_conversations": args.limit_conversations,
                "examples": {k: len(v) for k, v in parts.items()}, "stats": totals,
                "tokens": {k: tokens(v) for k, v in parts.items()}}
    json.dump(manifest, open(tmp / "dataset" / "manifest.json", "w"), indent=2)
    if out.exists():
        shutil.rmtree(out)
    (tmp / "dataset").replace(out)
    del parts  # release the memory-mapped cache files
    shutil.rmtree(tmp, ignore_errors=True)  # the generator cache and stats (a leftover only wastes space)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
