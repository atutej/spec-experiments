#!/usr/bin/env python
"""Turn the speculator pipeline's prepared data (`speculators prepare-data` output) into a dataset LLaMA-Factory can train on.

Use it to skip build_sft_dataset.py (hours of tokenization) when a prepared dataset exists:
    python convert_prepared_data.py --prepared <run>/data --out DIR     then   PREBUILT_DATASET=DIR bash vista/run.sh train

Prepared rows are one assistant turn each: `input_ids` (the Qwen3 chat template applied by vLLM's renderer, earlier turns' thinking
stripped) and `loss_mask` (one contiguous span: that turn's thinking and response). LLaMA-Factory's columns are made as
labels = input_ids where loss_mask else -100, attention_mask = 1.
Rows with seq_len >= --max-len are DROPPED: prepare-data clipped those at its sequence length (cut mid-turn, no <|im_end|>).
Differences from build_sft_dataset.py: the encoder is vLLM's template rendering, not LLaMA-Factory's (they agreed on 4,356 of 4,364
checked examples, the rest one `\\n` in the last turn's own thinking), and the validation rows are random rows, NOT whole
conversations (the prepared rows carry no conversation id), so a validation turn can share its conversation with training turns.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import DatasetDict, Features, Sequence, Value, load_from_disk

FEATURES = Features({"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int8")),
                     "labels": Sequence(Value("int32"))})


def convert(batch):
    ids = [np.asarray(x, dtype=np.int32) for x in batch["input_ids"]]
    labels = [np.where(np.asarray(m, dtype=bool), i, -100).astype(np.int32) for i, m in zip(ids, batch["loss_mask"])]
    return {"input_ids": [i.tolist() for i in ids], "attention_mask": [[1] * len(i) for i in ids],
            "labels": [x.tolist() for x in labels]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--val-rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-rows", type=int, default=None, help="only the first N rows (smoke test)")
    args = ap.parse_args()

    ds = load_from_disk(args.prepared)
    # The prepared data is saved with format 'torch'; keeping it would store that format in the output, and datasets 4.x's torch formatter
    # fails in the training dataloader workers (it imports torchvision.io.VideoReader, which torchvision 0.28 removed).
    ds.reset_format()
    if args.limit_rows:
        ds = ds.select(range(args.limit_rows))
    n0 = len(ds)
    ds = ds.filter(lambda b: [s < args.max_len for s in b["seq_len"]], batched=True, num_proc=args.workers)
    print(f"{n0} rows, {n0 - len(ds)} dropped (seq_len >= {args.max_len}), {len(ds)} kept", flush=True)
    ds = ds.map(convert, batched=True, num_proc=args.workers, remove_columns=ds.column_names, features=FEATURES)
    split = ds.train_test_split(test_size=min(args.val_rows, len(ds) // 10), seed=args.seed, shuffle=True)
    dd = DatasetDict({"train": split["train"], "validation": split["test"]})
    for part in dd.values():
        assert part.format["type"] is None, "output must carry no format"
    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    dd.save_to_disk(str(tmp))
    manifest = {"prepared": args.prepared, "max_len": args.max_len, "rows_in": n0, "dropped_ge_max_len": n0 - len(ds),
                "train": len(dd["train"]), "validation": len(dd["validation"]), "val_split": "random rows, not by conversation"}
    json.dump(manifest, open(tmp / "manifest.json", "w"), indent=2)
    shutil.rmtree(out, ignore_errors=True)
    tmp.replace(out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
