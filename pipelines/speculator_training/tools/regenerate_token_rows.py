"""Targetgen regeneration of tokenized dataset rows with a vLLM server, at the token level.

The Datakit SFT mix is stored as token ids plus a loss mask (assistant tokens = 1), not as chat messages, so the speculators `regenerate-responses`
tool (which needs messages) does not apply. This tool does the same job on the token ids, through the completions endpoint:

  for each sampled row, for each assistant turn in order:
      prompt = everything so far (original non-assistant tokens + the target's own earlier turns)
      the target generates the turn (stops at <|eot_id|> or the token cap)
      the generated tokens (mask 1) replace the original turn; the original tokens that follow it (tool output, next user turn, end of text) are kept (mask 0)
  a turn that hits the cap, or a row that reaches --max-total-len, ends the row there.

  regen   one shard of the work (run one copy per vLLM server): writes part-*.parquet files, resumable
  merge   all parts of all shards -> an Arrow dataset in the layout the speculators trainer reads (input_ids, loss_mask, seq_len), plus the draft vocabulary
          (token_freq.pt and the d2t/t2d maps) computed from the target's own regenerated tokens

Rows are sampled uniformly without replacement from the train split (the first --train-ratio of the rows, the trainer's own split rule).

  python regenerate_token_rows.py regen --dataset DIR --num 200000 --seed 0 --shard 0 --num-shards 2 --endpoint http://127.0.0.1:8000 \
      --model open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 --out-dir DIR/parts --concurrency 512 --max-new-tokens 32768
  python regenerate_token_rows.py merge --parts-dir DIR/parts --out DIR/data --compare-vocab-from /scratch/.../dataset_seq32768
"""
import argparse
import asyncio
import bisect
import glob
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

MERGE_MARKER = "regen_merge.json"
EOT = 128009  # <|eot_id|>, the end of every assistant turn in this dataset (98.7% of turns)


# ---------------------------------------------------------------------------------------------------------- dataset access
class RowStore:
    """Random access to the rows of the Arrow shards (memory-mapped, no thread pools)."""

    def __init__(self, path: str):
        self.shards = sorted(Path(path).glob("data-*.arrow"))
        self.tables = [ipc.open_stream(pa.memory_map(str(f))).read_all() for f in self.shards]
        self.offsets = np.cumsum([0] + [t.num_rows for t in self.tables])
        self.num_rows = int(self.offsets[-1])

    def get(self, idx: int) -> tuple[list[int], list[int]]:
        s = bisect.bisect_right(self.offsets, idx) - 1
        r = self.tables[s].slice(idx - int(self.offsets[s]), 1).to_pydict()
        return r["input_ids"][0], r["loss_mask"][0]


def assistant_spans(mask: list[int]) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


# ---------------------------------------------------------------------------------------------------------- regeneration
async def regenerate_row(session, args, ids, mask, row_idx, stats):
    """Returns (input_ids, spans, n_regenerated, status) for one row."""
    spans = assistant_spans(mask)
    if not spans:
        return None, None, 0, "no_assistant_turn"
    out = list(ids[: spans[0][0]])            # everything before the first assistant turn (ends with the assistant header)
    out_spans, n_regen, status = [], 0, "complete"
    for k, (a, b) in enumerate(spans):
        if len(out) >= args.max_total_len - 1:
            status = "total_len"
            break
        mt = min(args.max_new_tokens, args.max_total_len - len(out))
        body = {"model": args.model, "prompt": out, "max_tokens": mt, "temperature": args.temperature, "top_p": args.top_p, "stop_token_ids": [EOT],
                "return_token_ids": True, "skip_special_tokens": False, "seed": (args.seed * 1_000_003 + row_idx * 131 + k) & 0x7FFFFFFF}
        for attempt in range(args.retries + 1):
            try:
                async with session.post(f"{args.endpoint}/v1/completions", json=body) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                    c = (await resp.json())["choices"][0]
                break
            except Exception as e:  # noqa: BLE001
                if attempt == args.retries:
                    return None, None, n_regen, f"error: {type(e).__name__}: {str(e)[:150]}"
                await asyncio.sleep(2 ** attempt)
        gen = c["token_ids"]
        stats["gen_tokens"] += len(gen); stats["requests"] += 1
        out_spans.append((len(out), len(out) + len(gen)))
        out.extend(gen)
        n_regen += 1
        if c["finish_reason"] != "stop":          # hit the cap (or the context): the row ends inside this turn
            status = "capped"
            break
        end_of_turn = spans[k + 1][0] if k + 1 < len(spans) else len(ids)
        out.extend(ids[b:end_of_turn])            # the original tokens between this turn and the next (or the end of the row), mask 0
    return out[: args.max_total_len], [(s, min(e, args.max_total_len)) for s, e in out_spans if s < args.max_total_len], n_regen, status


async def run_regen(args):
    import aiohttp

    store = RowStore(args.dataset)
    n_train = int(store.num_rows * args.train_ratio)
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(n_train, size=args.num, replace=False)      # the sample is the same for every shard; each takes its share
    mine = [int(i) for i in chosen[args.shard :: args.num_shards]]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for f in glob.glob(str(out_dir / f"part-s{args.shard}-*.parquet")):
        done.update(pq.read_table(f, columns=["row"]).column("row").to_pylist())
    todo = [i for i in mine if i not in done]
    next_part = len(glob.glob(str(out_dir / f"part-s{args.shard}-*.parquet")))
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} rows assigned, {len(done)} already done, {len(todo)} to regenerate; dataset {store.num_rows} rows, train split {n_train}", flush=True)
    stats = {"gen_tokens": 0, "requests": 0, "rows": 0, "errors": 0, "capped": 0, "complete": 0, "total_len": 0, "skipped": 0}
    buf, t0, last_log, lock = [], time.time(), time.time(), asyncio.Lock()
    err_f = open(out_dir / f"errors-s{args.shard}.jsonl", "a")
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    conn = aiohttp.TCPConnector(limit=0)

    def flush(force=False):
        nonlocal buf, next_part
        if buf and (force or len(buf) >= args.part_rows):
            tbl = pa.table({"row": pa.array([r["row"] for r in buf], pa.int64()), "input_ids": pa.array([r["ids"] for r in buf], pa.list_(pa.int32())),
                            "spans": pa.array([r["spans"] for r in buf], pa.list_(pa.int32())), "n_regenerated": pa.array([r["n"] for r in buf], pa.int32()),
                            "status": pa.array([r["status"] for r in buf])})
            pq.write_table(tbl, out_dir / f"part-s{args.shard}-{next_part:05d}.parquet")
            next_part += 1; buf = []

    async def one(session, idx):
        nonlocal last_log
        async with sem:
            ids, mask = store.get(idx)
            new_ids, spans, n, status = await regenerate_row(session, args, ids, mask, idx, stats)
        async with lock:
            stats["rows"] += 1
            if new_ids is None:
                stats["errors" if status.startswith("error") else "skipped"] += 1
                err_f.write(json.dumps({"row": idx, "status": status}) + "\n"); err_f.flush()
            else:
                stats[status] += 1
                buf.append({"row": idx, "ids": new_ids, "spans": [x for s in spans for x in s], "n": n, "status": status})
                flush()
            if time.time() - last_log > 30:
                el = time.time() - t0; last_log = time.time()
                print(f"[shard {args.shard}] {stats['rows']}/{len(todo)} rows, {stats['rows']/el:.2f} rows/s, {stats['gen_tokens']/el:,.0f} gen tok/s, {stats['gen_tokens']/max(stats['rows'],1):,.0f} gen tok/row, "
                      f"{stats['requests']} requests, capped {stats['capped']}, errors {stats['errors']}, elapsed {el/60:.1f} min", flush=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        await asyncio.gather(*(one(session, i) for i in todo))
    flush(force=True); err_f.close()
    el = time.time() - t0
    print(f"[shard {args.shard}] done: {stats} in {el/60:.1f} min ({stats['rows']/max(el,1):.2f} rows/s, {stats['gen_tokens']/max(el,1):,.0f} gen tok/s)", flush=True)
    if todo and stats["errors"] / len(todo) > args.max_error_frac:
        raise SystemExit(f"too many failed rows: {stats['errors']}/{len(todo)}")


# ---------------------------------------------------------------------------------------------------------- merge
def run_merge(args):
    from datasets import Dataset

    if Path(args.out).exists() and not (Path(args.out) / MERGE_MARKER).exists():
        raise SystemExit(f"{args.out} exists and was not made by this merge (no {MERGE_MARKER}); refusing to touch it. Choose another --out.")
    files = sorted(glob.glob(str(Path(args.parts_dir) / "part-*.parquet")))
    if not files:
        raise SystemExit(f"no part files in {args.parts_dir}")

    def gen():
        for f in files:
            t = pq.read_table(f).to_pydict()
            for ids, spans, row in zip(t["input_ids"], t["spans"], t["row"]):
                if len(spans) == 0:                 # nothing was generated (the prompt filled the window): no supervised tokens, as prepare-data drops such rows
                    continue
                m = np.zeros(len(ids), dtype=np.int64)
                for s, e in zip(spans[0::2], spans[1::2]):
                    m[s:e] = 1
                yield {"input_ids": np.asarray(ids, dtype=np.int64), "loss_mask": m, "seq_len": len(ids), "source_row": row}

    ds = Dataset.from_generator(gen, cache_dir=str(Path(args.out).parent / ".gen_cache"))
    ds = ds.shuffle(seed=0)                         # rows of the two shards are interleaved by the trainer's val split rule (the last rows are held out)
    ds = ds.remove_columns(["source_row"]) if args.drop_source_row else ds
    ds.set_format("torch")
    out = Path(args.out)
    if out.exists():
        # never delete something this tool did not make (a mistyped --out once pointed at the source dataset)
        if not (out / MERGE_MARKER).exists():
            raise SystemExit(f"{out} exists and was not made by this merge (no {MERGE_MARKER}); refusing to delete it. Choose another --out.")
        shutil.rmtree(out)
    ds.save_to_disk(str(out), num_shards=max(1, len(ds) // 30000))
    (out / MERGE_MARKER).write_text(json.dumps({"rows": len(ds), "parts": len(files)}))
    write_vocab_files(files, out, args)
    shutil.rmtree(Path(args.out).parent / ".gen_cache", ignore_errors=True)
    seq = np.asarray(ds["seq_len"]) if "seq_len" in ds.column_names else None
    print(f"merged {len(ds)} rows from {len(files)} part files into {out}; tokens {int(seq.sum()):,}" if seq is not None else f"merged {len(ds)} rows into {out}", flush=True)


def write_vocab_files(files, out: Path, args):
    """Draft vocabulary from the target's OWN tokens: count the supervised (regenerated) tokens of every row, save token_freq.pt in the format the trainer reads
    (a dict token id -> count), and build the draft<->target maps with the trainer's own function, so the data directory is complete before training starts."""
    import torch
    from speculators.train.vocab_mapping import build_vocab_mappings_from_distribution

    V = args.target_vocab_size
    counts = np.zeros(V, dtype=np.int64)
    for f in files:
        t = pq.read_table(f, columns=["input_ids", "spans"]).to_pydict()
        pieces = []
        for ids, spans in zip(t["input_ids"], t["spans"]):
            if len(spans) == 0:
                continue
            a = np.asarray(ids, dtype=np.int64)
            pieces.extend(a[s:e] for s, e in zip(spans[0::2], spans[1::2]))
        if pieces:
            counts += np.bincount(np.concatenate(pieces), minlength=V)[:V]
    freq = {int(i): int(c) for i, c in enumerate(counts) if c > 0}
    torch.save(freq, out / "token_freq.pt")
    (out / "token_freq.json").write_text(json.dumps({str(k): v for k, v in freq.items()}))
    d2t, t2d = build_vocab_mappings_from_distribution(token_freq_dict=freq, draft_vocab_size=args.draft_vocab_size, target_vocab_size=V)
    np.save(out / f"d2t-{args.draft_vocab_size}.npy", d2t.cpu().numpy())
    np.save(out / f"t2d-{args.draft_vocab_size}.npy", t2d.cpu().numpy())
    total = counts.sum()
    covered = counts[t2d.cpu().numpy().astype(bool)].sum()
    print(f"draft vocabulary from {total:,} supervised tokens ({len(freq):,} distinct): the {args.draft_vocab_size:,} most frequent cover {covered / total:.2%}", flush=True)
    if args.compare_vocab_from:      # how much of these tokens the OTHER (e.g. the itw drafter's) vocabulary would cover
        other = np.load(Path(args.compare_vocab_from) / f"t2d-{args.draft_vocab_size}.npy").astype(bool)
        print(f"for comparison, the vocabulary in {args.compare_vocab_from} covers {counts[other].sum() / total:.2%} of them; the two vocabularies share {int((other & t2d.cpu().numpy().astype(bool)).sum()):,} tokens", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("regen")
    r.add_argument("--dataset", required=True); r.add_argument("--train-ratio", type=float, default=0.999)
    r.add_argument("--num", type=int, required=True, help="rows to sample (across all shards)"); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--shard", type=int, default=0); r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--endpoint", required=True); r.add_argument("--model", required=True); r.add_argument("--out-dir", required=True)
    r.add_argument("--concurrency", type=int, default=512); r.add_argument("--max-new-tokens", type=int, default=32768, help="per turn; the most liberal setting, in practice limited by the room left under --max-total-len")
    r.add_argument("--max-total-len", type=int, default=32768); r.add_argument("--temperature", type=float, default=1.0); r.add_argument("--top-p", type=float, default=1.0)
    r.add_argument("--retries", type=int, default=3); r.add_argument("--request-timeout", type=float, default=7200)
    r.add_argument("--part-rows", type=int, default=500); r.add_argument("--max-error-frac", type=float, default=0.02)
    m = sub.add_parser("merge")
    m.add_argument("--parts-dir", required=True); m.add_argument("--out", required=True)
    m.add_argument("--draft-vocab-size", type=int, default=32000); m.add_argument("--target-vocab-size", type=int, default=128256)
    m.add_argument("--compare-vocab-from", default=None, help="a dataset dir whose t2d map is compared with the new one (coverage only; nothing is copied)")
    m.add_argument("--drop-source-row", action="store_true")
    args = ap.parse_args()
    if args.cmd == "regen":
        asyncio.run(run_regen(args))
    else:
        run_merge(args)


if __name__ == "__main__":
    main()
