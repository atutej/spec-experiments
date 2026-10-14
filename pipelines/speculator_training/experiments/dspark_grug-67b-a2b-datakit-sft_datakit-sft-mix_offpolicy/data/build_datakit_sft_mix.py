"""Download and mix the Datakit SFT data that trained Grug-67B-A2B-Datakit-SFT-262K-2026.09.21, reusing Marin's Datakit code.

Runs in the Marin datakit env, with Marin at 82620ad62bbe23b9b12a89d656c24e6f80aff91a (see plan_datakit_sft_mix.py):
    export MARIN_PREFIX=<big scratch dir>   # where Datakit writes raw/, processed-chat/, normalized-chat/, rendered/, normalized/
    export HF_TOKEN=$HUG_TOKEN HF_HOME=<scratch dir>
    PY=/scratch/09749/atutej/marin_datakit/venv/bin/python
    $PY -I data/build_datakit_sft_mix.py fetch  [--sources A B ... | --sources-file F] [--max-concurrent 8] [--dry-run]
    $PY -I data/build_datakit_sft_mix.py stats  --out stats/ [--sources ... | --sources-file F] [--workers 64] [--verify-masks 200]
    $PY -I data/build_datakit_sft_mix.py shards --n 2 --out-dir shards/   # split the 203 sources into balanced source lists for several nodes
    $PY -I data/build_datakit_sft_mix.py select --stats stats/ --out selection.json [--seed 0]
    $PY -I data/build_datakit_sft_mix.py select --stats stats/ --sources-file shards/shard-1-of-2.txt --rows 100000 --out selection_100k.json   # a sample
    $PY -I data/build_datakit_sft_mix.py write  --selection selection.json --out dataset/ --seq-length 32768 [--workers 32]
    <speculators env> python data/finalize_dataset.py dataset/        # token_freq.pt + a load check (needs torch)

How this mirrors the original construction (checked 2026-10-13 against Marin sources; nothing below is reimplemented where Marin's own code can be called):

  Original (experiments/datakit/sft_chat_tokenize.py + experiments/grug_sft/special_token_lr.py)        This script
  ---------------------------------------------------------------------------------------------------   ------------------------------------------
  sources = sorted(special_token_1000_mix.json["allocations_tokens"]) (203)                              the same file, the same 203 sources
  per source: all_sft_sources()[name].normalized  =  download -> chat transform -> chat-normalize          fetch runs exactly that StepSpec chain
                -> render_chat_step (Marin chat template) -> normalize_step (drop empty text, compact          (including the render and the text normalize:
                whitespace runs > 128 chars, id = xxh3(text), global exact-text dedup)                         these change the text and the row set)
  tokenize_attributes_step on the normalized text: levanter BatchTokenizer(enforce_bos/eos True), text        stats uses the same levanter classes on the
                + " " + eos, BOS prepended, long-string workaround ON (split at whitespace > 10k chars)        same normalized text, with the same switches
  stores: per source, pack=262144; the token count of a store is its "tokens"                             stats reports the per-source token total; select
  mixture: weights = 0.8 * allocation / budget per source (pooled when < 1.000001 samples per 64,000-        checks it against the allocation (an exact match for
       sample block), shuffle=True, one epoch, allocation <= store tokens else error                          sources that were taken whole, the original's error
                                                                                                              condition if it is below)

What is NOT reproduced exactly, by necessity: the original samples packed 262,144-token sequences with a seeded
shuffle inside each source; `select` takes a seeded random subset of conversations with the same per-source token total.
The per-source token shares are the same; which conversations are in the subset is not.

fetch   Runs Marin's StepRunner over the `normalized` step of every selected source. Finished steps are skipped on a rerun.
stats   For every normalized text row: tokenizes it with Marin's tokenizer path (the Grug tokenizer; verified first), counts
        the stored tokens, and derives the assistant-only loss mask from the token stream. One Parquet per source.
select  Per source, a seeded random subset of rows reaching the source's allocation; reports sources below / exactly at it.
write   Tokenizes the selected rows again (the same code as `stats`), clips them to --seq-length, drops rows left with no
        supervised token, shuffles them into shards (each shard holds every source, in proportion) and writes the trainer's
        dataset directory: Arrow shards with input_ids / loss_mask / seq_len in `datasets`' own schema (what
        `speculators prepare-data` produces), plus token_freq.json (finalize_dataset.py turns it into token_freq.pt).
        --seq-length is also the trainer's --total-seq-len. Refuses to run on a partial selection unless --allow-partial.
"""

import argparse
import collections
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MIX = "/scratch/09749/atutej/marin_datakit/marin/experiments/grug_sft/special_token_1000_mix.json"
TOKENIZER_REPO = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
EXPECTED_VOCAB_SIZE = 128256
# Same reserved ids experiments/datakit/sft_chat_tokenize.py verifies for the tokenizer it tokenizes with.
EXPECTED_TOKEN_IDS = {
    "<|begin_of_text|>": 128000, "<|end_of_text|>": 128001, "<|start_think|>": 128002, "<|end_think|>": 128003,
    "<tool_call>": 128005, "<|start_header_id|>": 128006, "<|end_header_id|>": 128007, "<|eot_id|>": 128009,
    "</tool_call>": 128011,
}
START_HEADER, END_HEADER, EOT, START_THINK, TOOL_CALL = 128006, 128007, 128009, 128002, 128005


def sources_arg(args) -> list[str] | None:
    """--sources-file (one source per line) or --sources; None means every source in the mixture spec."""
    if getattr(args, "sources_file", None):
        return [l.strip() for l in Path(args.sources_file).read_text().splitlines() if l.strip()]
    return args.sources


def selected_sources(only: list[str] | None) -> dict[str, int]:
    allocations = json.loads(Path(MIX).read_text())["allocations_tokens"]
    if only:
        unknown = sorted(set(only) - set(allocations))
        if unknown:
            raise SystemExit(f"not in the mixture spec: {unknown}")
        allocations = {name: allocations[name] for name in only}
    return allocations


def normalized_files(source) -> list[str]:
    """Shards of the text-normalized main output: what the original tokenize step read (`main_output_dir`)."""
    return sorted(str(p) for p in (Path(source.normalized.output_path) / "outputs" / "main").glob("*.parquet"))


# --------------------------------------------------------------------------- fetch
def cmd_fetch(args) -> None:
    from marin.datakit.sft_sources import all_sft_sources
    from marin.execution.step_runner import StepRunner

    registry = all_sft_sources()
    names = list(selected_sources(sources_arg(args)))
    steps = [registry[name].normalized for name in names]  # the runner schedules each step's dependencies first
    print(f"fetch: {len(names)} sources (download -> chat -> chat-normalize -> render -> normalize); "
          f"shared downloads are deduplicated by the runner", flush=True)
    StepRunner().run(steps, dry_run=args.dry_run, max_concurrent=args.max_concurrent)


# --------------------------------------------------------------------------- tokenizer + masks
_STATE = {}


def _tokenizer_dir() -> str:
    from huggingface_hub import hf_hub_download

    files = [hf_hub_download(TOKENIZER_REPO, f) for f in ("tokenizer.json", "tokenizer_config.json")]
    return os.path.dirname(files[0])


def _load():
    """Levanter's own tokenizer loader and BatchTokenizer, configured as marin.processing.tokenize._core configures them."""
    if not _STATE:
        from levanter.data.text.formats import TextLmDatasetFormat, preprocessor_for_format
        from levanter.tokenizers import load_tokenizer

        tokenizer = load_tokenizer(_tokenizer_dir())
        inner = preprocessor_for_format(TextLmDatasetFormat(), tokenizer)
        inner._long_string_workaround = True  # tokenize_batches_with_id turns this on unconditionally
        assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
        _STATE.update(tokenizer=tokenizer, inner=inner, assistant_ids=assistant_ids)
    return _STATE


def verify_tokenizer() -> None:
    """The vocabulary and reserved-id checks verify_tokenizer() in sft_chat_tokenize.py applies to the tokenizer used for the stores.

    The original also probed apply_chat_template() against MARIN_CHAT_TEMPLATE. The Hub repo ships its templates as separate
    files (not in tokenizer_config.json) and its default one is the *inference* variant, so the probe is replaced by comparing
    the repo's training_chat_template.jinja with Marin's template text. The template does not affect tokenizing text that is
    already rendered; the stores were built from rendered text.
    """
    from huggingface_hub import hf_hub_download
    from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE

    tok = _load()["tokenizer"]
    assert len(tok) == EXPECTED_VOCAB_SIZE, f"vocab {len(tok)}"
    for token, expected in EXPECTED_TOKEN_IDS.items():
        assert tok.encode(token, add_special_tokens=False) == [expected], token
    assert tok.bos_token_id == 128000 and tok.eos_token_id == 128001
    training_template = Path(hf_hub_download(TOKENIZER_REPO, "training_chat_template.jinja")).read_text()
    assert training_template == MARIN_CHAT_TEMPLATE, "the repo's training template differs from Marin's MARIN_CHAT_TEMPLATE"


def tokenize_texts(texts: list[str]) -> list[list[int]]:
    """Token ids as the original stores hold them: [BOS] + ids(text + " " + eos), long-string workaround on."""
    inner = _load()["inner"]
    return [row["input_ids"] for row in inner([{"text": t} for t in texts])]


def assistant_mask(ids: list[int]) -> tuple[list[int], int]:
    """Assistant-only loss mask from the token stream, and the number of assistant turns.

    The template renders an assistant turn as <|start_header_id|>assistant<|end_header_id|>\\n + {% generation %}content
    <|eot_id|>{% endgeneration %}. The span is every token after the one carrying the newline, up to and including
    <|eot_id|>; the document-final " <|end_of_text|>" that the stores append is outside it.
    """
    assistant = _load()["assistant_ids"]
    n, k = len(ids), len(assistant)
    mask, turns, i = [0] * n, 0, 0
    while i < n:
        if ids[i] == START_HEADER and ids[i + 1 : i + 1 + k] == assistant and i + 1 + k < n and ids[i + 1 + k] == END_HEADER:
            start = i + 1 + k + 2  # skip <|end_header_id|> and the token holding the newline
            end = start
            while end < n and ids[end] != EOT:
                end += 1
            for j in range(start, min(end + 1, n)):
                mask[j] = 1
            turns += 1
            i = end + 1
        else:
            i += 1
    return mask, turns


# --------------------------------------------------------------------------- stats
BATCH = 64


def _stats_for_file(job: tuple[str, str, int]) -> tuple[str, list[dict]]:
    source, path, verify_n = job
    out, checked, bad = [], 0, 0
    table = pq.read_table(path)
    ids_col, text_col = table["id"].to_pylist(), table["text"].to_pylist()
    src_col = table["source_id"].to_pylist() if "source_id" in table.column_names else [None] * len(ids_col)
    for start in range(0, len(ids_col), BATCH):
        texts = text_col[start : start + BATCH]
        for offset, ids in enumerate(tokenize_texts(texts)):
            if not ids:  # the original drops empty documents
                continue
            mask, turns = assistant_mask(ids)
            out.append({
                "id": ids_col[start + offset], "source_id": src_col[start + offset], "n_chars": len(texts[offset]),
                "n_stored_tokens": len(ids), "n_supervised": sum(mask), "n_assistant": turns,
                "n_headers": ids.count(START_HEADER), "n_think": ids.count(START_THINK), "n_tool_calls": ids.count(TOOL_CALL),
                "file": os.path.basename(path),
            })
    return source, out


def verify_masks(source, n: int) -> str:
    """Compare the token-stream mask with the template's own {% generation %} spans on rows the original text step left unchanged."""
    import glob
    from marin.datakit.chat_render import _inference_messages
    from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
    from openai_harmony import Message
    from transformers.utils.chat_template_utils import render_jinja_template

    chat_files = sorted(glob.glob(str(Path(source.chat_normalized.output_path) / "outputs" / "main" / "*.parquet")))
    chat = {r["id"]: r for r in pq.read_table(chat_files[0]).slice(0, n * 4).to_pylist()}
    rows = pq.read_table(normalized_files(source)[0]).slice(0, n * 4).to_pylist()
    tok, same, compared = _load()["tokenizer"], 0, 0
    for row in rows:
        record = chat.get(row.get("source_id"))
        if record is None:
            continue
        messages = [Message.from_dict(m) for m in record["messages"]]
        kwargs = json.loads(record["chat_template_kwargs"]) if record.get("chat_template_kwargs") else {}
        rendered, spans = render_jinja_template(
            conversations=[list(_inference_messages(messages))], chat_template=MARIN_CHAT_TEMPLATE, bos_token="",
            return_assistant_tokens_mask=True, **kwargs)
        if rendered[0] != row["text"]:  # text changed by the normalize step (whitespace compaction): not comparable
            continue
        ids = tokenize_texts([row["text"]])[0]
        mask, _ = assistant_mask(ids)
        body = tok.decode([i for i, m in zip(ids, mask) if m])
        want = "".join(row["text"][a:b] for a, b in spans[0])
        compared += 1
        same += body == want
        if compared >= n:
            break
    return f"{same}/{compared} rows: decoded supervised tokens == template generation spans"


def cmd_stats(args) -> None:
    from marin.datakit.sft_sources import all_sft_sources

    verify_tokenizer()
    print("tokenizer verified against the checks the original applies", flush=True)
    registry = all_sft_sources()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, missing = [], []
    for name in selected_sources(sources_arg(args)):
        files = normalized_files(registry[name])
        (jobs.extend((name, f, 0) for f in files) if files else missing.append(name))
    print(f"stats: {len(jobs)} files from {len({j[0] for j in jobs})} sources; not fetched yet: {len(missing)} {missing[:3]}", flush=True)
    rows: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (source, part) in enumerate(pool.map(_stats_for_file, jobs, chunksize=1), 1):
            rows.setdefault(source, []).extend(part)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} files", flush=True)
    for source, part in rows.items():
        pq.write_table(pa.Table.from_pylist(part), out_dir / (source.replace("/", "__") + ".parquet"))
    print(f"wrote {len(rows)} per-source stats files to {out_dir}")
    if args.verify_masks:
        for name in rows:
            print(f"  mask check {name}: {verify_masks(registry[name], args.verify_masks)}")


# --------------------------------------------------------------------------- select
def cmd_select(args) -> None:
    allocations = selected_sources(sources_arg(args))
    stats_dir = Path(args.stats)
    if args.only_if_complete:   # for sharded runs: every shard job calls this at its end; the one that finds all stats does the work
        missing = [n for n in allocations if not (stats_dir / (n.replace("/", "__") + ".parquet")).exists()]
        if missing:
            print(f"select: stats for {len(missing)} of {len(allocations)} sources are not there yet (other shards still running); nothing to do")
            return
    loaded = {}
    for name in allocations:
        path = stats_dir / (name.replace("/", "__") + ".parquet")
        if path.exists():
            loaded[name] = [r for r in pq.read_table(path).to_pylist() if r["n_stored_tokens"] > 0]
    scale = 1.0
    if args.rows:
        # A sample of about --rows rows that mirrors the mix: every source keeps the same fraction `scale` of its allocation, so the token
        # shares stay those of the original run. scale = rows / sum over sources of (allocation / mean tokens per row).
        denom = sum(allocations[n] / (sum(r["n_stored_tokens"] for r in rows) / len(rows)) for n, rows in loaded.items() if rows)
        scale = args.rows / denom
        print(f"sample of ~{args.rows:,} rows: each source keeps {scale:.4f} of its allocation ({scale * sum(allocations[n] for n in loaded) / 1e9:.2f}B tokens)")
    selection, report = {}, []
    for name, allocated in sorted(allocations.items()):
        if name not in loaded:
            report.append((name, allocated, None, "no stats file"))
            continue
        rows = loaded[name]
        store_tokens = sum(r["n_stored_tokens"] for r in rows)
        target = allocated * scale
        random.Random(f"{args.seed}:{name}").shuffle(rows)
        chosen, total = [], 0
        for r in rows:
            if total >= target:
                break
            chosen.append(r["id"])
            total += r["n_stored_tokens"]
        status = ("BELOW ALLOCATION (the original would have raised: allocates more tokens than the store has)"
                  if store_tokens < allocated else "exact match: whole store taken" if store_tokens == allocated else "ok")
        selection[name] = {"allocated_tokens": allocated, "target_tokens": int(target), "store_tokens": store_tokens, "selected_tokens": total, "ids": chosen}
        report.append((name, allocated, store_tokens, status))
    Path(args.out).write_text(json.dumps(selection))
    have = [r for r in report if r[2] is not None]
    print(f"selected {sum(len(v['ids']) for v in selection.values()):,} rows from {len(selection)} sources, "
          f"{sum(v['selected_tokens'] for v in selection.values()) / 1e9:.3f}B tokens (target {sum(v['target_tokens'] for v in selection.values()) / 1e9:.3f}B; the full allocation of these sources is {sum(allocations[n] for n in selection) / 1e9:.3f}B)")
    print(f"sources with stats: {len(have)}; exact whole-store matches: {sum(r[3].startswith('exact') for r in have)}; "
          f"below allocation: {sum(r[3].startswith('BELOW') for r in have)}; no stats yet: {sum(r[2] is None for r in report)}")
    for r in report:
        if r[2] is not None and not r[3] == "ok":
            print("  ", r)


# --------------------------------------------------------------------------- write
import numpy as np  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

HF_FEATURES = {
    "input_ids": {"feature": {"dtype": "int64", "_type": "Value"}, "_type": "List"},
    "loss_mask": {"feature": {"dtype": "int64", "_type": "Value"}, "_type": "List"},
    "seq_len": {"dtype": "int64", "_type": "Value"},
}
HF_METADATA = {b"huggingface": json.dumps({"info": {"features": HF_FEATURES}}).encode()}
FINAL_SCHEMA = pa.schema(
    [("input_ids", pa.list_(pa.int64())), ("loss_mask", pa.list_(pa.int64())), ("seq_len", pa.int64())], metadata=HF_METADATA
)
TEMP_SCHEMA = pa.schema([("input_ids", pa.list_(pa.int32())), ("loss_mask", pa.list_(pa.int8())), ("seq_len", pa.int32())])
FLUSH_TOKENS = 200_000_000          # per worker, before its buffered rows are written to the bucket files
WRITE_BATCH_ROWS = 1000             # rows per Arrow record batch, as `datasets` writes


def _bucket_of(row_id: str, n_buckets: int) -> int:
    return int(row_id[:8], 16) % n_buckets


def _flush(buffers: dict, job_dir: Path, counter: list) -> None:
    for bucket, rows in buffers.items():
        if not rows:
            continue
        ids = pa.array([r[0] for r in rows], type=pa.list_(pa.int32()))
        mask = pa.array([r[1] for r in rows], type=pa.list_(pa.int8()))
        n = pa.array([len(r[0]) for r in rows], type=pa.int32())
        with pa.OSFile(str(job_dir / f"b{bucket:05d}-{counter[0]:03d}.arrow"), "wb") as sink, pa.ipc.new_file(sink, TEMP_SCHEMA) as w:
            w.write_table(pa.table([ids, mask, n], schema=TEMP_SCHEMA))
    counter[0] += 1
    buffers.clear()


def _write_job(job: dict) -> dict:
    """Phase A for one normalized parquet file: tokenize the selected rows, clip, filter, bucket by id hash."""
    work, name = Path(job["work"]), job["name"]
    done = work / "done" / f"{name}.json"
    if done.exists():
        return json.loads(done.read_text())
    job_dir = work / "jobs" / (name + ".tmp")
    if job_dir.exists():
        for f in job_dir.iterdir():
            f.unlink()
    job_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(job["path"], columns=["id", "text"])
    value_set = pq.read_table(job["ids_file"])["id"]
    table = table.filter(pc.is_in(table["id"], value_set=value_set))
    L, nb = job["seq_length"], job["n_buckets"]
    stats = collections.Counter()
    freq = np.zeros(EXPECTED_VOCAB_SIZE, dtype=np.int64)
    buffers, buffered, counter = collections.defaultdict(list), 0, [0]
    row_ids, texts = table["id"].to_pylist(), table["text"].to_pylist()
    for start in range(0, len(texts), BATCH):
        for k, ids in enumerate(tokenize_texts(texts[start : start + BATCH])):
            stats["rows_in"] += 1
            if not ids:
                stats["dropped_empty"] += 1
                continue
            mask, _ = assistant_mask(ids)
            if len(ids) > L:
                stats["rows_clipped"] += 1
                stats["tokens_clipped"] += len(ids) - L
                ids, mask = ids[:L], mask[:L]
            if not any(mask):
                stats["dropped_unsupervised"] += 1
                continue
            ids_a, mask_a = np.asarray(ids, dtype=np.int32), np.asarray(mask, dtype=np.int8)
            freq += np.bincount(ids_a[mask_a == 1], minlength=EXPECTED_VOCAB_SIZE)
            buffers[_bucket_of(row_ids[start + k], nb)].append((ids_a, mask_a))
            stats["rows_kept"] += 1; stats["tokens_kept"] += len(ids_a); stats["tokens_supervised"] += int(mask_a.sum())
            buffered += len(ids_a)
            if buffered >= FLUSH_TOKENS:
                _flush(buffers, job_dir, counter); buffered = 0
    _flush(buffers, job_dir, counter)
    job_dir.rename(work / "jobs" / name)
    nz = np.nonzero(freq)[0]
    result = {"name": name, "source": job["source"], "stats": dict(stats), "token_freq": {int(t): int(freq[t]) for t in nz}}
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(result))
    return result


def _write_shard(job: dict) -> dict:
    """Phase B for one bucket: concatenate every job's rows for it, shuffle, and write the final Arrow shard."""
    out = Path(job["shard_path"])
    if out.exists():
        return {"bucket": job["bucket"], "rows": None, "skipped": True}
    tables = []
    for f in sorted(Path(job["work"]).glob(f"jobs/*/b{job['bucket']:05d}-*.arrow")):
        tables.append(pa.ipc.open_file(str(f)).read_all())
    if not tables:
        return {"bucket": job["bucket"], "rows": 0}
    t = pa.concat_tables(tables)
    t = t.take(pa.array(np.random.default_rng([job["seed"], job["bucket"]]).permutation(t.num_rows)))
    final = pa.table(
        [pc.cast(t["input_ids"], pa.list_(pa.int64())), pc.cast(t["loss_mask"], pa.list_(pa.int64())), pc.cast(t["seq_len"], pa.int64())],
        schema=FINAL_SCHEMA,
    )
    tmp = out.with_suffix(".arrow.tmp")
    with pa.OSFile(str(tmp), "wb") as sink, pa.ipc.new_stream(sink, FINAL_SCHEMA) as w:
        for b in final.to_batches(max_chunksize=WRITE_BATCH_ROWS):
            w.write_batch(b)
    tmp.rename(out)
    return {"bucket": job["bucket"], "rows": final.num_rows, "bytes": out.stat().st_size}


def _guard(selection: dict, allocations: dict[str, int]) -> list[str]:
    problems = []
    missing = sorted(set(allocations) - set(selection))
    if missing:
        problems.append(f"{len(missing)} of {len(allocations)} sources have no selection, e.g. {missing[:3]}")
    below = sorted(n for n, v in selection.items() if v["store_tokens"] < v["allocated_tokens"])
    if below:
        problems.append(f"{len(below)} sources are below their allocation, e.g. {below[:3]}")
    return problems


def cmd_write(args) -> None:
    from marin.datakit.sft_sources import all_sft_sources

    selection = json.loads(Path(args.selection).read_text())
    allocations = selected_sources(None)
    problems = _guard(selection, allocations)
    if problems and not args.allow_partial:
        raise SystemExit("refusing to write from an incomplete selection:\n  " + "\n  ".join(problems)
                         + "\nRun `fetch`, `stats` and `select` for all 203 sources first (data/submit_fetch.sh), or pass --allow-partial for a test or a sample.")
    if args.sources:
        selection = {k: v for k, v in selection.items() if k in set(args.sources)}
    verify_tokenizer()
    registry = all_sft_sources()
    out = Path(args.out)
    work = Path(str(out) + ".work")
    (work / "ids").mkdir(parents=True, exist_ok=True)
    expected_tokens = sum(v["selected_tokens"] for v in selection.values())
    n_buckets = args.shards or max(1, -(-expected_tokens // args.shard_tokens))
    print(f"write: {len(selection)} sources, ~{expected_tokens/1e9:.3f}B selected tokens, seq_length {args.seq_length}, {n_buckets} shards", flush=True)
    jobs = []
    for name, v in sorted(selection.items()):
        ids_file = work / "ids" / (name.replace("/", "__") + ".parquet")
        if not ids_file.exists():
            pq.write_table(pa.table({"id": pa.array(v["ids"], type=pa.string())}), str(ids_file) + ".tmp"); os.rename(str(ids_file) + ".tmp", ids_file)
        for path in normalized_files(registry[name]):
            jobs.append({"work": str(work), "source": name, "path": path, "ids_file": str(ids_file), "seq_length": args.seq_length,
                         "n_buckets": n_buckets, "name": f"{name.replace('/', '__')}--{Path(path).stem}"})
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(_write_job, jobs, chunksize=1), 1):
            results.append(r)
            if i % 25 == 0 or i == len(jobs):
                print(f"  tokenized {i}/{len(jobs)} files", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    shard_jobs = [{"work": str(work), "bucket": b, "seed": args.seed, "shard_path": str(out / f"data-{b:05d}-of-{n_buckets:05d}.arrow")} for b in range(n_buckets)]
    # Phase B holds a whole shard in memory (~26 bytes per token while it is permuted, cast to int64 and written), so it gets its own, smaller pool.
    with ProcessPoolExecutor(args.shard_workers) as pool:
        shards = list(pool.map(_write_shard, shard_jobs, chunksize=1))
    total = collections.Counter(); per_source = collections.defaultdict(collections.Counter); freq = collections.Counter()
    for r in results:
        total.update(r["stats"]); per_source[r["source"]].update(r["stats"]); freq.update({int(k): v for k, v in r["token_freq"].items()})
    files = sorted(p.name for p in out.glob("data-*.arrow"))
    if not any(s.get("skipped") for s in shards):   # a resumed run cannot recount shards it skipped; finalize_dataset.py checks the total
        rows = sum(s.get("rows") or 0 for s in shards)
        if rows != total["rows_kept"]:
            raise SystemExit(f"row count mismatch: shards hold {rows}, tokenizing kept {total['rows_kept']}")
    (out / "token_freq.json").write_text(json.dumps({str(k): v for k, v in sorted(freq.items())}))
    size = sum((out / f).stat().st_size for f in files)
    (out / "state.json").write_text(json.dumps({"_data_files": [{"filename": f} for f in files], "_fingerprint": "0" * 16,
        "_format_columns": ["input_ids", "loss_mask", "seq_len"], "_format_kwargs": {}, "_format_type": "torch", "_output_all_columns": False, "_split": None}, indent=2))
    (out / "dataset_info.json").write_text(json.dumps({"builder_name": "json", "citation": "", "config_name": "default", "dataset_name": "json",
        "dataset_size": size, "description": "", "download_size": size, "features": HF_FEATURES, "homepage": "", "license": "", "size_in_bytes": size,
        "splits": {"train": {"name": "train", "num_bytes": size, "num_examples": total["rows_kept"], "dataset_name": "json"}},
        "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0}}, indent=2))
    (out / "manifest.json").write_text(json.dumps({"seq_length": args.seq_length, "seed": args.seed, "shards": n_buckets, "totals": dict(total),
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())}, "partial": bool(problems)}, indent=1))
    print(f"wrote {total['rows_kept']:,} rows ({total['tokens_kept']/1e9:.3f}B tokens, {total['tokens_supervised']/1e9:.3f}B supervised) in {len(files)} shards to {out}; "
          f"clipped {total['rows_clipped']:,} rows (-{total['tokens_clipped']/1e6:.1f}M tokens), dropped {total['dropped_unsupervised']:,} unsupervised")
    print("next: in the speculators env, `python data/finalize_dataset.py " + str(out) + "` (token_freq.pt and a load check); the work dir " + str(work) + " can then be deleted")


# --------------------------------------------------------------------------- paths
def cmd_paths(args) -> None:
    """Cache each source's normalized-output directory (a hash of the step's parameters, so it is stable for a pinned Marin commit and MARIN_PREFIX).
    `shards --only-todo` reads this file instead of importing Marin: that import starts polars threads, which a login node's per-user limit refuses."""
    from marin.datakit.sft_sources import all_sft_sources

    registry = all_sft_sources()
    paths = {name: registry[name].normalized.output_path for name in selected_sources(None)}
    Path(args.out).write_text(json.dumps({"marin_prefix": os.environ.get("MARIN_PREFIX", ""), "paths": paths}, indent=1, sort_keys=True))
    print(f"wrote the normalized output paths of {len(paths)} sources to {args.out} (MARIN_PREFIX={os.environ.get('MARIN_PREFIX', '')})")


# --------------------------------------------------------------------------- shards
def cmd_shards(args) -> None:
    """Split the sources into N balanced lists. Sources that share a Hugging Face download stay together (so no node downloads another's
    repository); groups are packed greedily by download size and estimated tokens, whichever is currently the lighter shard."""
    plan = json.loads(Path(args.plan).read_text())
    stats_dir = Path(args.stats_dir)
    cost_scale = {}      # per source: 1.0 = needs fetch + stats; 0.25 = fetched already, only stats left
    if args.only_todo:   # a resume: split only the work that is left, using what is on disk
        cache = Path(args.paths_file)
        if cache.exists():   # no Marin import: works on a login node
            cached = json.loads(cache.read_text())["paths"]
            normalized_path = lambda n: cached[n]
        else:
            from marin.datakit.sft_sources import all_sft_sources
            registry = all_sft_sources()
            normalized_path = lambda n: registry[n].normalized.output_path
        for n in list(plan):
            status = Path(normalized_path(n)) / ".executor_status"
            fetched = status.exists() and status.read_text().strip() == "SUCCESS"
            has_stats = (stats_dir / (n.replace("/", "__") + ".parquet")).exists()
            if fetched and has_stats:
                del plan[n]
            elif fetched:
                cost_scale[n] = 0.25
                plan[n] = {**plan[n], "downloads": []}   # nothing left to download
        print(f"resume: {len(plan)} sources have work left ({sum(1 for n in plan if n not in cost_scale)} need fetch, {len(cost_scale)} only stats)")
    parent = {n: n for n in plan}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}
    for n, e in plan.items():
        for d in e["downloads"]:
            k = (d["hf_dataset_id"], d["revision"], tuple(d["hf_urls_glob"]))
            if k in owner:
                parent[find(n)] = find(owner[k])
            else:
                owner[k] = n
    groups = collections.defaultdict(list)
    for n in plan:
        groups[find(n)].append(n)
    total_gb = sum({(d["hf_dataset_id"], d["revision"], tuple(d["hf_urls_glob"])): d["bytes"] for e in plan.values() for d in e["downloads"]}.values()) / 1e9
    total_tok = sum(e["allocated_tokens"] for e in plan.values()) / 0.5070686 / 1e9      # allocation / the constant fraction seen in the mix
    rows = []
    for members in groups.values():
        gb = sum({(d["hf_dataset_id"], d["revision"], tuple(d["hf_urls_glob"])): d["bytes"] for m in members for d in plan[m]["downloads"]}.values()) / 1e9
        rows.append((gb / total_gb, sum(plan[m]["allocated_tokens"] for m in members) / 0.5070686 / 1e9 / total_tok, gb, sorted(members)))
    shards = [{"load": 0.0, "gb": 0.0, "tok": 0.0, "sources": []} for _ in range(args.n)]
    scale_of = lambda members: sum(cost_scale.get(m, 1.0) * plan[m]["allocated_tokens"] for m in members) / max(1, sum(plan[m]["allocated_tokens"] for m in members))
    for gfrac, tfrac, gb, members in sorted(rows, key=lambda r: -(r[0] + r[1] * scale_of(r[3]))):
        sh = min(shards, key=lambda x: x["load"])
        sh["load"] += gfrac + tfrac * scale_of(members); sh["gb"] += gb; sh["tok"] += tfrac * total_tok; sh["sources"] += members
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    seen = [x for sh in shards for x in sh["sources"]]
    assert sorted(seen) == sorted(plan) and len(seen) == len(set(seen)), "every source must be in exactly one shard"
    tag = f"-{args.tag}" if args.tag else ""
    for i, sh in enumerate(shards):
        (out / f"shard-{i}-of-{args.n}{tag}.txt").write_text("\n".join(sorted(sh["sources"])) + "\n")
        print(f"shard {i}/{args.n}: {len(sh['sources']):3d} sources, {sh['gb']:6.1f} GB left to download, ~{sh['tok']:5.1f}B tokens -> {out}/shard-{i}-of-{args.n}{tag}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--sources", nargs="*")
    p.add_argument("--sources-file")
    p.add_argument("--max-concurrent", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("stats")
    p.add_argument("--sources", nargs="*")
    p.add_argument("--sources-file")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--verify-masks", type=int, default=0, help="compare the token mask with the template spans on N rows per source")
    p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("select")
    p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only-if-complete", action="store_true", help="do nothing unless every source has stats (sharded runs)")
    p.add_argument("--sources-file", help="restrict to these sources (one per line)")
    p.add_argument("--rows", type=int, default=0, help="a sample of about this many rows, in proportion to the allocation (default: the full allocation)")
    p.set_defaults(fn=cmd_select)
    p = sub.add_parser("shards")
    p.add_argument("--plan", default=str(Path(__file__).with_name("plan.json")))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--only-todo", action="store_true", help="resume: split only the sources that still need fetch or stats")
    p.add_argument("--tag", default="", help="suffix for the list files, so a resume does not overwrite the first split")
    p.add_argument("--stats-dir", default="/scratch/09749/atutej/marin_datakit/stats")
    p.add_argument("--paths-file", default=str(Path(__file__).with_name("normalized_paths.json")), help="cached output paths (`paths` subcommand); avoids importing Marin")
    p.set_defaults(fn=cmd_shards)
    p = sub.add_parser("paths")
    p.add_argument("--out", default=str(Path(__file__).with_name("normalized_paths.json")))
    p.set_defaults(fn=cmd_paths)
    p = sub.add_parser("write")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq-length", type=int, required=True, help="clip rows to this many tokens (the trainer's --total-seq-len)")
    p.add_argument("--shards", type=int, default=0, help="number of Arrow shards (default: from --shard-tokens)")
    p.add_argument("--shard-tokens", type=int, default=250_000_000, help="target tokens per shard (about 7 GB of RAM per shard worker)")
    p.add_argument("--workers", type=int, default=32, help="tokenizing processes (about 3 GB each)")
    p.add_argument("--shard-workers", type=int, default=8, help="processes that write shards")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources", nargs="*")
    p.add_argument("--allow-partial", action="store_true", help="skip the completeness guard (tests only)")
    p.set_defaults(fn=cmd_write)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
