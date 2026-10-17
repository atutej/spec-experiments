"""How many more training tokens the `speculators regenerate-responses` output layout needs than one row per conversation.

The CLI writes one row per generation: the whole prefix (earlier turns, tool outputs, the target's earlier replies) plus that generation. For a conversation with
several regenerated turns the prefix is therefore repeated once per turn. tools/regenerate_token_rows.py writes one row per conversation.

  python boundary_layout_inflation.py --parts-dir <regen parts dir> [--max-len 32768]
Reads the part-*.parquet files (columns input_ids, spans = flat [start, end, start, end, ...] of the regenerated turns).
"""
import argparse
import glob

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--parts-dir", required=True)
ap.add_argument("--max-len", type=int, default=32768)
a = ap.parse_args()
compact = boundary = gens = rows = multi_b = multi_c = 0
for f in sorted(glob.glob(f"{a.parts_dir}/part-*.parquet")):
    t = pq.read_table(f, columns=["input_ids", "spans"]).to_pydict()
    for ids, sp in zip(t["input_ids"], t["spans"]):
        if len(sp) == 0:
            continue
        ends = sp[1::2]
        c = min(len(ids), a.max_len)                       # one row per conversation
        b = sum(min(e, a.max_len) for e in ends)           # one row per generation: prefix + that generation
        compact += c; boundary += b; gens += len(ends); rows += 1
        if len(ends) > 1:
            multi_b += b; multi_c += c
print(f"rows {rows}, generations {gens} ({gens / rows:.2f} per row)")
print(f"training tokens: one row per conversation {compact / 1e6:,.1f}M ({compact / rows:,.0f} per conversation); one row per generation {boundary / 1e6:,.1f}M ({boundary / rows:,.0f} per conversation); ratio x{boundary / compact:.2f}")
print(f"multi-turn rows only: x{multi_b / multi_c:.2f}")
