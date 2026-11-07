"""Pick a random subset of the regenerated parts of the 400k-row Dr. Doom Datakit targetgen run and copy it into this experiment's fresh PARTS_DIR.

    python3 data/select_parts.py [--parts-per-shard 210] [--seed 0]

SRC (runs/grug_dspark_drdoom_targetgen/regen_parts) is READ ONLY: 2 shards x 400 parts x 500 rows. The parts are chosen uniformly at random (a prefix would favour the rows that
finished first, which are shorter: the first 210 parts hold rows about 5% shorter than the last 190, in a 14-part-per-half sample). The same number of parts is taken from each shard,
so the subset keeps the two shards' halves of the sampled rows balanced. The merge step (regenerate_token_rows.py merge, run by the train job) then reads PARTS_DIR as it would a full run.
"""
import argparse, json, random, re, shutil
from pathlib import Path

ROOT = Path("/scratch/09749/atutej/marin_speculator/runs")
SRC = ROOT / "grug_dspark_drdoom_targetgen" / "regen_parts"
DST = ROOT / "grug_dspark_drdoom_targetgen_210k" / "regen_parts"

ap = argparse.ArgumentParser()
ap.add_argument("--parts-per-shard", type=int, default=210)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if DST.exists():
    raise SystemExit(f"refusing: {DST} exists")
chosen = []
for shard in (0, 1):
    files = sorted(SRC.glob(f"part-s{shard}-*.parquet"))
    assert len(files) == 400, (shard, len(files))
    chosen += sorted(random.Random(a.seed * 10 + shard).sample(files, a.parts_per_shard))
DST.mkdir(parents=True)
for f in chosen:
    shutil.copy2(f, DST / f.name)
(DST.parent / "subset_parts.json").write_text(json.dumps({"source": str(SRC), "seed": a.seed, "parts_per_shard": a.parts_per_shard, "parts": [f.name for f in chosen], "rows": 500 * len(chosen)}, indent=0))
print(f"copied {len(chosen)} parts ({500 * len(chosen):,} rows) to {DST}")
