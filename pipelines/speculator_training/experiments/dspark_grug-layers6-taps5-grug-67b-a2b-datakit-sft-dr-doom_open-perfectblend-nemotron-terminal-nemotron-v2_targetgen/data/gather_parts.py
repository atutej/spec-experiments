"""Gather ALL regenerated BigMix parts into this experiment's fresh PARTS_DIR: the 180,000 rows regenerated earlier plus the 116,000 more.

    python3 data/gather_parts.py        # idempotent; run by vista/run_train.sbatch before the merge

Sources (READ ONLY): runs/grug_dspark_drdoom_big_pool_targetgen/regen_parts (the 180k run) and runs/grug_dspark_drdoom_big_pool_targetgen_extra116k/regen_parts (the extra run).
Both name their files part-s<shard>-<n>.parquet from 0, so the extra run's parts are copied with the number offset by 1000 (nothing is overwritten or skipped). The row sets are disjoint
(the extra rows were sampled from the train rows the 180k run did not regenerate). The merge step (regenerate_token_rows.py merge) then reads PARTS_DIR as it would a single run's parts.
"""
import json, re, shutil, sys
from pathlib import Path

R = Path("/scratch/09749/atutej/marin_speculator/runs")
OLD, NEW = R / "grug_dspark_drdoom_big_pool_targetgen" / "regen_parts", R / "grug_dspark_drdoom_big_pool_targetgen_extra116k" / "regen_parts"
DST = R / "grug_dspark_drdoom_grug6t5_bigmix_targetgen296k" / "regen_parts"
OFFSET = 1000
old = sorted(OLD.glob("part-s*-*.parquet")); new = sorted(NEW.glob("part-s*-*.parquet"))
assert len(old) == 360, f"expected 360 old parts, found {len(old)}"
assert len(new) >= 200, f"expected about 232 new parts (116,000 rows / 500), found {len(new)}: did the extra regeneration finish?"
marker = DST / "gather.json"
want = {"old_parts": len(old), "new_parts": len(new)}
if marker.exists() and json.loads(marker.read_text()) == want:
    print(f"gather: {DST} already holds these {len(old)} + {len(new)} parts; skipping"); sys.exit(0)
if DST.exists() and any(DST.iterdir()):
    sys.exit(f"refusing: {DST} exists and holds something other than a finished gather of these parts")
DST.mkdir(parents=True, exist_ok=True)
for f in old:
    shutil.copy2(f, DST / f.name)
for f in new:
    m = re.fullmatch(r"part-s(\d+)-(\d+)\.parquet", f.name); assert m, f.name
    shutil.copy2(f, DST / f"part-s{m.group(1)}-{int(m.group(2)) + OFFSET:05d}.parquet")
marker.write_text(json.dumps(want))
print(f"gathered {len(old)} old + {len(new)} new parts into {DST}")
