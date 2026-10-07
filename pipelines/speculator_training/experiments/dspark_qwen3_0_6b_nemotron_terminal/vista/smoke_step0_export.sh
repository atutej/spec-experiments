#!/bin/bash
# Smoke test of step 0 of the DSpark nemotron-terminal pipeline: export 300 seeded rows and compare
# with the genai reference (docs/vista_setup.md, "Reference results"). Needs no GPU. The first run
# downloads the whole corpus (~13 GB) into $HF_HOME and reads it with pyarrow.
#   nohup bash pipelines/speculator_training/experiments/dspark_qwen3_0_6b_nemotron_terminal/vista/smoke_step0_export.sh > "$PROJECT_ROOT/logs/smoke/step0-export.log" 2>&1 &
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
source "$REPO_DIR/env.sh"
OUT=$PROJECT_ROOT/runs/smoke_step0/nemotron-terminal_300_seed0.jsonl
REF_SHA=32f16db6c70dbc1f7a7563942b8bacf52747f26d678f1f7fcfb65326fd71ae68
REF_HEAD='task_90527__CMJA82Z/episode-8 task_101944__NX3GDrK/episode-4 task_17047__7eZ59DU/episode-10'

set +u; conda activate speculators || exit 1; set -u
echo "node=$(hostname) kind=$NODE_KIND HF_HOME=$HF_HOME"
rm -f "$OUT"
time python "$REPO_DIR/pipelines/speculator_training/tools/export_registry_dataset.py" \
    --dataset nemotron-terminal --limit 300 --seed 0 --out "$OUT" || { echo "STEP0 FAILED: export"; exit 1; }

sha=$(sha256sum "$OUT" | cut -d' ' -f1)
# iterate the file object, never splitlines (U+2028 appears inside records)
head3=$(python -I - "$OUT" <<'PY'
import json, sys
rows = []
with open(sys.argv[1], encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
print(len(rows), file=sys.stderr)
print(" ".join(f'{r["trial_name"]}/{r["episode"]}' for r in rows[:3]))
PY
)
echo "rows/lines: $(wc -l < "$OUT")"
echo "sha256:     $sha"
echo "first 3:    $head3"
ok=1
[[ "$sha" == "$REF_SHA" ]] && echo "sha256 matches genai" || { echo "sha256 DIFFERS from genai reference ($REF_SHA)"; ok=0; }
[[ "$head3" == "$REF_HEAD" ]] && echo "first rows match genai" || { echo "first rows DIFFER from reference: $REF_HEAD"; ok=0; }
[[ $ok -eq 1 ]] && echo "STEP0 OK (matches genai)" || echo "STEP0 DONE, DIFFERS from genai reference (inspect before trusting)"
