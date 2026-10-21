#!/bin/bash
# END-TO-END smoke test at a small size, on ONE 4-GPU gb node (run inside an allocation, or via sbatch): pool -> regen (Dr. Doom served here, REGEN_ROWS rows) -> merge (with the checkpoint's draft vocabulary)
# -> train (the itw folder's single-node run.sh, warm start, MAX_STEPS steps, no W&B, no upload). Everything goes to a FRESH work dir (default runs/grug_dspark_drdoom_targetgen_warm80k_smoke); refused if it exists or is the real run's.
# Checks at the end: the data directory holds the checkpoint's d2t/t2d; the first training steps start from a trained drafter (accept_len well above the 1.0 of random weights); a checkpoint was written.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; EXP="$(cd "$HERE/.." && pwd)"; REPO_DIR="$(cd "$HERE/../../../../.." && pwd)"
set +u; source "$REPO_DIR/env.sh"; set -u
SMOKE_DIR="${DRDOOM_TG_WORK_DIR:-$PROJECT_ROOT/runs/grug_dspark_drdoom_targetgen_warm80k_smoke}"
if [[ "$SMOKE_DIR" == "$PROJECT_ROOT/runs/grug_dspark_drdoom_targetgen_warm80k" ]]; then echo "refusing: that is the real run's directory" >&2; exit 1; fi
SMOKE_STAGES="${SMOKE_STAGES:-pool regen merge train}"     # SMOKE_STAGES=train re-runs only the training stage on the data a previous smoke run produced
if [[ "$SMOKE_STAGES" == pool* ]]; then
    if [[ -e "$SMOKE_DIR" ]]; then echo "refusing: $SMOKE_DIR exists (to redo only later stages: SMOKE_STAGES=train)" >&2; exit 1; fi
else
    [[ -f "$SMOKE_DIR/regen_parts/../data/regen_merge.json" ]] || { echo "SMOKE_STAGES='$SMOKE_STAGES' needs a smoke dir with merged data: $SMOKE_DIR" >&2; exit 1; }
fi
export DRDOOM_TG_WORK_DIR="$SMOKE_DIR" REGEN_ROWS="${REGEN_ROWS:-512}" DRDOOM_TG_TRAIN_DATA_RATIO="${DRDOOM_TG_TRAIN_DATA_RATIO:-0.95}" DRDOOM_TG_MAX_STEPS="${DRDOOM_TG_MAX_STEPS:-10}" DRDOOM_TG_CKPT_FREQ="${DRDOOM_TG_CKPT_FREQ:-0.5}" SMOKE=1
export EXP_DIR_OVERRIDE="$EXP"
echo "smoke work dir: $SMOKE_DIR | rows $REGEN_ROWS | train ratio $DRDOOM_TG_TRAIN_DATA_RATIO | max steps $DRDOOM_TG_MAX_STEPS | $(date)"
SECONDS=0
for stage in $SMOKE_STAGES; do
    if [[ "$stage" == train ]]; then
        echo "=== stage train (single node, warm start) | ${SECONDS}s"
        bash "$EXP/../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/vista/run.sh" train 2>&1 | tee "$SMOKE_DIR/train_stage.log"
        [[ ${PIPESTATUS[0]} -eq 0 ]] || { echo "SMOKE FAILED in the train stage" >&2; exit 1; }
    else
        echo "=== stage $stage | ${SECONDS}s"; NODE_ROLE=1 bash "$HERE/run_regen.sh" "$stage" || { echo "SMOKE FAILED in stage $stage" >&2; exit 1; }
    fi
done
set +u; source "$EXP/settings.sh"; set -u
echo "=== checks | ${SECONDS}s"
python3 - "$DATA_DIR" "$WARM_START/model.safetensors" <<'PY' || exit 1
import sys, numpy as np
from safetensors import safe_open
data, ck = sys.argv[1], sys.argv[2]
with safe_open(ck, "np") as f: d, t = f.get_tensor("d2t"), f.get_tensor("t2d")
ok = np.array_equal(np.load(f"{data}/d2t-32000.npy"), d) and np.array_equal(np.load(f"{data}/t2d-32000.npy").astype(bool), t.astype(bool))
print("data dir holds the checkpoint's d2t/t2d:", ok); sys.exit(0 if ok else 1)
PY
python3 - "$SMOKE_DIR/train_stage.log" "$CKPT_DIR" <<'PY' || exit 1
import re, sys, os
log, ck = sys.argv[1], sys.argv[2]
txt = open(log, errors="ignore").read()
al = [float(x) for x in re.findall(r"train/accept_len=([0-9.]+)", txt)]
print("first logged train accept_len values:", al[:5], "| checkpoints:", sorted(os.listdir(ck))[:6])
assert al and al[0] > 2.0, "the first training steps look like random weights (accept_len near 1.0): the warm start did not load"
assert any(d.startswith("checkpoint") or d == "0" for d in os.listdir(ck)), "no checkpoint was written"
PY
echo "SMOKE PASSED | ${SECONDS}s | $(date)"
