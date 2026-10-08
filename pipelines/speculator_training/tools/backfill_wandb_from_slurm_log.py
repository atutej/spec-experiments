#!/usr/bin/env python
"""Backfill a Weights & Biases run from the console log of a `speculators.train` job that ran without `--logger wandb`.

The trainer prints each logged record to the console and (when a logger is on) hands the same record to the W&B handler.
This tool reads the printed per-step records back from a Slurm log and logs them the way `WandbHandler` does:

  - metrics: the printed keys are already the flattened keys (`train/loss`, `profile/step_ms`, `lr/Muon`, `epoch`,
    `global_step`, ...); each record is sent with `run.log(flat, step=global_step)`.
  - config: rebuilt with the trainer's own `TrainConfig.resolve()` from the saved command line (`train_command.txt`),
    dumped like `log_run_config` does (`model_dump(mode="json")`), flattened with the trainer's `_flatten_dict`, and set
    with `run.config[k] = v`.

Limits you must know: the console prints rounded values (about 3 significant digits), so the uploaded curves have that
precision, not the exact floats the trainer would have sent. Timestamps are the time of upload. Validation records
(`val/...`) are not parsed: they exist in the log only if the validation epoch ran.

  python backfill_wandb_from_slurm_log.py --log job.out --train-command checkpoints/train_command.txt \
      --run-name NAME [--entity E --project P] [--parse-only] [--max-steps N] [--offline]
"""
import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
# first line of a record: "           INFO     train/confidence_loss=0.018,        trainer.py:569"
START = re.compile(r"^(?:\[\d\d:\d\d:\d\d\])?\s+INFO\s+(.*?)\s+trainer\.py:\d+\s*$")
CONT = re.compile(r"^ {18,}(\S.*?)\s*$")
PAIR = re.compile(r"([A-Za-z_]\w*(?:/\w+)*)=([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|None)(?=[,\s]|$)")


def _num(s: str):
    if s == "None":
        return None
    f = float(s)
    return int(f) if re.fullmatch(r"[-+]?\d+", s) else f


def parse_log(path: str, max_steps: int | None = None):
    """Return (records, anomalies). A record is a flat dict of printed keys incl. `global_step`."""
    records, anomalies = [], {"truncated_blocks": 0, "missing_keys": 0, "duplicate_steps": 0}
    cur = None  # list of text pieces of the record being read
    seen = set()
    expected = None
    with open(path, encoding="utf-8", errors="replace", newline=None) as f:  # newline=None: tqdm's \r count as line ends
        for raw in f:
            line = ANSI.sub("", raw.rstrip("\n"))
            m = START.match(line)
            if m and ("train/" in m.group(1) or "profile/" in m.group(1) or "lr/" in m.group(1)):
                if cur is not None:
                    anomalies["truncated_blocks"] += 1
                cur = [m.group(1)]
                continue
            if cur is None:
                continue
            c = CONT.match(line)
            if not c:
                continue  # interleaved output of other processes
            cur.append(c.group(1))
            if re.search(r"(?:^|[\s,])global_step=\d+\s*$", c.group(1)):
                rec = {k: _num(v) for k, v in PAIR.findall(" ".join(cur))}
                cur = None
                if "global_step" not in rec:
                    anomalies["truncated_blocks"] += 1
                    continue
                if expected is None:
                    expected = set(rec)
                elif set(rec) != expected:
                    anomalies["missing_keys"] += 1
                if rec["global_step"] in seen:
                    anomalies["duplicate_steps"] += 1
                    continue
                seen.add(rec["global_step"])
                records.append(rec)
                if max_steps is not None and len(records) >= max_steps:
                    break
    return records, anomalies


def build_config(train_command: str) -> dict:
    """The flat config the trainer logs as hyperparameters, rebuilt from the saved command line."""
    from speculators.train.config import TrainConfig
    from speculators.train.logger import _flatten_dict

    argv = None
    for line in Path(train_command).read_text().splitlines():
        if line and not line.startswith("#"):
            argv = shlex.split(line)
            break
    if argv is None:
        raise SystemExit(f"no command line in {train_command}")
    old = sys.argv
    sys.argv = ["speculators.train", *argv[1:]]  # argv[0] is the path of __main__.py
    try:
        cfg = TrainConfig.resolve()
    finally:
        sys.argv = old
    resolved = cfg.model_dump(mode="json")
    if getattr(cfg, "backend_args", None):
        resolved["hidden_states_backend_args"] = dict(cfg.backend_args)
    return _flatten_dict(resolved)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--train-command", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
    ap.add_argument("--notes", default="")
    ap.add_argument("--tags", nargs="*", default=["backfilled-from-slurm-log"])
    ap.add_argument("--max-steps", type=int, default=None, help="only the first N records (testing)")
    ap.add_argument("--parse-only", action="store_true", help="parse and print statistics; do not touch W&B")
    ap.add_argument("--offline", action="store_true", help="W&B offline mode: write locally, upload nothing")
    ap.add_argument("--wandb-dir", default=None)
    ap.add_argument("--dump-jsonl", default=None, help="also write the parsed records here")
    args = ap.parse_args()

    records, anomalies = parse_log(args.log, args.max_steps)
    steps = [r["global_step"] for r in records]
    print(f"parsed {len(records)} records; steps {min(steps)}..{max(steps)}; anomalies: {anomalies}")
    gaps = sorted(set(range(min(steps), max(steps) + 1)) - set(steps))
    print(f"missing steps inside the range: {len(gaps)}" + (f" (first few {gaps[:5]})" if gaps else ""))
    print("keys per record:", len(records[0]), "->", ", ".join(records[0]))
    if args.dump_jsonl:
        with open(args.dump_jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    if args.parse_only:
        return

    if not args.entity or not args.project:
        raise SystemExit("set --entity and --project (or WANDB_ENTITY / WANDB_PROJECT)")
    config = build_config(args.train_command)
    print(f"config: {len(config)} flattened hyperparameters (rebuilt with TrainConfig.resolve())")
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
    import wandb

    run = wandb.init(entity=args.entity, project=args.project, name=args.run_name, notes=args.notes,
                     tags=args.tags, dir=args.wandb_dir, config={})
    for k, v in config.items():  # as WandbHandler does for hparams records
        run.config[k] = v
    for i, rec in enumerate(records):
        run.log(rec, step=int(rec["global_step"]))
        if (i + 1) % 10000 == 0:
            print(f"  logged {i + 1}/{len(records)}", flush=True)
    print(f"run: entity={run.entity} project={run.project} name={run.name} id={run.id} url={getattr(run, 'url', None)}")
    run.finish()


if __name__ == "__main__":
    main()
