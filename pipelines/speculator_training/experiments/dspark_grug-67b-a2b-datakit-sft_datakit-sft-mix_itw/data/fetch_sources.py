"""Download and chat-normalize the Datakit SFT sources of the Grug 2026.09.21 mix, reusing Marin's Datakit code as is.

Run with the marin datakit env (not the speculators env):
    /scratch/09749/atutej/marin_datakit/venv/bin/python -I fetch_sources.py            # print the plan
    /scratch/09749/atutej/marin_datakit/venv/bin/python -I fetch_sources.py --run      # execute

What it does: the sources are the ones in Marin's `experiments/grug_sft/special_token_1000_mix.json` (the 203 sources Grug was trained on),
looked up in Marin's registry (`marin.datakit.sft_sources.all_sft_sources`). For each, Marin's own steps run through its own `StepRunner`:
download the raw HF dataset at the pinned revision -> source-specific chat transform -> `normalize_chat` (Harmony messages, exact-id dedup).
It stops at `chat_normalized`: it does not render or tokenize (Marin's `text` stage in `experiments/datakit/sft_chat_tokenize.py` renders all
249 registered sources to flat text, which loses the assistant spans we need as a loss mask).

Output: $MARIN_PREFIX/normalized-chat/<source>_<hash>/outputs/main/*.parquet (columns id, messages, source, source_id, chat_template_kwargs).
Steps whose status file says success are skipped, so a rerun resumes. Env: MARIN_DATAKIT_DIR (default /scratch/09749/atutej/marin_datakit),
MARIN_PREFIX (default $MARIN_DATAKIT_DIR/out), HF_TOKEN.
"""
import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

DATAKIT_DIR = Path(os.environ.get("MARIN_DATAKIT_DIR", "/scratch/09749/atutej/marin_datakit"))
os.environ.setdefault("MARIN_PREFIX", str(DATAKIT_DIR / "out"))
MIX = DATAKIT_DIR / "marin/experiments/grug_sft/special_token_1000_mix.json"

from marin.datakit.sft_sources import all_sft_sources  # noqa: E402  (after MARIN_PREFIX is set)
from marin.execution.step_runner import StepRunner  # noqa: E402


def selected(patterns: list[str], exclude: list[str]) -> dict[str, int]:
    alloc = json.loads(MIX.read_text())["allocations_tokens"]
    names = {n: t for n, t in alloc.items() if (not patterns or any(fnmatch.fnmatch(n, p) for p in patterns))}
    return {n: t for n, t in names.items() if not any(fnmatch.fnmatch(n, p) for p in exclude)}


def succeeded(step) -> bool:
    status = Path(step.output_path) / ".executor_status"
    return status.exists() and "SUCCESS" in status.read_text()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=[], help="fnmatch patterns on source names (default: all 203)")
    ap.add_argument("--exclude", nargs="*", default=[], help="fnmatch patterns to skip")
    ap.add_argument("--max-concurrent", type=int, default=4, help="steps run at the same time")
    ap.add_argument("--run", action="store_true", help="execute; otherwise print the plan")
    args = ap.parse_args()

    registry = all_sft_sources()
    chosen = selected(args.only, args.exclude)
    missing = sorted(set(chosen) - set(registry))
    if missing:
        sys.exit(f"sources in the mix but not in Marin's registry: {missing}")
    steps = [registry[n].chat_normalized for n in chosen]
    done = [succeeded(s) for s in steps]
    tokens = sum(chosen.values())
    print(f"MARIN_PREFIX={os.environ['MARIN_PREFIX']}")
    print(f"{len(steps)} sources, {tokens/1e9:.2f}B allocated SFT tokens, {sum(done)} already done, {len(steps)-sum(done)} to run", flush=True)
    for (n, t), s, d in zip(chosen.items(), steps, done):
        print(f"  {'done' if d else 'todo'}  {t/1e6:10.1f}M  {n}")
    if args.run:
        StepRunner().run(steps, max_concurrent=args.max_concurrent)
        print("finished", flush=True)


if __name__ == "__main__":
    main()
