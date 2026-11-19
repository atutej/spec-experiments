"""Plan the download of a Datakit SFT mixture: resolve every source to its Hugging Face dataset(s), pinned revisions and file globs, and size them.

Reads the vendored mix spec (mixes/<mix>.json; `allocations_tokens` per source) and resolves each source through Marin's own registry
(`marin.datakit.sft_sources.all_sft_sources`) at the Marin commit pinned in mixes/pins.json. Asks the Hub how large each download is; nothing
is downloaded. Writes a JSON plan and prints a summary by source family. Runs in the venv `setup_env.sh` builds:

    <root>/venv/bin/python -I tools/plan.py --mix datakit_sft_1005 --out <root>/plan.json
"""

import argparse
import collections
import fnmatch
import json
from pathlib import Path

from huggingface_hub import HfApi
from marin.datakit.sft_sources import all_sft_sources

PIPELINE_DIR = Path(__file__).resolve().parent.parent


def download_steps(step, seen=None):
    """Yield every step in step's dependency closure whose hash_attrs name a Hugging Face dataset."""
    seen = seen if seen is not None else set()
    if id(step) in seen:
        return
    seen.add(id(step))
    if "hf_dataset_id" in step.hash_attrs:
        yield step
    for dep in step.deps:
        yield from download_steps(dep, seen)


def family(name: str) -> str:
    return name.split("/")[0]


def repo_size(api: HfApi, repo: str, revision: str, globs: list[str], cache: dict) -> tuple[int, int, str | None]:
    """(bytes, files, error) of the files in a dataset repo at a revision matching any glob (all files if no globs)."""
    key = (repo, revision, tuple(globs))
    if key not in cache:
        try:
            files = [
                (e.path, e.size or 0)
                for e in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
                if hasattr(e, "size")
            ]
            if globs:
                files = [(p, s) for p, s in files if any(fnmatch.fnmatch(p, g) for g in globs)]
            cache[key] = (sum(s for _, s in files), len(files), None)
        except Exception as error:  # noqa: BLE001 - report per repo, keep planning
            cache[key] = (0, 0, f"{type(error).__name__}: {str(error)[:100]}")
    return cache[key]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mix", required=True, help="mix name in mixes/pins.json, e.g. datakit_sft_1005")
    parser.add_argument("--out", required=True, help="where to write the JSON plan")
    args = parser.parse_args()

    pin = json.loads((PIPELINE_DIR / "mixes" / "pins.json").read_text())[args.mix]
    allocations = json.loads((PIPELINE_DIR / "mixes" / pin["mix_file"]).read_text())["allocations_tokens"]
    registry = all_sft_sources()
    missing = sorted(set(allocations) - set(registry))
    if missing:
        raise SystemExit(f"Mixture sources missing from the registry (wrong Marin commit?): {missing[:5]}")

    api, cache = HfApi(), {}
    plan = {}
    for name in sorted(allocations):
        source = registry[name]
        downloads = []
        for step in download_steps(source.chat_steps[-1]):
            attrs = step.hash_attrs
            size, n_files, error = repo_size(
                api, attrs["hf_dataset_id"], attrs.get("revision"), list(attrs.get("hf_urls_glob") or []), cache
            )
            downloads.append(
                {
                    "hf_dataset_id": attrs["hf_dataset_id"],
                    "revision": attrs.get("revision"),
                    "hf_urls_glob": list(attrs.get("hf_urls_glob") or []),
                    "bytes": size,
                    "files": n_files,
                    "error": error,
                }
            )
        plan[name] = {
            "allocated_tokens": allocations[name],
            "rough_token_count_b": source.rough_token_count_b,
            "chat_steps": [s.name for s in source.chat_steps],
            "downloads": downloads,
        }

    Path(args.out).write_text(json.dumps(plan, indent=2, sort_keys=True))

    by_family = collections.defaultdict(lambda: {"sources": 0, "tokens": 0, "repos": {}})
    for name, entry in plan.items():
        fam = by_family[family(name)]
        fam["sources"] += 1
        fam["tokens"] += entry["allocated_tokens"]
        for d in entry["downloads"]:
            fam["repos"][(d["hf_dataset_id"], d["revision"], tuple(d["hf_urls_glob"]))] = d["bytes"]
    print(f"{'family':36s} {'sources':>7s} {'alloc tokens':>14s} {'download GB':>12s}")
    total_bytes = 0
    for fam, v in sorted(by_family.items(), key=lambda kv: -kv[1]["tokens"]):
        gb = sum(v["repos"].values()) / 1e9
        total_bytes += sum(v["repos"].values())
        print(f"{fam:36s} {v['sources']:7d} {v['tokens'] / 1e9:13.3f}B {gb:12.1f}")
    unique = {k for v in by_family.values() for k in v["repos"]}
    errors = [(n, d["hf_dataset_id"], d["error"]) for n, e in plan.items() for d in e["downloads"] if d["error"]]
    no_download = [n for n, e in plan.items() if not e["downloads"]]
    print(f"total unique downloads: {len(unique)}  ~{total_bytes / 1e9:.1f} GB (sum over families; a repo shared by two families is counted twice)")
    print(f"sources with no HF download step found: {len(no_download)} {no_download[:5]}")
    print(f"repos that errored: {len(errors)} {errors[:5]}")


if __name__ == "__main__":
    main()
