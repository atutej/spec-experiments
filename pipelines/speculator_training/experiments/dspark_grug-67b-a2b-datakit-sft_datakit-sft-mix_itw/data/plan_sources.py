"""Plan the Datakit SFT download: for each of the 203 sources in the Grug 09-21 mix, which raw HF dataset/revision/files, how big, and its token allocation."""
import csv, fnmatch, json, sys
from collections import defaultdict
from huggingface_hub import HfApi
from marin.datakit.sft_sources import all_sft_sources

MIX = "/scratch/09749/atutej/marin_datakit/marin/experiments/grug_sft/special_token_1000_mix.json"
alloc = json.load(open(MIX))["allocations_tokens"]
sources = all_sft_sources()

def raw_steps(step, seen=None):
    seen = seen if seen is not None else set()
    out = []
    for d in step.deps:
        if id(d) in seen: continue
        seen.add(id(d))
        if d.name.startswith("raw/") or "hf_dataset_id" in (d.hash_attrs or {}): out.append(d)
        out += raw_steps(d, seen)
    return out

api = HfApi()
tree_cache = {}
def tree(repo, rev):
    k = (repo, rev)
    if k not in tree_cache:
        tree_cache[k] = [(f.path, f.size) for f in api.list_repo_tree(repo, revision=rev, repo_type="dataset", recursive=True) if hasattr(f, "size") and f.size is not None]
    return tree_cache[k]

rows = []
for name in sorted(alloc):
    src = sources[name]
    raws = raw_steps(src.chat_normalized)
    if not raws: rows.append(dict(source=name, alloc_tokens=alloc[name], raw="?", repo="?", revision="?", glob="?", files=0, bytes=0)); continue
    for r in raws:
        a = r.hash_attrs
        repo, rev, globs = a.get("hf_dataset_id"), a.get("revision"), a.get("hf_urls_glob") or []
        try:
            files = [(p, s) for p, s in tree(repo, rev) if not globs or any(fnmatch.fnmatch(p, g) or fnmatch.fnmatch(p.split("/")[-1], g) for g in globs)]
        except Exception as e:
            files = []; print("tree failed", repo, str(e)[:80], file=sys.stderr)
        rows.append(dict(source=name, alloc_tokens=alloc[name], raw=r.name, repo=repo, revision=(rev or "")[:10], glob=";".join(globs), files=len(files), bytes=sum(s for _, s in files)))

out = "/scratch/09749/atutej/marin_datakit/source_plan.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
by_repo = defaultdict(lambda: [0, 0, set()])
for r in rows:
    k = (r["repo"], r["revision"]); by_repo[k][0] = r["bytes"]; by_repo[k][2].add(r["source"])
print("sources", len(alloc), "| plan rows", len(rows), "| distinct (repo, revision) downloads", len(by_repo))
tot = sum(v[0] for v in by_repo.values()); print(f"total raw download (distinct repo+glob, approx): {tot/1e9:,.1f} GB")
for (repo, rev), (b, _, srcs) in sorted(by_repo.items(), key=lambda kv: -kv[1][0])[:15]:
    print(f"  {b/1e9:9.2f} GB  {repo}  ({len(srcs)} sources)")
print("unresolved:", [r["source"] for r in rows if r["repo"] in ("?", None)][:10])
