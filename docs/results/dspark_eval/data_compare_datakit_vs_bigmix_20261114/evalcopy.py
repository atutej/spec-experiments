import glob, json, random, sys
import numpy as np
W = "/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114"; R = "/scratch/09749/atutej/marin_speculator/runs"
N = f"{R}/eval_matrix_nofin24h_panel_and_snowball"
P = np.uint64(1000003)
def ngram_hash(a, n=8):
    a = a.astype(np.uint64); k = len(a) - n + 1; h = np.zeros(k, np.uint64)
    for j in range(n): h = h * P + a[j:j + k]
    return h
def load_prompts(files, keyf):
    d = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); d[keyf(r)] = r["prompt"]
    return d
idx = lambda r: r["index"]; vt = lambda r: (r["val_row"], r["turn"]); st = lambda r: (r["task"], r["turn_message_index"])
SETS = {  # name: (completions dir, prompt files, key function)
 "GSM8K": (f"{N}/paper/nofin_orig_24h/gsm8k/c16", [f"{R}/paper_prompts_20261107/gsm8k/first_turn.jsonl"], idx),
 "MATH-500": (f"{N}/paper/nofin_orig_24h/math500/c16", [f"{R}/paper_prompts_20261107/math500/first_turn.jsonl"], idx),
 "HumanEval": (f"{N}/paper/nofin_orig_24h/humaneval/c16", [f"{R}/paper_prompts_20261107/humaneval/first_turn.jsonl"], idx),
 "MBPP": (f"{N}/paper/nofin_orig_24h/mbpp/c16", [f"{R}/paper_prompts_20261107/mbpp/first_turn.jsonl"], idx),
 "MT-Bench": (f"{N}/paper/nofin_orig_24h/mtbench/c16", [f"{R}/paper_prompts_20261107/mtbench/first_turn.jsonl"], idx),
 "AIME": (f"{R}/eval_matrix_aime_20261112/aime/nofin_orig_24h/aime/c16", [f"{R}/aime_prompts_20261112/aime/first_turn.jsonl"], idx),
 "Datakit held-out": (f"{N}/datakit/nofin_orig_24h/datakit/c16", glob.glob(f"{R}/val_prompts_multiturn_1024/*_turn.jsonl"), vt),
 "BigMix held-out": (f"{N}/bigmix/nofin_orig_24h/bigmix/c16", glob.glob(f"{R}/bigpool_val_prompts_1024/*_turn.jsonl"), vt),
 "finance_v1": (f"{N}/finance/nofin_orig_24h/finance/c16", [f"{R}/finance_val_prompts_20261111/first_turn.jsonl"], vt),
 "Snowball replay": (f"{N}/snowball_swe/nofin_orig_24h/snowball_swe/c16", glob.glob(f"{R}/snowball_swe_replay_prompts_20261112/ctx_*.jsonl"), st)}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom", trust_remote_code=True); rng = random.Random(0)
print(f"{'eval set':18} {'replies':>7} {'reply toks':>10} {'copied from prompt':>19} {'repeated within reply':>22} {'any (8-gram seen earlier)':>26}")
res = {}
for name, (cd, pf, kf) in SETS.items():
    prompts = load_prompts(pf, kf); recs = []
    for f in sorted(glob.glob(f"{cd}/completions*.jsonl")): recs += [json.loads(l) for l in open(f)]
    rng.shuffle(recs); recs = recs[:300]; n = ctx = selfc = matched = 0
    for r in recs:
        key = kf(r)
        if key not in prompts: continue
        pi = np.array(prompts[key], dtype=np.int64); ri = np.array(tok.encode(r["completion"], add_special_tokens=False), dtype=np.int64)
        if ri.size < 12: continue
        ids = np.concatenate([pi, ri]); h = ngram_hash(ids); u, fi, inv = np.unique(h, return_index=True, return_inverse=True); first = fi[inv]; rep = first != np.arange(h.size)
        L = ids.size; cs_ = lambda x: np.concatenate([[0], np.cumsum(x)])
        rep_ctx = rep & (first + 7 < pi.size); rep_self = rep & ~(first + 7 < pi.size)
        def cover(x):
            c = cs_(x); p = np.arange(L); lo = np.clip(p - 7, 0, h.size - 1); hi = np.clip(p, 0, h.size - 1); return (c[hi + 1] - c[lo]) > 0
        cc = cover(rep_ctx); cself = cover(rep_self) & ~cc; sl = slice(pi.size, L)
        n += ri.size; ctx += int(cc[sl].sum()); selfc += int(cself[sl].sum()); matched += 1
    res[name] = (matched, n, ctx / n if n else 0, selfc / n if n else 0)
    print(f"{name:18} {matched:7} {n:10} {100*ctx/max(n,1):18.1f}% {100*selfc/max(n,1):21.1f}% {100*(ctx+selfc)/max(n,1):25.1f}%", flush=True)
json.dump(res, open(f"{W}/results/evalcopy.json", "w"))
