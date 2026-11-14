#!/usr/bin/env python
"""Eval side: re-tokenize the target's saved replies for each eval set and measure, per drafter vocabulary, how many reply tokens it can propose at all.
  python evalside.py     -> results/evalside.npz (unigram counts per set, coverage per drafter, class mix) and results/evalside_top_uncovered.json"""
import glob, json, random, sys
import numpy as np
W = "/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114"; R = "/scratch/09749/atutej/marin_speculator/runs"
N = f"{R}/eval_matrix_nofin24h_panel_and_snowball"; A = f"{R}/eval_matrix_aime_20261112/aime/nofin_orig_24h/aime/c16"
SETS = {"GSM8K": f"{N}/paper/nofin_orig_24h/gsm8k/c16", "MATH-500": f"{N}/paper/nofin_orig_24h/math500/c16", "HumanEval": f"{N}/paper/nofin_orig_24h/humaneval/c16",
        "MBPP": f"{N}/paper/nofin_orig_24h/mbpp/c16", "MT-Bench": f"{N}/paper/nofin_orig_24h/mtbench/c16", "AIME": A,
        "Datakit held-out": f"{N}/datakit/nofin_orig_24h/datakit/c16", "Datakit long rows": f"{N}/datakit_full/nofin_orig_24h/datakit_full/c16",
        "BigMix held-out": f"{N}/bigmix/nofin_orig_24h/bigmix/c16", "finance_v1": f"{N}/finance/nofin_orig_24h/finance/c16", "Snowball replay": f"{N}/snowball_swe/nofin_orig_24h/snowball_swe/c16"}
MAXREP = 400
from transformers import AutoTokenizer
z = np.load(f"{W}/results/resources.npz", allow_pickle=True); cls = z["cls"].astype(np.int64); M = z["masks"]; ML = list(z["mask_labels"]); CN = list(z["class_names"])
tok = AutoTokenizer.from_pretrained("open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom", trust_remote_code=True)
V = 128256; out = {}; top = {}; rng = random.Random(0)
for name, d in SETS.items():
    reps = []
    for f in sorted(glob.glob(f"{d}/completions*.jsonl")): reps += [json.loads(l)["completion"] for l in open(f)]
    n_all = len(reps); rng.shuffle(reps); reps = reps[:MAXREP]
    ids = np.concatenate([np.array(x, dtype=np.int64) for x in tok(reps, add_special_tokens=False)["input_ids"]])
    uni = np.bincount(ids, minlength=V); cov = [float(M[k][ids].mean()) for k in range(len(M))]
    out[name] = dict(n_replies=n_all, n_used=len(reps), tokens=int(ids.size), classes={CN[i]: float((cls[ids] == i).mean()) for i in range(len(CN))}, coverage=dict(zip(ML, cov)))
    t = {}
    for lab in ("Datakit orig 24h", "BigMix TG 28h"):
        k = ML.index(lab); miss = np.where(~M[k])[0]; c = uni[miss]; o = np.argsort(-c)[:12]
        t[lab] = [(tok.decode([int(miss[i])]), round(100 * c[i] / ids.size, 3)) for i in o if c[i] > 0]
    top[name] = t; np.save(f"{W}/results/evaluni_{name.replace(' ', '_')}.npy", uni)
    print(f"{name:18} replies {n_all:5} (used {len(reps)}) tokens {ids.size:8} | uncovered %: " + " ".join(f"{ML[k].split()[0][:3]}{ML[k].split()[1][:2]}={100*(1-cov[k]):.2f}" for k in range(len(M))), flush=True)
json.dump(out, open(f"{W}/results/evalside.json", "w"), indent=1); json.dump(top, open(f"{W}/results/evalside_top_uncovered.json", "w"), indent=1)
