#!/usr/bin/env python
"""Turn results/stats_*.npz and evalside into the comparison tables (prints, and writes results/report_tables.md)."""
import json, sys
import numpy as np
W = "/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114"
R = np.load(f"{W}/results/resources.npz", allow_pickle=True); CN = list(R["class_names"]); ML = list(R["mask_labels"]); V = 128256
DS = {"datakit_orig": "Datakit orig", "bigmix_tg": "BigMix TG", "datakit_tg": "Datakit TG", "bigmix_orig": "BigMix orig"}
S = {k: np.load(f"{W}/results/stats_{k}.npz", allow_pickle=True) for k in DS}; CAT = ["think", "nothink", "none"]
out = []
def p(s=""): print(s); out.append(s)
def row(label, vals, fmt): p(f"| {label} | " + " | ".join(fmt(v) if v is not None else "-" for v in vals) + " |")
def header(first): p(f"| {first} | " + " | ".join(DS.values()) + " |"); p("|---|" + "---:|" * len(DS))
def tot(z, k): return z[k].sum()
p("## 1. Structure (per row)"); header("measure")
n = {k: S[k]["rows"].sum() - S[k]["empty"].sum() for k in S}
row("rows scanned", [int(S[k]["rows"].sum()) for k in S], lambda v: f"{v:,}")
row("tokens per row", [tot(S[k], "tokens") / S[k]["rows"].sum() for k in S], lambda v: f"{v:,.0f}")
row("supervised tokens per row", [tot(S[k], "sup") / n[k] for k in S], lambda v: f"{v:,.0f}")
row("prompt-side (unsupervised) tokens per row", [tot(S[k], "prompt") / n[k] for k in S], lambda v: f"{v:,.0f}")
row("assistant turns per row", [tot(S[k], "turns") / n[k] for k in S], lambda v: f"{v:.2f}")
row("rows at the 32,768 window", [100 * tot(S[k], "window") / n[k] for k in S], lambda v: f"{v:.1f}%")
row("supervised share of all tokens", [100 * tot(S[k], "sup") / tot(S[k], "tokens") for k in S], lambda v: f"{v:.1f}%")
p(); p("## 2. Reasoning mode and terminal-agent prompts"); header("share")
for mi, mn in enumerate(CAT):
    row(f"rows with `/{mn}` header", [100 * (S[k]["rows"][mi * 2:mi * 2 + 2].sum()) / S[k]["rows"].sum() for k in S], lambda v: f"{v:.1f}%")
    row(f"supervised tokens in `/{mn}` rows", [100 * S[k]["sup"][mi * 2:mi * 2 + 2].sum() / tot(S[k], "sup") for k in S], lambda v: f"{v:.1f}%")
row("rows with the Terminus-2 terminal prompt", [100 * S[k]["rows"][[1, 3, 5]].sum() / S[k]["rows"].sum() for k in S], lambda v: f"{v:.1f}%")
row("supervised tokens in terminal rows", [100 * S[k]["sup"][[1, 3, 5]].sum() / tot(S[k], "sup") for k in S], lambda v: f"{v:.1f}%")
row("rows containing a think block", [100 * tot(S[k], "thinkrows") / n[k] for k in S], lambda v: f"{v:.1f}%")
row("think-block tokens as share of supervised", [100 * tot(S[k], "think_tok") / tot(S[k], "sup") for k in S], lambda v: f"{v:.1f}%")
row("think blocks never closed (truncated)", [100 * tot(S[k], "unclosed") / max(1, tot(S[k], "thinkrows")) for k in S], lambda v: f"{v:.1f}% of think rows")
p(); p("## 3. Per-category: supervised tokens per row, window rows, context copying"); 
CN6 = ["think / non-terminal", "think / terminal", "nothink / non-terminal", "nothink / terminal", "no header / non-terminal", "no header / terminal"]
for c in range(6):
    vals = []
    for k in S:
        z = S[k]; nn = z["rows"][c] - z["empty"][c]
        vals.append(None if nn < 200 else (100 * z["rows"][c] / z["rows"].sum(), z["sup"][c] / nn, 100 * z["window"][c] / nn, 100 * z["cp_ctx"][c] / z["sup"][c], 100 * z["cp_self"][c] / z["sup"][c]))
    if all(v is None for v in vals): continue
    p(f"\n**{CN6[c]}**: row share | supervised/row | window rows | supervised tokens copied from context | repeated from earlier assistant text"); header("")
    row("", [None if v is None else v for v in vals], lambda v: f"{v[0]:.1f}% / {v[1]:,.0f} / {v[2]:.1f}% / {v[3]:.1f}% / {v[4]:.1f}%")
p(); p("## 4. Whole-dataset copy shares (8-gram seen earlier in the row)"); header("share of supervised tokens")
row("copied from the prompt or tool output", [100 * tot(S[k], "cp_ctx") / tot(S[k], "sup") for k in S], lambda v: f"{v:.1f}%")
row("repeated from earlier assistant text", [100 * tot(S[k], "cp_self") / tot(S[k], "sup") for k in S], lambda v: f"{v:.1f}%")
p(); p("## 5. Token classes of supervised text"); header("class")
for i, c in enumerate(CN):
    row(c, [100 * S[k]["cls_hist"][:, i].sum() / tot(S[k], "sup") for k in S], lambda v: f"{v:.2f}%")
p(); p("## 6. Draft-vocabulary coverage of each dataset's own supervised tokens (rows: drafter vocabulary)"); header("drafter")
for m, lab in enumerate(ML): row(lab, [100 * S[k]["cov"][:, m].sum() / tot(S[k], "sup") for k in S], lambda v: f"{v:.2f}%")
p(); p("## 7. Supervised length per row: share of rows by size"); header("supervised tokens")
for lo, hi in [(0, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 10**9)]:
    def share(z): h = z["sup_hist"].sum(0); b = [(1 << i) - 1 for i in range(20)]; return 100 * sum(h[i] for i in range(20) if lo <= (1 << i) - 1 < hi) / h.sum()
    row(f"{lo:,} to {hi:,}" if hi < 10**9 else f"{lo:,}+", [share(S[k]) for k in S], lambda v: f"{v:.1f}%")
p(); p("## 8. Text-sample features (sampled rows)"); header("feature")
row("rows in the text sample", [int(tot(S[k], "txt_rows")) for k in S], lambda v: f"{v:,}")
row("zlib bytes / raw bytes of supervised text", [tot(S[k], "txt_zbytes") / tot(S[k], "txt_bytes") for k in S], lambda v: f"{v:.3f}")
mk = [r"Wait", r"Hmm", r"Alternatively", r"But wait", r"Let me", r"Actually", r"Okay"]
for q, name in enumerate(mk): row(f"`{name}` per 1k supervised tokens", [1000 * S[k]["markers"][:, q].sum() / tot(S[k], "txt_sup") for k in S], lambda v: f"{v:.2f}")
for key, name in [("f_boxed", "prompt asks for \\boxed"), ("f_step", "prompt says step by step"), ("f_fence", "prompt has a code fence"), ("f_latex", "prompt has LaTeX"), ("f_nonascii", "prompt mostly non-ASCII (>30%)")]:
    row(name, [100 * tot(S[k], key) / tot(S[k], "f_prompt600") for k in S], lambda v: f"{v:.1f}%")
# eval-likeness: cross-entropy of the target's eval replies under each dataset's unigram model of supervised /think text
p(); p("## 9. Unigram closeness of eval replies to each dataset's supervised text (bits per token of the replies under the dataset's unigram model; lower = closer)"); header("eval set")
evals = ["GSM8K", "MATH-500", "HumanEval", "MBPP", "MT-Bench", "AIME", "Datakit held-out", "BigMix held-out", "finance_v1", "Snowball replay"]
Q = {}
for k in S:
    u = S[k]["uni"][0].astype(np.float64) + 0.0; Q[k] = (u + 0.5) / (u.sum() + 0.5 * V)
for e in evals:
    cnt = np.load(f"{W}/results/evaluni_{e.replace(' ', '_')}.npy").astype(np.float64); pe = cnt / cnt.sum(); nz = pe > 0
    row(e, [-(pe[nz] * np.log2(Q[k][nz])).sum() for k in S], lambda v: f"{v:.3f}")
open(f"{W}/results/report_tables.md", "w").write("\n".join(out)); print("\n[written results/report_tables.md]")
