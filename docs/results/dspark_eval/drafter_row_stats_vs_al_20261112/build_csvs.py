#!/usr/bin/env python
"""Build the two CSVs of this folder from the eval result folders (runs/eval_matrix_*) and the dataset manifests. Run with the speculators env (needs torch for token_freq.pt):
   envs/speculators/bin/python -I build_csvs.py
Outputs: method_summary.csv (one row per method/checkpoint: training-data row statistics, steps, tokens seen, mean AL) and al_by_benchmark.csv (long: method x benchmark: AL, tokens/s, source folder)."""
import csv, json, os, sys
import torch
ROOT = "/scratch/09749/atutej/marin_speculator"; R = f"{ROOT}/runs"
sys.path.insert(0, f"{ROOT}/spec-experiments/pipelines/speculator_training/tools/eval_acceptance")
import speedup_report as sr
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- datasets: rows, tokens, supervised tokens, train-split share, epoch length (steps) ------------------------------------------------------------------------------------------------
def manifest(p): return json.load(open(p))["totals"]
def tf_sum(p): t = torch.load(p, weights_only=False); return sum(t.values())
DK = manifest(f"{ROOT.replace('marin_speculator','')}marin_datakit/dataset_seq32768/manifest.json")
NF = manifest(f"{ROOT.replace('marin_speculator','')}marin_datakit/dataset_seq32768_nofinance/manifest.json")
BM = manifest(f"{ROOT.replace('marin_speculator','')}marin_big_pool/dataset_itw_seq32768/manifest.json")
DATA = {  # key: (rows, total tokens, supervised tokens, train share, steps per epoch, tokens_exact)
 "datakit":   (DK["rows_kept"], DK["tokens_kept"], DK["tokens_supervised"], 6464687 / 6471159, 382747, True),
 "nofinance": (NF["rows_kept"], NF["tokens_kept"], NF["tokens_supervised"], 0.999, 300412, True),          # epoch length derived from the scheduler's learning rate (validated: gives 382,745 for the full-Datakit run, known 382,747)
 "bigmix":    (BM["rows_kept"], BM["tokens_kept"], BM["tokens_supervised"], 0.995, 34167, True),
 "bigmix_tg": (296000, 3215825644, tf_sum(f"{R}/grug_dspark_drdoom_grug6t5_bigmix_targetgen296k/data/token_freq.pt"), 0.995, 27081, True),   # tokens from the merge log
 "datakit_tg": (281000, round(281000 * 9490), tf_sum(f"{R}/grug_dspark_drdoom_grug6t5_datakit_targetgen281k/data/token_freq.pt"), 0.995, 22472, False)}  # total tokens ESTIMATED from a row sample (9,490 per row)

# ---- methods: (label, short, data key, prompt set, reply source, checkpoint step, used in the figure, checkpoint dir, {benchmark: (bases..., arm)}) ----------------------------------------
E = lambda n: f"{R}/eval_matrix_{n}"
BENCH = ["datakit", "datakit_full", "finance", "bigmix", "gsm8k", "math500", "humaneval", "mbpp", "mtbench", "speedbench"]
def where(arm_by_base):  # list of (eval folder, arm) tried in order
    return arm_by_base
METHODS = [
 ("Datakit original 12h", "Datakit orig 12h", "datakit", "Datakit-SFT", "original", 37917, False, "runs/eval_ckpts/drdoom_grug6_taps5_itw_step37917", [(E("g6t5-step37917-speedup"), "g6t5_step37917"), (E("g6t5-paper"), "g6t5_step37917")]),
 ("Datakit original 24h", "Datakit orig", "datakit", "Datakit-SFT", "original", 75834, True, "runs/eval_ckpts/drdoom_grug6_taps5_itw_step75834", [(E("g6t5-step75834"), "g6t5_step75834"), (E("datakit-full-4way"), "datakit_orig_step75834"), (E("finance-datakitorig-bigmixtg"), "datakit_orig_step75834")]),
 ("NoFin original 6h", "NoFin orig 6h", "nofinance", "Datakit-SFT w/o finance", "original", 17986, False, "runs/eval_ckpts/datakit_nofinance6_taps5_itw_step17986", [(E("datakit-nofinance-6h"), "datakit_nofinance_6h_step17986")]),
 ("NoFin original 14h", "NoFin orig", "nofinance", "Datakit-SFT w/o finance", "original", 45747, True, "runs/eval_ckpts/datakit_nofinance6_taps5_itw_step45747", [(E("datakit-nofinance-step45747"), "datakit_nofinance_step45747")]),
 ("BigMix original 24h", "BigMix orig", "bigmix", "BigMix", "original", 95146, True, "runs/eval_ckpts/bigmix6_taps5_itw_step95146", [(E("bigmix6-12h-24h"), "bigmix6_step95146"), (E("datakit-full-4way"), "bigmix_orig_step95146")]),
 ("BigMix targetgen", "BigMix TG", "bigmix_tg", "BigMix", "targetgen", 27081, True, "runs/eval_ckpts/bigmix6_taps5_targetgen_epoch0_end", [(E("bigmix6-targetgen-24h"), "bigmix6_targetgen_epoch0_end"), (E("datakit-full-4way"), "bigmix_tg_epoch_end"), (E("finance-datakitorig-bigmixtg"), "bigmix_tg_epoch_end")]),
 ("Datakit targetgen", "Datakit TG", "datakit_tg", "Datakit-SFT", "targetgen", 22472, True, "runs/eval_ckpts/datakit6_taps5_targetgen_epoch0_end", [(E("datakit6-targetgen-24h"), "datakit6_targetgen_epoch0_end"), (E("datakit-full-4way"), "datakit_tg_epoch_end")]),
]
PAPER = {"gsm8k", "math500", "humaneval", "mbpp", "mtbench"}
def get(bases, bench):
    for b, arm in bases:
        d = f"{b}/{'paper' if bench in PAPER else bench}/{arm}"
        if not os.path.isdir(d): continue
        if bench == "speedbench":
            rep = f"{d}/speed_bench_report.csv"
            if not os.path.exists(rep): continue
            a = sr.speedbench(d).get("ALL"); return float(list(csv.DictReader(open(rep)))[0]["Overall"]), (a["tok_s"] if a else None), b
        x = sr.heldout(d, bench).get("c16")
        if x and x.get("al"): return x["al"], x["tok_s"], b
    return None
NEUTRAL = ["gsm8k", "math500", "humaneval", "mbpp", "mtbench", "speedbench"]
long_rows, summary = [], []
for label, short, dk, pset, reply, step, used, ckdir, bases in METHODS:
    rows, tok, sup, share, T, tok_exact = DATA[dk]
    train_rows, train_sup, train_tok = rows * share, sup * share, tok * share
    rps, per_step_sup, per_step_tok = train_rows / T, train_sup / T, train_tok / T
    seen_sup, seen_tok = per_step_sup * step, per_step_tok * step
    al = {}
    for b in BENCH:
        r = get(bases, b)
        if r: al[b] = r[0]; long_rows.append([label, step, b, round(r[0], 4), None if r[1] is None else round(r[1], 1), os.path.relpath(r[2], R)])
    neutral = [al[b] for b in NEUTRAL if b in al]; assert len(neutral) == 6, (label, neutral)
    summary.append({"method": label, "short_label": short, "used_in_figure": used, "checkpoint_step": step, "checkpoint_dir": ckdir, "prompt_set": pset, "reply_source": reply,
        "dataset_rows": rows, "dataset_tokens": tok, "dataset_supervised_tokens": sup, "train_split_share": round(share, 6), "epoch_length_steps": T, "epochs_seen": round(step / T, 4),
        "rows_per_step": round(rps, 3), "tokens_per_row_mean": round(tok / rows, 1), "supervised_per_row_mean": round(sup / rows, 1), "supervised_share": round(sup / tok, 4),
        "tokens_per_step": round(per_step_tok), "supervised_per_step": round(per_step_sup), "supervised_seen_B": round(seen_sup / 1e9, 4), "unique_supervised_seen_B": round(min(seen_sup, train_sup) / 1e9, 4),
        "total_tokens_seen_B": round(seen_tok / 1e9, 4), "anchor_cap_share_of_supervised_per_step": round(min(1, 4 * 12288 / per_step_sup), 3),
        "mean_AL_six_neutral": round(sum(neutral) / 6, 4), **{f"AL_{b}": (round(al[b], 4) if b in al else "") for b in BENCH},
        "tokens_per_row_is_estimate": not tok_exact,
        "notes": ("total tokens per row estimated from a row sample (9,490)" if not tok_exact else "") })
with open(f"{HERE}/method_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
with open(f"{HERE}/al_by_benchmark.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "checkpoint_step", "benchmark", "acceptance_length", "tokens_per_s", "source_eval_folder"]); w.writerows(long_rows)
print(f"method_summary.csv: {len(summary)} methods; al_by_benchmark.csv: {len(long_rows)} rows")
for s in summary: print(f"{s['method']:24} step {s['checkpoint_step']:6} epochs {s['epochs_seen']:5.2f} sup/row {s['supervised_per_row_mean']:8.0f} tok/row {s['tokens_per_row_mean']:8.0f} share {s['supervised_share']:.2f} seen {s['supervised_seen_B']:5.2f}B unique {s['unique_supervised_seen_B']:5.2f}B meanAL {s['mean_AL_six_neutral']:.3f}")
