"""Acceptance length of ONE draft round started at uniformly random positions of the target's own completions, measured on real vLLM (the trainer's anchor scheme).

  phase 1 (if --seqs-file is missing): the target generates its own reply to each prompt (T=1, token ids kept) -> saved to --seqs-file, so every drafter is probed on the SAME text
  phase 2: for each reply, K positions m drawn uniformly inside it; request prompt = prompt + reply[:m], max_tokens=2. The prefill samples one token, the next engine step
           is the first draft round (8 drafts verified); the counters of that one round are read from /metrics (num_drafts, num_accepted_tokens). Sanity check: the number of
           rounds counted must equal the number of probes.
  python probe_random_anchor.py --target http://127.0.0.1:8000 --model <target> --prompts-dir runs/val_prompts_1024 --seqs-file <json> --out <json> [--rows 400 --per-row 8]
"""
import argparse, json, os, random, sys, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/scratch/09749/atutej/marin_speculator/speculators/scripts/evaluate")
from perf_utils import extract_spec_decode_metrics, fetch_metrics, parse_prometheus_metrics  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
ap.add_argument("--prompts-dir"); ap.add_argument("--seqs-file"); ap.add_argument("--out", required=True)
ap.add_argument("--dataset", help="instead of replies to prompts: probe a trainer-format dataset's HELD-OUT rows (last 1-train-ratio by position), at uniformly random SUPERVISED positions, with the whole row prefix as context")
ap.add_argument("--train-ratio", type=float, default=0.995)
ap.add_argument("--rows", type=int, default=400); ap.add_argument("--per-row", type=int, default=8)
ap.add_argument("--gen-max-tokens", type=int, default=2048); ap.add_argument("--concurrency", type=int, default=64); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

def post(body, rid=None):
    req = urllib.request.Request(f"{a.target}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"][0]

if a.dataset:
    import numpy as np
    from datasets import load_from_disk
    ds = load_from_disk(a.dataset); N = len(ds); lo = int(N * a.train_ratio)
    rng = random.Random(a.seed + 1); probes = []
    pick = rng.sample(range(lo, N), min(a.rows, N - lo))
    for i in pick:
        r = ds[i]; ids, mk = r["input_ids"], np.asarray(r["loss_mask"])
        ok = np.nonzero((mk[1:] == 1) & (mk[:-1] == 1))[0] + 1                # m inside a supervised span
        ok = ok[(ok + 4 < len(ids)) & (ok < 30000)]
        if len(ok) == 0: continue
        for m in rng.sample(ok.tolist(), min(a.per_row, len(ok))):
            probes.append([int(t) for t in ids[:m]])
    seqs = [None] * len(pick)
    print(f"held-out rows {lo}..{N}: {len(pick)} rows sampled, mean prefix tokens {sum(map(len, probes))/max(len(probes),1):.0f}", flush=True)
else:
    rows = [json.loads(l) for l in open(f"{a.prompts_dir}/first_turn.jsonl")][: a.rows]
    if not os.path.exists(a.seqs_file):
        print(f"phase 1: {len(rows)} target completions", flush=True)
        def gen(i):
            c = post({"model": a.model, "prompt": rows[i]["prompt"], "max_tokens": a.gen_max_tokens, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
                      "seed": a.seed + i, "return_token_ids": True, "skip_special_tokens": False})
            return {"prompt": rows[i]["prompt"], "reply": c["token_ids"], "finish": c["finish_reason"]}
        with ThreadPoolExecutor(a.concurrency) as ex: seqs = list(ex.map(gen, range(len(rows))))
        json.dump(seqs, open(a.seqs_file, "w"))
    else:
        seqs = json.load(open(a.seqs_file))
    rng = random.Random(a.seed + 1); probes = []
    for s in seqs:
        n = len(s["reply"])
        if n < 6: continue
        for m in rng.sample(range(1, n - 3), min(a.per_row, n - 4)):
            probes.append(s["prompt"] + s["reply"][:m])

print(f"phase 2: {len(probes)} probes", flush=True)
base = parse_prometheus_metrics(fetch_metrics(f"{a.target}/metrics"))
errs = [0]; lock = threading.Lock()
def one(i):
    try: post({"model": a.model, "prompt": probes[i], "max_tokens": 2, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": a.seed + 10_000 + i})
    except Exception:  # noqa: BLE001
        with lock: errs[0] += 1
with ThreadPoolExecutor(a.concurrency) as ex: list(ex.map(one, range(len(probes))))
cur = parse_prometheus_metrics(fetch_metrics(f"{a.target}/metrics"))
m = extract_spec_decode_metrics(cur, baseline_metrics=base)
res = {"probes": len(probes), "errors": errs[0], **{k: m[k] for k in m if not isinstance(m[k], (list, dict))}}
print(json.dumps(res, indent=1)); json.dump(res, open(a.out, "w"))
print(f"one-round acceptance length {res.get('acceptance_length')}: rounds counted {res.get('num_drafts')} vs probes {len(probes)}")
