"""Write eval prompt sets without the problems found in any of the four training datasets (>= 50% of the problem's 13-grams in a training prompt).
Problems too short for the 13-gram check (under 30 tokens) cannot be tested and stay in the 'clean' sets: MBPP is almost entirely untested."""
import json, re, os, numpy as np
from transformers import AutoTokenizer
W = "/scratch/09749/atutej/marin_speculator/runs/data_compare_20261114"; R = "/scratch/09749/atutej/marin_speculator/runs"; OUT = f"{R}/paper_prompts_decontam_20261114"
TOK = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"; SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SETS = {"GSM8K": f"{R}/paper_prompts_20261107/gsm8k", "MATH-500": f"{R}/paper_prompts_20261107/math500", "HumanEval": f"{R}/paper_prompts_20261107/humaneval",
        "MBPP": f"{R}/paper_prompts_20261107/mbpp", "MT-Bench": f"{R}/paper_prompts_20261107/mtbench", "AIME": f"{R}/aime_prompts_20261112/aime"}
assert not os.path.exists(OUT), "fresh dir only"
tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
z = np.load(f"{W}/results/contam_index.npz", allow_pickle=True); setof = list(z["set_of"])
best = np.max([np.load(f"{W}/results/contam_{k}.npz")["best"] for k in ("datakit_orig", "bigmix_orig", "datakit_tg", "bigmix_tg")], axis=0)
e = 0; flagged = {}; checked = {}
for name, d in SETS.items():
    seen = set(); flagged[name] = set(); checked[name] = 0
    for line in open(f"{d}/first_turn.jsonl"):
        r = json.loads(line); key = r.get("index", len(seen))
        if key in seen: continue
        seen.add(key); s = tok.decode(r["prompt"], skip_special_tokens=False)
        m = re.search(r"user<\|end_header_id\|>\n(.*)<\|eot_id\|>\n<\|start_header_id\|>assistant", s, re.S)
        if not m: continue
        c = m.group(1); c = c[:-len(SUFFIX)] if c.endswith(SUFFIX) else c
        if name == "HumanEval": c = re.sub(r"^Write a solution to the following problem and make sure that it passes the tests:\n```python\n", "", c); c = re.sub(r"\n```$", "", c)
        if len(tok.encode(c, add_special_tokens=False)) < 30: continue
        assert setof[e] == name, (e, setof[e], name); checked[name] += 1
        if best[e] >= 0.5: flagged[name].add(key)
        e += 1
assert e == len(setof), (e, len(setof))
man = {}
for name, d in SETS.items():
    os.makedirs(f"{OUT}/{name.lower().replace('-', '')}", exist_ok=True); kept = dropped = 0
    with open(f"{OUT}/{name.lower().replace('-', '')}/first_turn.jsonl", "w") as f:
        for line in open(f"{d}/first_turn.jsonl"):
            r = json.loads(line)
            if r.get("index") in flagged[name]: dropped += 1; continue
            f.write(line); kept += 1
    man[name] = dict(rows_kept=kept, rows_dropped=dropped, problems_checked=checked[name], problems_flagged=len(flagged[name]))
    print(f"{name:10} kept {kept:5} dropped {dropped:4} (problems checked {checked[name]:5}, flagged {len(flagged[name]):4})")
json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)
