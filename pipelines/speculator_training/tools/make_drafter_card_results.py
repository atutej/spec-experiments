"""Markdown results tables for the drafter model cards, built from the saved serving-eval results (docs/results/dspark_eval/*/{plain,itw_80k,targetgen_final}/results.json).

  python make_drafter_card_results.py --results-dir docs/results/dspark_eval --itw-repo atutej/<name> --targetgen-repo atutej/<name> --this itw --out card_results_itw.md

Acceptance length is vLLM's `1 + accepted/drafts` over whole rollouts; the throughput gain is output tokens/s divided by plain decoding on the same prompts.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--itw-repo", required=True); ap.add_argument("--targetgen-repo", required=True)
ap.add_argument("--this", choices=["itw", "targetgen"], required=True, help="which drafter this card is for (its row is marked)")
a = ap.parse_args()
R = Path(a.results_dir)


def load(setdir: str, arm: str) -> dict:
    rows = json.loads((R / setdir / arm / "results.json").read_text())["results"]
    return {r["subset"]: r for r in rows}


def pooled_tps(d: dict) -> float:
    rs = [d[k] for k in ("first_turn", "later_turn") if k in d]
    return sum(r["completion_tokens"] for r in rs) / sum(r["wall_s"] for r in rs)


def mark(arm_kind: str, text: str) -> str:
    return f"**{text}**" if arm_kind == a.this else text


L = ["## Results (target: open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21)", "",
     f"Serving in the pinned Marin vLLM fork on one 4-GPU GB200 node (TP=1, DP=4, expert parallel), concurrency 16, T=1, 8 drafted tokens, probabilistic drafting with standard verification; "
     f"one run per arm. itw: [`{a.itw_repo}`](https://huggingface.co/{a.itw_repo}); targetgen: [`{a.targetgen_repo}`](https://huggingface.co/{a.targetgen_repo}). "
     "This card's drafter is in bold.", ""]

# (a) RL prompts
plain, itw, tg = (load("rl_prompts_256_target-09.21", x)["first_turn"] for x in ("plain", "itw_80k", "targetgen_final"))
L += ["### RL prompts (Snowball broad-domain RLVR training set, first 256 of a random 1,024, thinking on, up to 49,152 new tokens)", "",
      "| Arm | Acceptance length | Output tokens/s | Gain vs plain |", "|---|---|---|---|",
      f"| Plain decoding | n/a | {plain['output_tokens_per_s']:,.0f} | 1.00x |",
      f"| {mark('itw', 'itw drafter, step 80,813')} | {mark('itw', format(itw['acceptance_length'], '.3f'))} | {itw['output_tokens_per_s']:,.0f} | {mark('itw', format(itw['output_tokens_per_s'] / plain['output_tokens_per_s'], '.2f') + 'x')} |",
      f"| {mark('targetgen', 'targetgen drafter, step 17,737')} | {mark('targetgen', format(tg['acceptance_length'], '.3f'))} | {tg['output_tokens_per_s']:,.0f} | {mark('targetgen', format(tg['output_tokens_per_s'] / plain['output_tokens_per_s'], '.2f') + 'x')} |", ""]

# (a2) the same RL prompts on the Dr. Doom target (a fine-tune of the 09.21 target; the drafters were trained on 09.21's hidden states)
if (R / "rl_prompts_256_target-dr-doom").exists():
    dp, di, dt = (load("rl_prompts_256_target-dr-doom", x)["first_turn"] for x in ("plain", "itw_80k", "targetgen_final"))
    L += ["### RL prompts on the Dr. Doom target ([`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom), a de-looping fine-tune of the target above; same 256 prompts and settings)", "",
          "| Arm | Acceptance length | Output tokens/s | Gain vs plain |", "|---|---|---|---|",
          f"| Plain decoding | n/a | {dp['output_tokens_per_s']:,.0f} | 1.00x |",
          f"| {mark('itw', 'itw drafter, step 80,813')} | {mark('itw', format(di['acceptance_length'], '.3f'))} | {di['output_tokens_per_s']:,.0f} | {mark('itw', format(di['output_tokens_per_s'] / dp['output_tokens_per_s'], '.2f') + 'x')} |",
          f"| {mark('targetgen', 'targetgen drafter, step 17,737')} | {mark('targetgen', format(dt['acceptance_length'], '.3f'))} | {dt['output_tokens_per_s']:,.0f} | {mark('targetgen', format(dt['output_tokens_per_s'] / dp['output_tokens_per_s'], '.2f') + 'x')} |",
          "", "Both drafters were trained on the 09.21 target's hidden states, so this is transfer to a fine-tuned target: acceptance falls by about 0.4 to 0.5 relative to the table above.", ""]

# (b) Datakit validation, multi-turn
p, i, t = (load("datakit_val_multiturn_1024", x) for x in ("plain", "itw_80k", "targetgen_final"))
L += ["### Datakit validation prompts, multi-turn (1,024 held-out rows, random assistant turn with earlier turns as context, up to 4,096 new tokens)", "",
      "| Subset | Plain tok/s | itw: acceptance length | tok/s | gain | targetgen: acceptance length | tok/s | gain |", "|---|---|---|---|---|---|---|---|"]
for key, label, n in (("first_turn", "First turn", 833), ("later_turn", "Later turn", 191)):
    pt = p[key]["output_tokens_per_s"]
    L.append(f"| {label} ({n}) | {pt:,.0f} | {mark('itw', format(i[key]['acceptance_length'], '.3f'))} | {i[key]['output_tokens_per_s']:,.0f} | {mark('itw', format(i[key]['output_tokens_per_s'] / pt, '.2f') + 'x')} | "
             f"{mark('targetgen', format(t[key]['acceptance_length'], '.3f'))} | {t[key]['output_tokens_per_s']:,.0f} | {mark('targetgen', format(t[key]['output_tokens_per_s'] / pt, '.2f') + 'x')} |")
pp = pooled_tps(p)
L.append(f"| Pooled (1,024) | {pp:,.0f} | {mark('itw', format(i['all']['acceptance_length'], '.3f'))} | {pooled_tps(i):,.0f} | {mark('itw', format(pooled_tps(i) / pp, '.2f') + 'x')} | "
         f"{mark('targetgen', format(t['all']['acceptance_length'], '.3f'))} | {pooled_tps(t):,.0f} | {mark('targetgen', format(pooled_tps(t) / pp, '.2f') + 'x')} |")
L += ["", "Acceptance length here is measured over whole chained rollouts. It is lower than the trainer's teacher-forced `accept_len` because serving's rounds are weighted toward hard positions.", ""]
Path(a.out).write_text("\n".join(L))
print(f"wrote {a.out}")
