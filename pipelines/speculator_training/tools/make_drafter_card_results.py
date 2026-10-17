"""Markdown results tables for the drafter model cards, built from the saved serving-eval results (docs/results/dspark_eval/*/{plain,itw_80k,targetgen_final}/results.json).

  python make_drafter_card_results.py --results-dir docs/results/dspark_eval --itw-repo atutej/<name> --targetgen-repo atutej/<name> --this itw --out card_results_itw.md

Acceptance length is vLLM's `1 + accepted/drafts` over whole rollouts; the throughput gain is output tokens/s divided by plain decoding on the same prompts.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--results-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--itw-repo", default=""); ap.add_argument("--targetgen-repo", default="")
ap.add_argument("--this", choices=["itw", "targetgen"], required=True, help="which drafter this card is for")
ap.add_argument("--gh-ref", default="<GH-COMMIT-AFTER-PUSH>", help="commit SHA of atutej/spec-experiments the links point at")
a = ap.parse_args()
R = Path(a.results_dir)
GB = f"https://github.com/atutej/spec-experiments/blob/{a.gh_ref}"
GT = f"https://github.com/atutej/spec-experiments/tree/{a.gh_ref}"
EV = "pipelines/speculator_training/tools/eval_acceptance"
RES = "docs/results/dspark_eval"


def lk(name: str, path: str, tree: bool = False) -> str:
    return f"[`{name}`]({(GT if tree else GB)}/{path})"


def load(setdir: str, arm: str) -> dict:
    rows = json.loads((R / setdir / arm / "results.json").read_text())["results"]
    return {r["subset"]: r for r in rows}


def pooled_tps(d: dict) -> float:
    rs = [d[k] for k in ("first_turn", "later_turn") if k in d]
    return sum(r["completion_tokens"] for r in rs) / sum(r["wall_s"] for r in rs)


ARM = {"itw": ("itw_80k", "itw drafter, step 80,813"), "targetgen": ("targetgen_final", "targetgen drafter, step 17,737")}[a.this]
arm_dir, arm_label = ARM
L = ["## Results (target: open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21)", "",
     "Serving in the pinned Marin vLLM fork on one 4-GPU GB200 node (TP=1, DP=4, expert parallel), concurrency 16, T=1, 8 drafted tokens, probabilistic drafting with standard verification; "
     "one run per arm. Gain is output tokens/s divided by plain decoding on the same prompts.", ""]

# (a) RL prompts
plain, me = (load("rl_prompts_256_target-09.21", x)["first_turn"] for x in ("plain", arm_dir))
L += ["### RL prompts (Snowball broad-domain RLVR training set, first 256 of a random 1,024, thinking on, up to 49,152 new tokens)", "",
      "| Arm | Acceptance length | Output tokens/s | Gain vs plain |", "|---|---|---|---|",
      f"| Plain decoding | n/a | {plain['output_tokens_per_s']:,.0f} | 1.00x |",
      f"| {arm_label} | {me['acceptance_length']:.3f} | {me['output_tokens_per_s']:,.0f} | {me['output_tokens_per_s'] / plain['output_tokens_per_s']:.2f}x |", "",
      "**Prompts.** 1,024 prompts sampled at random (seed 0) from the train split of "
      "[`open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30`](https://huggingface.co/datasets/open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30) "
      "(84,403 rows; the corrected RLVR1 set of the Snowball broad-domain RL runs, see [`CLEANED_DATASET.md`](https://huggingface.co/datasets/open-athena/snowball-broad-domain-rlvr-campaign-2026/blob/main/audits/CLEANED_DATASET.md)), "
      "rendered as the RL requests carry them (stored messages including earlier tool calls, each row's tool schemas as `tools=`, the chat template's default `Reasoning: /think`) by "
      f"{lk('build_nemotron_rlvr_prompts.py', EV + '/build_nemotron_rlvr_prompts.py')}. The prompt file's sha256 and the sampling settings are in "
      f"{lk('nemotron_rlvr_prompts_1024.json', RES + '/prompt_manifests/nemotron_rlvr_prompts_1024.json')}. The first 256 prompts are used.", "",
      f"**Evaluation.** {lk('serve_eval.sh', EV + '/serve_eval.sh')} starts one vLLM server per arm and {lk('run_eval.py', EV + '/run_eval.py')} sends the prompts and reads vLLM's speculative-decoding counters. "
      "Commands (from the `tools/eval_acceptance` folder of the repo; `<drafter>` is this repo's id or a local copy):", "",
      "```",
      "python build_nemotron_rlvr_prompts.py --output-dir <prompts> --num 1024 --seed 0",
      "PROMPTS_DIR=<prompts> MAX_PROMPTS=256 MAX_TOKENS=49152 MAX_MODEL_LEN=102400 OUT=<out> PLAIN=1 \\",
      "  ARMS=\"drafter=<drafter>\" SPEC_EXTRA=', \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"standard\"' bash serve_eval.sh",
      "```", "",
      f"`PLAIN=1` adds the no-draft baseline. Raw results: {lk('rl_prompts_256_target-09.21/', RES + '/rl_prompts_256_target-09.21', tree=True)}.", ""]

# (a2) the same RL prompts on the Dr. Doom target (a fine-tune of the 09.21 target; the drafter was trained on 09.21's hidden states)
if (R / "rl_prompts_256_target-dr-doom").exists():
    dp, dm = (load("rl_prompts_256_target-dr-doom", x)["first_turn"] for x in ("plain", arm_dir))
    L += ["### RL prompts on the Dr. Doom target ([`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom), a de-looping fine-tune of the target above; same 256 prompts and settings)", "",
          "| Arm | Acceptance length | Output tokens/s | Gain vs plain |", "|---|---|---|---|",
          f"| Plain decoding | n/a | {dp['output_tokens_per_s']:,.0f} | 1.00x |",
          f"| {arm_label} | {dm['acceptance_length']:.3f} | {dm['output_tokens_per_s']:,.0f} | {dm['output_tokens_per_s'] / dp['output_tokens_per_s']:.2f}x |",
          "", f"The drafter was trained on the 09.21 target's hidden states, so this is transfer to a fine-tuned target: acceptance falls by {me['acceptance_length'] - dm['acceptance_length']:.2f} relative to the table above. Same prompts and commands, with `MODEL=open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom` set for `serve_eval.sh` (one-node job: {lk('submit_eval.sh', EV + '/submit_eval.sh')}); raw results: {lk('rl_prompts_256_target-dr-doom/', RES + '/rl_prompts_256_target-dr-doom', tree=True)}.", ""]

# (b) Datakit validation, multi-turn
p_, m_ = (load("datakit_val_multiturn_1024", x) for x in ("plain", arm_dir))
L += ["### Datakit validation prompts, multi-turn (1,024 held-out rows, random assistant turn with earlier turns as context, up to 4,096 new tokens)", "",
      "| Subset | Plain tok/s | Acceptance length | Output tokens/s | Gain vs plain |", "|---|---|---|---|---|"]
for key, label, n in (("first_turn", "First turn", 833), ("later_turn", "Later turn", 191)):
    pt = p_[key]["output_tokens_per_s"]
    L.append(f"| {label} ({n}) | {pt:,.0f} | {m_[key]['acceptance_length']:.3f} | {m_[key]['output_tokens_per_s']:,.0f} | {m_[key]['output_tokens_per_s'] / pt:.2f}x |")
pp = pooled_tps(p_)
L.append(f"| Pooled (1,024) | {pp:,.0f} | {m_['all']['acceptance_length']:.3f} | {pooled_tps(m_):,.0f} | {pooled_tps(m_) / pp:.2f}x |")
L += ["", "**Prompts.** 1,024 prompts from the held-out validation rows of the Datakit dataset described by `training_data_manifest.json` (the last 0.1% of its rows, which this drafter did not train on): for each sampled row, a uniformly random assistant turn with the dataset's earlier turns as context (833 first-turn and 191 later-turn prompts, up to 28,000 prompt tokens). Built by "
      f"{lk('build_val_prompts.py', EV + '/build_val_prompts.py')} (seed 0; prompt-file hashes in {lk('datakit_val_multiturn_1024.json', RES + '/prompt_manifests/datakit_val_multiturn_1024.json')}). "
      "The dataset directory is rebuilt with the `data/` scripts linked under Reproduce; it is not published.", "",
      "```",
      "python build_val_prompts.py --dataset <Datakit dataset dir> --output-dir <prompts> --num 1024 --turns random --max-prompt-tokens 28000",
      "PROMPTS_DIR=<prompts> SUBSETS='first_turn later_turn' OUT=<out> PLAIN=1 \\",
      "  ARMS=\"drafter=<drafter>\" SPEC_EXTRA=', \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"standard\"' bash serve_eval.sh",
      "```", "", f"Replies are capped at 4,096 tokens (the default). Raw results: {lk('datakit_val_multiturn_1024/', RES + '/datakit_val_multiturn_1024', tree=True)}.",
      "", "Acceptance length is `1 + accepted/drafts` from vLLM's speculative-decoding counters, over whole chained rollouts. It is lower than the trainer's teacher-forced `accept_len` because serving's rounds are weighted toward hard positions.", ""]
Path(a.out).write_text("\n".join(L))
print(f"wrote {a.out}")
