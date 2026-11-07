#!/usr/bin/env python
"""Rebuild the exact prompts a finished RL job served, from its Harbor trajectories, as a run_eval.py prompts dir (subsets by context length).

  python build_replay_prompts.py <trace_jobs dir> <out dir> [--per-bin 128] [--seed 0]

Each agent step in trace_jobs/*/attempts/*/agent/trajectory.json carries metrics.prompt_token_ids: the token ids vLLM received for that turn. They are written as the
"prompt" field (a list of ids, which /v1/completions accepts), so the replay sees the identical context. The target re-samples the reply (T=1), as in the RL run; the
recorded completion is not replayed. Steps are drawn at random, at most 2 per trial, per context-length bin; turn 0 of each trial is also kept as its own subset."""
import argparse, glob, hashlib, json, os, random

BINS = [("ctx_0_4k", 0, 4096), ("ctx_4_8k", 4096, 8192), ("ctx_8_16k", 8192, 16384), ("ctx_16_24k", 16384, 24577)]
ap = argparse.ArgumentParser()
ap.add_argument("trace_jobs"); ap.add_argument("out")
ap.add_argument("--per-bin", type=int, default=128); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(f"{a.out} exists; pick a fresh dir")
rng = random.Random(a.seed)
steps = []   # (trial, turn index, ids, recorded completion tokens)
for f in sorted(glob.glob(os.path.join(a.trace_jobs, "*", "attempts", "*", "agent", "trajectory.json"))):
    trial = f.split(os.sep)[-5]
    agent_steps = [s for s in json.load(open(f))["steps"] if s.get("source") == "agent" and (s.get("metrics") or {}).get("prompt_token_ids")]
    for turn, s in enumerate(agent_steps):
        steps.append((trial, turn, s["metrics"]["prompt_token_ids"], s["metrics"].get("completion_tokens")))
print(len(steps), "agent steps from", len({s[0] for s in steps}), "trials")
os.makedirs(a.out)
files = {}
def write(name, picked):
    with open(os.path.join(a.out, f"{name}.jsonl"), "w") as fh:
        for trial, turn, ids, ct in picked:
            fh.write(json.dumps({"prompt": ids, "trial": trial, "turn": turn, "ctx_tokens": len(ids), "recorded_completion_tokens": ct}) + "\n")
    lens = sorted(len(p[2]) for p in picked)
    files[name] = {"prompts": len(picked), "sha256": hashlib.sha256(open(os.path.join(a.out, f"{name}.jsonl"), "rb").read()).hexdigest(),
                   "prompt_tokens_median": lens[len(lens) // 2], "prompt_tokens_max": lens[-1]}
for name, lo, hi in BINS:
    pool = [s for s in steps if lo <= len(s[2]) < hi]
    rng.shuffle(pool)
    picked, per_trial = [], {}
    for s in pool:
        if per_trial.get(s[0], 0) < 2:
            picked.append(s); per_trial[s[0]] = per_trial.get(s[0], 0) + 1
        if len(picked) == a.per_bin:
            break
    write(name, picked)
first = [s for s in steps if s[1] == 0]
rng.shuffle(first)
write("turn0", first[: a.per_bin])
json.dump({"source": a.trace_jobs, "seed": a.seed, "files": files}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print(json.dumps(files, indent=1))
