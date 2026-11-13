#!/usr/bin/env python
"""Replay prompts from published SWE-bench Verified agent trajectories (mini-swe-agent), for run_eval.py: the context an agent had before an assistant turn, rendered with the target's chat template.

Source: open-athena/swebench-context-compaction-ablation-2026-10-22 (traces/<model>/<arm>.jsonl.zst, one row per task, `attempts[0].trajectory.messages` in OpenAI chat format with tool calls).
Each selected turn is rendered with the target's chat template + mini-swe-agent 2.1.0's bash tool schema and CHECKED against the true prompt length the trajectory recorded
(`extra.response.usage.prompt_tokens`): a mismatch is dropped and counted. The reconstruction was exact (0 difference) on every turn checked while developing this.

  python build_snowball_swe_replay_prompts.py --out-dir <NEW dir> [--model snowball-v0.1] [--arm 131k-16k-no-summary] [--seed 0] [--workers 32]
Output: <out-dir>/ctx_<bucket>.jsonl (prompt = token ids, prompt_tokens, task, turn_message_index, true_completion_tokens, reward, bucket) + manifest.json. Buckets by true prompt tokens.
"""
import argparse, hashlib, json, random, subprocess, sys
from multiprocessing import Pool
from pathlib import Path

REPO = "open-athena/swebench-context-compaction-ablation-2026-10-22"
BASH_TOOL = {"type": "function", "function": {"name": "bash", "description": "Execute a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}}, "required": ["command"]}}}
BUCKETS = [("ctx_lt4k", 0, 4096, 300), ("ctx_4k_16k", 4096, 16384, 300), ("ctx_16k_32k", 16384, 32768, 300), ("ctx_32k_64k", 32768, 65536, 300), ("ctx_64k_128k", 65536, 131072, 200)]
PER_TASK_PER_BUCKET = 3
TARGET = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom"
_tok = None; ROWS = {}


def clean(m):
    o = {"role": m["role"], "content": m["content"]}
    if m["role"] == "assistant" and m.get("tool_calls"):
        o["tool_calls"] = [{"id": t["id"], "type": "function", "function": {"name": t["function"]["name"], "arguments": json.loads(t["function"]["arguments"])}} for t in m["tool_calls"]]
    if m["role"] == "tool": o["tool_call_id"] = m["tool_call_id"]
    return o


def render(job):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    task, i, true_prompt = job
    try:
        ids = _tok.apply_chat_template([clean(m) for m in ROWS[task][:i]], tools=[BASH_TOOL], add_generation_prompt=True, tokenize=True)
        ids = ids["input_ids"] if hasattr(ids, "input_ids") else ids
    except Exception as e:  # noqa: BLE001
        return job, None, f"{type(e).__name__}: {str(e)[:80]}"
    return job, list(ids), ("ok" if len(ids) == true_prompt else f"length mismatch: rendered {len(ids)} vs true {true_prompt}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", required=True); ap.add_argument("--model", default="snowball-v0.1"); ap.add_argument("--arm", default="131k-16k-no-summary")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=32); a = ap.parse_args()
    out = Path(a.out_dir)
    if (out / "manifest.json").exists(): sys.exit(f"{out} already holds prompts: old outputs are never overwritten")
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, f"traces/{a.model}/{a.arm}.jsonl.zst", repo_type="dataset"); print("source:", p, flush=True)
    cands, rewards = [], {}
    for line in subprocess.run(["zstdcat", p], capture_output=True, check=True).stdout.decode().splitlines():
        r = json.loads(line); at = r["attempts"][0] if r.get("attempts") else None
        if not at or not isinstance(at.get("trajectory"), dict): continue
        msgs = at["trajectory"]["messages"]; ROWS[r["task_name"]] = msgs; rewards[r["task_name"]] = r.get("reward")
        for i, m in enumerate(msgs):
            u = ((m.get("extra") or {}).get("response") or {}).get("usage") if m["role"] == "assistant" else None
            if u and u.get("prompt_tokens"): cands.append((r["task_name"], i, u["prompt_tokens"], u.get("completion_tokens")))
    print(f"{len(ROWS)} tasks, {len(cands):,} assistant turns with recorded usage", flush=True)
    rng = random.Random(a.seed); chosen = []
    for name, lo, hi, n in BUCKETS:
        pool = [c for c in cands if lo <= c[2] < hi]; rng.shuffle(pool); per, picked = {}, []
        for c in pool:
            if per.get(c[0], 0) >= PER_TASK_PER_BUCKET: continue
            per[c[0]] = per.get(c[0], 0) + 1; picked.append((name,) + c)
            if len(picked) == n * 12 // 10: break          # 20% spare for turns dropped by the length check
        chosen += picked; print(f"  {name}: {len(pool):,} candidate turns, {len(picked)} picked (target {n})", flush=True)
    jobs = [(c[1], c[2], c[3]) for c in chosen]; meta = {(c[1], c[2]): c for c in chosen}
    done, bad = {}, {}
    with Pool(a.workers) as pool:
        for k, (job, ids, status) in enumerate(pool.imap_unordered(render, jobs, chunksize=4)):
            if ids is not None and status == "ok": done[(job[0], job[1])] = ids
            else: bad[status[:60]] = bad.get(status[:60], 0) + 1
            if k % 200 == 0: print(f"  rendered {k + 1}/{len(jobs)}", flush=True)
    out.mkdir(parents=True, exist_ok=True); files, counts = {}, {}
    for name, lo, hi, n in BUCKETS:
        rows = []
        for (task, i), ids in sorted(done.items()):
            c = meta[(task, i)]
            if c[0] == name: rows.append({"prompt": ids, "prompt_tokens": len(ids), "task": task, "turn_message_index": i, "true_completion_tokens": c[4], "reward": rewards[task], "bucket": name})
        random.Random(a.seed + 1).shuffle(rows); rows = rows[:n]
        path = out / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        files[f"{name}.jsonl"] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}; counts[name] = len(rows)
    json.dump({"files": files, "source": f"{REPO}/traces/{a.model}/{a.arm}.jsonl.zst", "rendering": "target chat template + mini-swe-agent 2.1.0 bash tool; each prompt's length equals the recorded usage.prompt_tokens",
               "dropped": bad, "seed": a.seed, "per_task_per_bucket_cap": PER_TASK_PER_BUCKET, "target_model": TARGET}, open(out / "manifest.json", "w"), indent=1)
    print("rows per bucket:", counts, "| dropped:", bad or "none")


if __name__ == "__main__":
    main()
