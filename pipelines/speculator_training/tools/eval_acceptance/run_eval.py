#!/usr/bin/env python
"""Acceptance length of a drafter on the target model's OWN generations (rollouts), from vLLM's speculative-decoding counters.

Send held-out prompts (build_prompts.py) to a vLLM server that serves the target WITH the draft, let the target generate, and read
the server's counters before and after each prompt set. Acceptance is computed by the speculators repo's own functions
(scripts/evaluate/perf_utils.py: `extract_spec_decode_metrics`), so the definition is the official one:

    acceptance_length = 1 + num_accepted_tokens / num_drafts        acceptance_at_pos_i = accepted at position i / num_drafts

Unlike the trainer's `eal` (teacher-forced on dataset text), the text being drafted here is sampled from the target model.

  python run_eval.py --target http://localhost:8000 --prompts-dir runs/eval_prompts_500 --output-dir OUT \
      [--wandb-name NAME] [--max-tokens 8192] [--temperature 0.6 --top-p 0.95 --top-k 20] [--concurrency 32]
"""
import argparse
import csv
import json
import sys
import threading
import time
import urllib.request
from collections import Counter
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# this file: <workspace>/spec-experiments/pipelines/speculator_training/tools/eval_acceptance/run_eval.py
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "speculators" / "scripts" / "evaluate"))
from perf_utils import acceptance_csv_columns, extract_spec_decode_metrics, fetch_metrics, parse_prometheus_metrics  # noqa: E402


def post_json(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def scrape(target: str):
    text = fetch_metrics(f"{target}/metrics")
    if text is None:
        sys.exit(f"cannot read {target}/metrics")
    return parse_prometheus_metrics(text)


def run_subset(name: str, rows: list[dict], args) -> tuple[dict, list[dict]]:
    baseline = scrape(args.target)
    results: list[dict | None] = [None] * len(rows)
    lock = threading.Lock()
    done = [0]

    def one(i: int):
        body = {"model": args.model, "prompt": rows[i]["prompt"], "max_tokens": args.max_tokens, "temperature": args.temperature,
                "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed + i, "skip_special_tokens": False}
        try:
            r = post_json(f"{args.target}/v1/completions", body, args.request_timeout)
            c = r["choices"][0]
            out = {"completion": c["text"], "finish_reason": c["finish_reason"],
                   "completion_tokens": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"]}
        except Exception as e:  # noqa: BLE001
            out = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        results[i] = out
        with lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(rows):
                print(f"  [{name}] {done[0]}/{len(rows)} requests done", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(one, range(len(rows))))
    wall = time.time() - t0
    current = scrape(args.target)

    spec = extract_spec_decode_metrics(current, baseline_metrics=baseline)
    ok = [r for r in results if r and "error" not in r]
    errors = [r["error"] for r in results if r and "error" in r]
    if errors and len(errors) > 0.02 * len(rows):
        sys.exit(f"[{name}] {len(errors)} of {len(rows)} requests failed, e.g. {errors[0]}")
    if spec["num_drafts"] <= 0:
        sys.exit(f"[{name}] the server reported no speculative-decoding drafts: it is not serving a draft "
                 f"(check --speculative-config and {args.target}/metrics)")
    toks = sum(r["completion_tokens"] for r in ok)
    spec.update({"subset": name, "requests": len(rows), "failed": len(errors), "completion_tokens": toks,
                 "completion_tokens_mean": toks / max(1, len(ok)), "wall_s": wall, "output_tokens_per_s": toks / wall,
                 "finish_reasons": dict(Counter(r["finish_reason"] for r in ok))})
    merged = [{k: v for k, v in rows[i].items() if k != "prompt"} | (results[i] or {}) for i in range(len(rows))]
    return spec, merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, help="server root, e.g. http://localhost:8000")
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--subsets", nargs="*", default=["first_turn", "later_turn"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B", help="served model name")
    ap.add_argument("--max-prompts", type=int, default=None, help="only the first N prompts of each subset (testing)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="request i uses seed+i, so the rollouts are repeatable")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--request-timeout", type=float, default=3600)
    ap.add_argument("--wandb-name", default=None, help="log the results to W&B under this run name (entity/project from WANDB_*)")
    ap.add_argument("--checkpoint", default="", help="draft checkpoint path, recorded in the results")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.load(open(Path(args.prompts_dir) / "manifest.json"))
    per_subset, pooled = [], {"num_drafts": 0.0, "num_draft_tokens": 0.0, "num_accepted_tokens": 0.0, "pos": []}
    for name in args.subsets:
        rows = [json.loads(line) for line in open(Path(args.prompts_dir) / f"{name}.jsonl", encoding="utf-8")]
        if args.max_prompts:
            rows = rows[: args.max_prompts]
        print(f"[{name}] {len(rows)} prompts", flush=True)
        spec, merged = run_subset(name, rows, args)
        with open(out / f"completions_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        per_subset.append(spec)
        print(f"[{name}] acceptance_length={spec['acceptance_length']:.3f} over {spec['num_drafts']:.0f} drafts; "
              f"{spec['completion_tokens_mean']:.0f} tokens/completion; finish {spec['finish_reasons']}", flush=True)
        pooled["num_drafts"] += spec["num_drafts"]
        pooled["num_draft_tokens"] += spec["num_draft_tokens"]
        pooled["num_accepted_tokens"] += spec["num_accepted_tokens"]
        counts, i = [], 0
        while f"acceptance_at_pos_{i}" in spec:  # accepted-at-position counts = rate * drafts
            counts.append(spec[f"acceptance_at_pos_{i}"] * spec["num_drafts"])
            i += 1
        pooled["pos"] = [a + b for a, b in zip_longest(pooled["pos"], counts, fillvalue=0.0)]
    if len(per_subset) > 1:
        d = pooled["num_drafts"]
        allrow = {"subset": "all", "num_drafts": d, "num_draft_tokens": pooled["num_draft_tokens"],
                  "num_accepted_tokens": pooled["num_accepted_tokens"], "acceptance_length": 1 + pooled["num_accepted_tokens"] / d}
        allrow |= {f"acceptance_at_pos_{i}": c / d for i, c in enumerate(pooled["pos"])}
        allrow |= {"requests": sum(s["requests"] for s in per_subset), "failed": sum(s["failed"] for s in per_subset)}
        per_subset.append(allrow)
        print(f"[all] pooled acceptance_length={allrow['acceptance_length']:.3f}", flush=True)

    cols = ["subset", "requests", "failed", *acceptance_csv_columns(per_subset[0])]
    with open(out / "acceptance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_subset)
    config = {"checkpoint": args.checkpoint, "model": args.model, "max_tokens": args.max_tokens, "temperature": args.temperature,
              "top_p": args.top_p, "top_k": args.top_k, "seed": args.seed, "concurrency": args.concurrency,
              "prompts_dir": str(args.prompts_dir), "prompts_manifest": manifest["files"]}
    json.dump({"config": config, "results": per_subset}, open(out / "results.json", "w"), indent=2, default=str)
    print(f"wrote {out}/acceptance.csv and results.json")

    if args.wandb_name:
        try:
            import wandb

            run = wandb.init(name=args.wandb_name, job_type="eval", config=config, dir=str(out))
            for s in per_subset:
                for k, v in s.items():
                    if isinstance(v, (int, float)) and k != "subset":
                        run.summary[f"eval/{s['subset']}/{k}"] = v
            run.finish()
            print(f"logged to W&B run {args.wandb_name}")
        except Exception as e:  # noqa: BLE001  never lose the results over logging
            print(f"WARNING: W&B logging failed ({type(e).__name__}: {str(e)[:120]}); results are in {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
