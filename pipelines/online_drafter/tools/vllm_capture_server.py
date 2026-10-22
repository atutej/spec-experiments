#!/usr/bin/env python
"""A vLLM server for Aurora's online drafter training (Aurora's "external" mode): serves the target with EAGLE-3 speculative decoding, and for every rollout batch
captures the target's hidden states, publishes them to Mooncake in Aurora's layout, and tells Aurora's trainer (POST <push_url>/push_sample, the same contract
Aurora's patched SGLang uses). Aurora's controller reloads the draft with POST /update_weights_from_disk {"model_path": ..., "update_draft_model": true}.

Run in the `vllm` env (the Marin vLLM fork, unmodified); needs this tools/ dir on PYTHONPATH so vLLM can import capture_ext.
  POST /rollout   {prompts: [{"token_ids": [...]} | {"text": "..."}], max_tokens, temperature, top_p, seed, data_id_prefix, push_url, mooncake: {...}}
  POST /update_weights_from_disk   {"model_path": dir, "update_draft_model": true}
  GET  /health
One batch at a time (a lock), like Aurora's engine calls. Prefix caching is off: after a draft update, cached draft KV would be stale.
"""
import argparse
import json
import threading
import time

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--target", required=True)
p.add_argument("--draft", required=True, help="initial draft checkpoint dir (HF format: an EAGLE-3 drafter, or a DSpark one with --method dspark)")
p.add_argument("--method", default="eagle3", choices=["eagle3", "dspark"], help="the drafter's speculative method")
p.add_argument("--num-spec-tokens", type=int, default=3, help="eagle3: draft length; dspark: the block size of the drafter (8)")
p.add_argument("--port", type=int, default=30000)
p.add_argument("--host", default="0.0.0.0")
p.add_argument("--max-model-len", type=int, default=8192)
p.add_argument("--gpu-memory-utilization", type=float, default=0.4)
p.add_argument("--tensor-parallel-size", type=int, default=1)
p.add_argument("--data-parallel-size", type=int, default=1, help="capture needs TP=1; MoE models use DP with --enable-expert-parallel")
p.add_argument("--enable-expert-parallel", action="store_true")
p.add_argument("--enforce-eager", action="store_true")
p.add_argument("--max-window-tokens", type=int, default=0, help="0: the model length")
p.add_argument("--capture-budget-tokens", type=int, default=50_000_000, help="token budget of one capture interval (a rollout batch)")
p.add_argument("--scratch-dir", default="")
args = p.parse_args()

llm = LLM(model=args.target, speculative_config={"method": args.method, "model": args.draft, "num_speculative_tokens": args.num_spec_tokens,
                                **({"draft_sample_method": "probabilistic", "rejection_sample_method": "standard"} if args.method == "dspark" else {})},
          async_scheduling=False, enable_prefix_caching=False, disable_log_stats=False, enforce_eager=args.enforce_eager, dtype="bfloat16",
          gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, tensor_parallel_size=args.tensor_parallel_size,
          data_parallel_size=args.data_parallel_size, enable_expert_parallel=args.enable_expert_parallel,
          worker_extension_cls="capture_ext.AuroraCaptureExt")
lock = threading.Lock()
state = {"rollouts": 0, "draft_updates": 0, "started": time.time()}
app = FastAPI()


class Rollout(BaseModel):
    prompts: list[dict]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    data_id_prefix: str = "rollout"
    push_url: str | None = None
    mooncake: dict | None = None
    capture: bool = True            # False: plain serving with the current draft (the static baseline), nothing captured or published
    capture_revision: str = "online"


def _spec_counters() -> dict[str, float]:
    out = {"drafts": 0.0, "accepted": 0.0, "draft_tokens": 0.0}
    names = {"vllm:spec_decode_num_drafts": "drafts", "vllm:spec_decode_num_accepted_tokens": "accepted", "vllm:spec_decode_num_draft_tokens": "draft_tokens"}
    for m in llm.get_metrics():
        if m.name in names and hasattr(m, "value"):
            out[names[m.name]] += float(m.value)
    return out


@app.get("/health")
def health():
    return {"status": "ok", **state, "uptime_s": round(time.time() - state["started"], 1)}


@app.get("/draft_checksum")
def draft_checksum():
    with lock:
        return llm.collective_rpc("ac_draft_checksum")[0]


@app.post("/rollout")
def rollout(r: Rollout):
    with lock:
        t0 = time.time()
        cfg = {"step": state["rollouts"], "max_tokens": args.capture_budget_tokens, "target_revision": args.target, "draft_revision": r.capture_revision,
               "capture_target_snapshot": False}
        if args.max_window_tokens:
            cfg["max_window_tokens"] = args.max_window_tokens
        if r.capture:
            llm.collective_rpc("ac_begin", args=(cfg,))
        prompts = [{"prompt_token_ids": x["token_ids"]} if "token_ids" in x else x["text"] for x in r.prompts]
        sp = SamplingParams(max_tokens=r.max_tokens, temperature=r.temperature, top_p=r.top_p, seed=r.seed)
        c0 = _spec_counters()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        c1 = _spec_counters()
        t_gen = time.time() - t0
        keys = {o.request_id: f"{r.data_id_prefix}_{i}" for i, o in enumerate(outs)}
        spec = {"mooncake": r.mooncake, "keys": keys, "scratch_dir": args.scratch_dir or None}
        results = llm.collective_rpc("ac_seal_publish", args=(spec,)) if r.capture else [{"windows": []}]
        windows = [w for res in results for w in res["windows"]]
        by_key = {w["mooncake_key"]: w for w in windows}
        pushed = 0
        if r.push_url:
            for i, o in enumerate(outs):
                w = by_key.get(f"{r.data_id_prefix}_{i}")
                if w is None:
                    continue
                resp = requests.post(f"{r.push_url}/push_sample", json={
                    "data_id": f"{r.data_id_prefix}_{i}", "mooncake_key": w["mooncake_key"], "tensor_shapes": w["tensor_shapes"],
                    "prompt_tokens": w["prompt_tokens"], "completion_tokens": w["completion_tokens"]}, timeout=60)
                resp.raise_for_status()
                pushed += 1
        state["rollouts"] += 1
        drafts, accepted = c1["drafts"] - c0["drafts"], c1["accepted"] - c0["accepted"]
        gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
        return {"requests": len(outs), "windows": len(windows), "pushed": pushed, "generated_tokens": gen_tokens,
                "spec_drafts": drafts, "spec_accepted": accepted, "accept_length": (1 + accepted / drafts) if drafts else None,
                "gen_seconds": round(t_gen, 2), "total_seconds": round(time.time() - t0, 2), "gen_tokens_per_s": round(gen_tokens / t_gen, 1),
                "dropped_requests": [res.get("dropped_requests") for res in results], "dropped_windows": [res.get("dropped_windows") for res in results],
                "sample_keys": [w["mooncake_key"] for w in windows[:3]]}


class Update(BaseModel):
    model_path: str
    update_draft_model: bool = True


@app.post("/update_weights_from_disk")
def update_weights_from_disk(u: Update):
    if not u.update_draft_model:
        raise HTTPException(400, "only draft updates are supported")
    with lock:
        res = llm.collective_rpc("ac_reload_draft", args=(u.model_path,))
        state["draft_updates"] += 1
    return {"success": True, "message": json.dumps(res[0]) if res else "", "updates": state["draft_updates"]}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
