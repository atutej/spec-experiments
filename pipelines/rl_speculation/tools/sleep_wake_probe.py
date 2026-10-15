#!/usr/bin/env python
"""Does a speculative-decoding draft survive vLLM's sleep / wake cycle?

Why: in the colocated RL run (2026-10-14) the engines sleep during training (MarinSkyRL default sleep level 2, which discards weights) and the policy-to-engine
weight sync refills only the TARGET model. The DSpark draft is separate weights inside the engine; the engine's own log then showed accept length 1.000
(57 accepted of 465,002 drafts), against 2.87 in the split layout, where the engine never sleeps. This probe isolates that, on one GPU, outside RL.

Follow-up (2026-10-14, after the colocated RL run with sleep level 1 still showed accept length 1.0): the first three variants changed the sleep level AND the wake style at once
(sleep1 woke everything at once; sleep2 woke by tag, "weights" then "kv_cache", as RL does, and reloaded the target). The four extra variants below change one thing at a time.

Variants (one process, one GPU each):
  none    generate, generate again                                         (control: accept length must not drift)
  sleep1  generate, sleep(level=1), wake_up(), generate                    (weights go to host memory and come back)
  sleep2  generate, sleep(level=2), wake_up(weights), reload_weights(),   (what the weight sync does: refill the target only)
          wake_up(kv_cache), generate

Accept length = 1 + accepted / drafts from vLLM's counters (deltas per phase). Also logs GPU memory while asleep and a sanity sample of the text.

  python sleep_wake_probe.py --variant sleep2 --model <Grug> --draft <drafter dir> --prompts <first_turn.jsonl> --out result.json
"""
import argparse
import json
import subprocess
import time


def counters(llm):
    drafts = accepted = 0
    per_pos = None
    metrics = llm.get_metrics()
    if not any(m.name.startswith("vllm:spec_decode") for m in metrics) and not getattr(counters, "warned", False):
        counters.warned = True
        print("no vllm:spec_decode metrics; names seen:", sorted({m.name for m in metrics})[:40], flush=True)
    for m in metrics:
        if m.name == "vllm:spec_decode_num_drafts":
            drafts += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            accepted += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            vals = list(m.values)
            per_pos = vals if per_pos is None else [a + b for a, b in zip(per_pos, vals)]
    return drafts, accepted, per_pos


def gpu_used_mib(phys_gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(phys_gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def phase(llm, prompts, params, name, phys_gpu):
    d0, a0, p0 = counters(llm)
    t0 = time.time()
    outs = llm.generate(prompts, params)
    wall = time.time() - t0
    d1, a1, p1 = counters(llm)
    drafts, accepted = d1 - d0, a1 - a0
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    res = {"phase": name, "drafts": drafts, "accepted": accepted, "accept_length": (1 + accepted / drafts) if drafts else None,
           "tokens": toks, "wall_s": round(wall, 2), "tok_per_s": round(toks / wall, 1), "gpu_used_mib": gpu_used_mib(phys_gpu),
           "sample": outs[0].outputs[0].text[:240]}
    if p1 is not None and p0 is not None and drafts:
        res["accept_at_pos"] = [round((b - a) / drafts, 3) for a, b in zip(p0, p1)]
    print(json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["none", "sleep1", "sleep2", "sleep1_tags", "sleep1_reload", "sleep2_full", "sleep1_tags_reload"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--phys-gpu", type=int, default=0, help="physical GPU index, for nvidia-smi")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)][: args.num_prompts]
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=0)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len, gpu_memory_utilization=0.9, enable_sleep_mode=True, disable_log_stats=False,
              speculative_config={"method": "dspark", "model": args.draft, "num_speculative_tokens": 8})
    results = [phase(llm, prompts, params, "before", args.phys_gpu)]
    if args.variant == "none":
        results.append(phase(llm, prompts, params, "again", args.phys_gpu))
    elif args.variant == "sleep1":
        llm.sleep(level=1)
        results.append({"phase": "asleep", "gpu_used_mib": gpu_used_mib(args.phys_gpu)})
        print(json.dumps(results[-1]), flush=True)
        llm.wake_up()
        results.append(phase(llm, prompts, params, "after_sleep1", args.phys_gpu))
    elif args.variant in ("sleep1_tags", "sleep1_reload", "sleep2_full", "sleep1_tags_reload"):
        # sleep1_tags:        level 1, wake by tag (weights, kv_cache), NO weight update       -> is the wake style the problem?
        # sleep1_reload:      level 1, wake everything at once, reload the target             -> does a target weight update disturb the draft?
        # sleep2_full:        level 2, wake everything at once, reload the target             -> is level 2 itself the problem?
        # sleep1_tags_reload: level 1, wake by tag, reload the target                         -> the colocated RL flow with level 1
        level = 2 if args.variant == "sleep2_full" else 1
        by_tag = args.variant in ("sleep1_tags", "sleep1_tags_reload")
        reload_target = args.variant != "sleep1_tags"
        llm.sleep(level=level)
        results.append({"phase": "asleep", "gpu_used_mib": gpu_used_mib(args.phys_gpu)})
        print(json.dumps(results[-1]), flush=True)
        if by_tag:
            llm.wake_up(tags=["weights"])
            if reload_target:
                llm.collective_rpc("reload_weights")
            llm.wake_up(tags=["kv_cache"])
        else:
            llm.wake_up()
            if reload_target:
                llm.collective_rpc("reload_weights")
        results.append(phase(llm, prompts, params, "after_" + args.variant, args.phys_gpu))
    else:
        llm.sleep(level=2)
        results.append({"phase": "asleep", "gpu_used_mib": gpu_used_mib(args.phys_gpu)})
        print(json.dumps(results[-1]), flush=True)
        llm.wake_up(tags=["weights"])
        llm.collective_rpc("reload_weights")   # refills the target model only, like the policy-to-engine weight sync
        llm.wake_up(tags=["kv_cache"])
        results.append(phase(llm, prompts, params, "after_sleep2_reload_target", args.phys_gpu))
    json.dump({"variant": args.variant, "model": args.model, "draft": args.draft, "results": results}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
