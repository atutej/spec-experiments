#!/usr/bin/env python
"""Feeds prompts to the capture server (POST /rollout) in batches until a stop file appears or the trainer's callback server stops answering; logs one JSON line per
batch (accept length, tokens/s, samples pushed). This is the prompt source of the online loop. One file serves the client and Aurora's trainer, like Aurora's examples;
each row is one of:
  {"conversations": [{"role"|"from", "content"|"value"}, ...]}   rendered here with the target's chat template (--tokenizer): every message through the LAST user turn,
                                                                  add_generation_prompt, like Aurora's examples/send_user_requests.py (+ the server's chat endpoint)
  {"text": "..."}                                                  a finished prompt, sent as is
  {"token_ids": [...]}                                             sent as is (a prompt that already ends where the target should start)
  {"input_ids": [...], "loss_mask": [...] | "labels": [...]}       a full tokenised row (Datakit; Aurora's own `pretokenized` format uses "labels" with -100): cut at the start
                                                                  of the LAST supervised span, everything before it is the prompt (no template involved)
Swap the file (and --tokenizer) to change dataset or model."""
import argparse, itertools, json, os, socket, time
import requests

p = argparse.ArgumentParser()
p.add_argument("--server", default="http://127.0.0.1:30000")
p.add_argument("--callback", default="", help="Aurora's training callback server (push_url); polled for /health. Empty: static serving, nothing captured")
p.add_argument("--prompts", required=True, help="jsonl (one row per prompt, see above) or a single .arrow shard of tokenised rows")
p.add_argument("--tokenizer", default="", help="the target model (HF id or dir): needed for 'conversations' rows (its chat template)")
p.add_argument("--chat-template-kwargs", default="{}", help="JSON passed to apply_chat_template, e.g. '{\"enable_thinking\": true}' for Qwen3")
p.add_argument("--max-prompt-tokens", type=int, default=0, help="skip conversations whose rendered prompt is longer (the server's context is limited); 0: no filter")
p.add_argument("--limit", type=int, default=0, help="use only the first N prompts (0: all)")
p.add_argument("--dump-rendered", default="", help="write the rendered prompts to this jsonl and exit (no server needed): the check that rendering is unchanged")
p.add_argument("--batch", type=int, default=32)
p.add_argument("--max-tokens", type=int, default=256)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--mooncake-master", default="")
p.add_argument("--mooncake-metadata", default="")
p.add_argument("--local-hostname", default="")
p.add_argument("--mooncake-segment-gb", type=int, default=16, help="memory the server contributes to the Mooncake pool: the pool holds every sample the trainer has not fetched yet, and Mooncake EVICTS the oldest when it is full")
p.add_argument("--max-pool", type=int, default=96, help="wait before a batch while the trainer's sample pool holds more than this many unfetched samples (0: never wait). The trainer cannot slow the server "
               "down in external mode, so without this a fast server (or a slow first training step) overruns the Mooncake pool and the oldest samples vanish before they are fetched")
p.add_argument("--stop-file", default="")
p.add_argument("--log", default="")
p.add_argument("--max-batches", type=int, default=0)
p.add_argument("--loop", action="store_true", help="cycle through the prompts again at the end (default: stop after one pass, like send_user_requests.py without --loop)")
p.add_argument("--wandb-project", default="", help="log every batch's accept length, tokens/s, ... to this W&B project (empty: off); the run is named <group>-client in --wandb-group")
p.add_argument("--wandb-entity", default="")
p.add_argument("--wandb-group", default="", help="the trainer's group, so both runs sit together")
p.add_argument("--wandb-dir", default="")
p.add_argument("--steps-per-sync", type=int, default=0, help="the trainer's decode.weight_sync_interval: logs trainer_step = draft updates x this, to line the curve up with the trainer's step axis")
a = p.parse_args()

ROLE = {"human": "user", "gpt": "assistant"}
_tok = None


def render(row):
    """One prompt dict for the server: {"text": ...} or {"token_ids": [...]}; None for a conversation that cannot be a prompt."""
    global _tok
    if "token_ids" in row:
        return {"token_ids": row["token_ids"]}
    if "input_ids" in row:   # a full tokenised row: cut at the start of its LAST supervised (assistant) span, like the conversation rule
        mask = row["loss_mask"] if "loss_mask" in row else [lab != -100 for lab in row["labels"]]
        start = None
        for j, m in enumerate(mask):
            if m and (j == 0 or not mask[j - 1]):
                start = j
        if not start or (a.max_prompt_tokens and start > a.max_prompt_tokens):
            return None   # no assistant span, nothing before it, or too long
        return {"token_ids": row["input_ids"][:start]}
    if "text" in row:
        return {"text": row["text"]}
    conv = row["conversations"]
    if conv and "from" in conv[0]:
        conv = [{"role": ROLE.get(m["from"], m["from"]), "content": m["value"]} for m in conv]
    last_user = max((i for i, m in enumerate(conv) if m["role"] == "user"), default=-1)
    if last_user < 0 or conv[0]["role"] != "user":
        return None
    if _tok is None:
        from transformers import AutoTokenizer
        assert a.tokenizer, "--tokenizer is required for 'conversations' rows"
        _tok = AutoTokenizer.from_pretrained(a.tokenizer)
    text = _tok.apply_chat_template(conv[: last_user + 1], tokenize=False, add_generation_prompt=True, **json.loads(a.chat_template_kwargs))
    if a.max_prompt_tokens and len(_tok(text, add_special_tokens=False)["input_ids"]) > a.max_prompt_tokens:
        return None
    return {"text": text}


def rows():
    """The rows of --prompts: a jsonl file, or one Arrow shard (e.g. a Datakit data-XXXXX-of-YYYYY.arrow), read lazily."""
    if a.prompts.endswith(".arrow"):
        from datasets import Dataset
        yield from Dataset.from_file(a.prompts)
    else:
        for line in open(a.prompts):
            yield json.loads(line)


prompts, skipped = [], 0
for row in rows():
    r = render(row)
    if r is None:
        skipped += 1
    else:
        prompts.append(r)
    if a.limit and len(prompts) >= a.limit:
        break
print(f"{len(prompts)} prompts ready ({skipped} conversations skipped: no user turn, not starting with one, or longer than --max-prompt-tokens)", flush=True)
if a.dump_rendered:
    with open(a.dump_rendered, "w") as f:
        for r in prompts:
            f.write(json.dumps(r) + "\n")
    raise SystemExit(0)
mc = {"master_server_address": a.mooncake_master, "metadata_server": a.mooncake_metadata, "protocol": "tcp",
      "local_hostname": a.local_hostname or socket.gethostbyname(socket.gethostname()), "global_segment_size": a.mooncake_segment_gb * 1024**3, "local_buffer_size": 1024**3}
cyc = itertools.cycle(range(0, len(prompts) - a.batch + 1, a.batch)) if a.loop else iter(range(0, len(prompts) - a.batch + 1, a.batch))
n = 0
wb = None
if a.wandb_project:
    import wandb
    if a.wandb_dir:
        os.makedirs(a.wandb_dir, exist_ok=True)
    wb = wandb.init(entity=a.wandb_entity or None, project=a.wandb_project, group=a.wandb_group or None, name=f"{a.wandb_group or 'rollouts'}-client", dir=a.wandb_dir or None,
                    config={k: v for k, v in vars(a).items() if not k.startswith("wandb")})
with open(a.log, "a") as log:
    for s in cyc:
        if (a.stop_file and os.path.exists(a.stop_file)) or (a.max_batches and n >= a.max_batches):
            break
        if a.callback:
            try:
                while True:
                    h = requests.get(f"{a.callback}/health", timeout=5); h.raise_for_status()
                    if not a.max_pool or h.json().get("pool_size", 0) <= a.max_pool or (a.stop_file and os.path.exists(a.stop_file)):
                        break
                    time.sleep(2)   # the trainer is behind: let it drain the pool
            except Exception:  # noqa: BLE001
                log.write(json.dumps({"event": "callback_gone", "time": time.time()}) + "\n"); log.flush(); break
        r = requests.post(f"{a.server}/rollout", json={"prompts": prompts[s:s + a.batch], "max_tokens": a.max_tokens, "temperature": a.temperature,
                                                         "data_id_prefix": f"b{n:06d}", "push_url": a.callback or None, "mooncake": mc if a.callback else None,
                                                         "capture": bool(a.callback)}, timeout=3600)
        rec = {"batch": n, "requests_done": (n + 1) * a.batch, "time": time.time(), "status": r.status_code}
        if r.ok:
            j = r.json(); rec.update({k: j[k] for k in ("requests", "windows", "pushed", "generated_tokens", "accept_length", "gen_tokens_per_s", "total_seconds")})
        else:
            rec["error"] = r.text[:300]
        log.write(json.dumps(rec) + "\n"); log.flush(); n += 1
        if wb is not None and r.ok:
            try:   # logging must never stop the run
                upd = requests.get(f"{a.server}/health", timeout=5).json().get("draft_updates", 0)
                wb.log({"rollout/accept_length": rec["accept_length"], "rollout/gen_tokens_per_s": rec["gen_tokens_per_s"], "rollout/generated_tokens": rec["generated_tokens"],
                        "rollout/requests_done": rec["requests_done"], "rollout/windows": rec["windows"], "rollout/draft_updates": upd,
                        "rollout/trainer_step": upd * a.steps_per_sync}, step=rec["batch"])
            except Exception as e:  # noqa: BLE001
                print(f"wandb log failed: {e}", flush=True)
if wb is not None:
    wb.finish()
