#!/usr/bin/env python
"""Build the JSON that Marin's Evalchemy client (marin/evaluation/evalchemy/client.py) reads from $EVALCHEMY_CLIENT_CONFIG, for one policy task.

  python make_client_config.py --policy-dir <.../evalchemy-configs> --task gsm8k-0shot --model <served model id> --out-path <dir> [--seed N] [--temperature 1.0]

Fields follow the policy's own mapping (experiments/evaluation/evals.py `evalchemy_run_config`/`config_for` and marin runner `_run_config_json`) for the Dr. Doom model
config: chat template on, no generation cap (Evalchemy sizes it from the served context), num_concurrent 16 unless the task file sets one, the task file's
chat_template_kwargs / gen_kwargs / batch_size / seed. --temperature (off by default: sampling is whatever the policy sets, i.e. the task file's gen_kwargs, else the task's own default) adds temperature=<T> to gen_kwargs;
--seed replaces the task file's seed. Prints the JSON on stdout.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

DEFAULT_NUM_CONCURRENT = 16   # marin.evaluation.evalchemy.runner


def eval_task_directory(name, num_fewshot, task_alias):   # marin.evaluation.evaluation_config
    return task_alias or f"{name}_{'default' if num_fewshot is None else num_fewshot}shot"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-dir", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--out-path", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--limit", type=int, help="evaluate only this many items per task (smoke tests)")
    a = p.parse_args()
    src = yaml.safe_load((Path(a.policy_dir) / f"{a.task}.yaml").read_text())
    if src.get("judge"):
        sys.exit(f"{a.task} needs a judge; not supported here")
    tasks = []
    for name in src["tasks"]:
        o = (src.get("task_options") or {}).get(name) or {}
        shots = o.get("num_fewshot", src.get("num_fewshot"))
        tasks.append({"name": name, "num_fewshot": shots, "dir": eval_task_directory(name, shots, o.get("task_alias")),
                      "generation": bool(o.get("generation", False)), "unsafe_code": bool(o.get("unsafe_code", False)),
                      "completion_only": bool(o.get("completion_only", False))})
    extra_model_args = dict(src.get("extra_model_args") or {})
    num_concurrent = int(extra_model_args.pop("num_concurrent", DEFAULT_NUM_CONCURRENT))
    for k in ("max_length", "max_model_len"):
        extra_model_args.pop(k, None)
    gk = src.get("gen_kwargs") or ""   # the policy's form: a comma-separated "key=value" string
    gen = {}
    for item in gk.split(","):
        if item.strip():
            k, v = item.split("=", 1)
            if k.strip() not in ("max_tokens", "max_new_tokens", "max_gen_toks"):
                gen[k.strip()] = v.strip()
    if a.temperature is not None:
        gen.update({"temperature": str(a.temperature), "do_sample": "true"})
    cfg = {"base_url": a.base_url, "model_id": a.model, "tokenizer": a.model, "tasks": tasks, "out_path": a.out_path,
           "apply_chat_template": True,                                  # the Dr. Doom model config: apply_chat_template: true
           "chat_tokenizer_backend": src.get("chat_tokenizer_backend", "none"),
           "debug": False, "max_gen_toks": src.get("max_tokens"), "extra_gen_kwargs": gen,
           "chat_template_kwargs": dict(src.get("chat_template_kwargs") or {}), "max_eval_instances": a.limit if a.limit is not None else src.get("limit"),
           "num_concurrent": num_concurrent, "batch_size": None if src.get("batch_size") is None else str(src["batch_size"]),
           "seed": a.seed if a.seed is not None else src.get("seed"), "extra_model_args": extra_model_args, "max_length": src.get("max_length")}
    print(json.dumps(cfg))


main()
