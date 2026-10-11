#!/usr/bin/env python
"""Check that two models tokenize and render chat conversations identically: same vocabulary, same special tokens, same chat-template
output and token ids on the first N conversations of a JSONL source (`conversations`: [{role, content}, ...]).

Why: data prepared for model A (`speculators prepare-data`: chat template + tokenizer) can be reused for a fine-tuned copy B of A only
if this holds. Exit code 1 on any difference.

  python check_same_tokenizer.py --a Qwen/Qwen3-0.6B --b <fine-tuned model> --source sample.jsonl [--n 100]
"""
import argparse
import itertools
import json
import sys

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()

    ta, tb = AutoTokenizer.from_pretrained(args.a), AutoTokenizer.from_pretrained(args.b)
    problems = []
    if ta.get_vocab() != tb.get_vocab():
        problems.append("vocabularies differ")
    if ta.special_tokens_map != tb.special_tokens_map or ta.get_added_vocab() != tb.get_added_vocab():
        problems.append("special tokens differ")
    n = bad_text = bad_ids = 0
    with open(args.source, encoding="utf-8") as f:  # iterate the file object: U+2028 appears inside records
        for line in itertools.islice(f, args.n):
            msgs = [{"role": m["role"], "content": m["content"]} for m in json.loads(line)["conversations"]]
            for j in sorted({1, len(msgs) // 2 | 1, len(msgs) - 1}):
                sub = msgs[: j + 1]
                a = ta.apply_chat_template(sub, tokenize=False, enable_thinking=True)
                b = tb.apply_chat_template(sub, tokenize=False, enable_thinking=True)
                n += 1
                bad_text += a != b
                bad_ids += ta(a, add_special_tokens=False)["input_ids"] != tb(a, add_special_tokens=False)["input_ids"]
    if bad_text:
        problems.append(f"{bad_text} of {n} chat renderings differ")
    if bad_ids:
        problems.append(f"{bad_ids} of {n} texts get different token ids")
    print(f"{args.a} vs {args.b}: {n} renderings compared; vocab size {len(ta)} / {len(tb)}")
    if problems:
        sys.exit("NOT identical: " + "; ".join(problems))
    print("identical tokenization and chat template")


if __name__ == "__main__":
    main()
