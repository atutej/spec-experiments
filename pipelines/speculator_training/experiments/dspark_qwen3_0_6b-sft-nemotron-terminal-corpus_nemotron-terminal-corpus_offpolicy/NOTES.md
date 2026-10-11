# DSpark drafter for the Nemotron-Terminal-fine-tuned Qwen3-0.6B, trained on the corpus's own completions

Sibling of `../dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy` (read its `NOTES.md` for the pipeline, the render server and the
Vista learnings). That experiment's target is the base `Qwen/Qwen3-0.6B`. **This one's target is Qwen3-0.6B fine-tuned on the same
corpus** (`pipelines/sft/experiments/sft_qwen3_0_6b_nemotron-terminal-corpus`), published at
https://huggingface.co/atutej/sft_qwen3_0_6b_nemotron-terminal-corpus-20261011 (commit `cd76e0562da3c278e2e031b961fd7cd260d12d6d`).
The point: after SFT on the corpus, the corpus's completions are close to the target's own text, so the off-policy drafter trained on
them should be much closer to on-policy than with the base model (the open question of the off-policy experiment).

|  | base-target off-policy | this experiment |
|---|---|---|
| Target model (hidden states in training, `lm_head`, rollouts in the eval) | `Qwen/Qwen3-0.6B` | the SFT model above (bf16: it is saved in fp32) |
| Conversations, completions, rows, sequence length / anchors, DSpark settings, LR | | the same |
| Prepared data | rendered by `prepare-data` (~3 h) | **the same files**: `prepare` links the sibling's `data/` after `tools/check_same_tokenizer.py` finds identical tokenization |
| `run.sh` | | `--dtype bfloat16` for the hidden-state server; `prepare` links the data |

## Why the prepared data can be reused

The data holds token ids, a loss mask and the reduced-vocabulary maps (`d2t`/`t2d`, from token frequencies of the data); nothing depends on
the target's weights. It stays valid if the target tokenizes and renders chat like the base model. Checked on 2026-10-11: same vocabulary
(151,669), same special tokens, 900 chat renderings of corpus conversations identical, same token ids (the SFT run re-saved the tokenizer with
transformers 5.8, so the files differ byte for byte). `prepare` re-checks this on 100 conversations of the sample before linking, and falls
back to rendering the corpus itself if `$SHARED_DATA_DIR` does not exist.

## Run

```
bash vista/submit_chain.sh [--dry-run]        # login node: gg job (export, prepare = link), then a gb job (train)
bash vista/run.sh [export] [prepare] [train]  # as-is on an idev node (train needs 4 GPUs)
bash vista/submit_eval.sh [CHECKPOINT_NAME]   # rollout acceptance eval, target = the SFT model (see ../../tools/eval_acceptance/)
```

`MODEL` can be a local path of the same model instead of the Hub id. W&B: project `marin_speculator`, run name = this folder's name.

## Status (2026-10-11)

Smoke test on a gb idev node through `run.sh export prepare train` (300-conversation sample for `export`, `MAX_STEPS=10`): the model was
downloaded from the Hub by vLLM and the trainer, the tokenizer check passed, the prepared data was linked, 10 steps ran (train loss 1.02 -> 0.98),
the validation epoch ran (about 18 minutes over the whole validation split, as in the sibling run), `checkpoint_best` was written, and its
config records the verifier `atutej/sft_qwen3_0_6b_nemotron-terminal-corpus-20261011`. Not done: the full training run (about 6 h on a gb node
in the sibling), and its rollout eval.

Comparing results: the acceptance lengths of this drafter and of the base-target drafters are measured against different targets, so they do
not rank each other directly. The comparison to make is each drafter against its own target's rollouts, on the same prompts
(`runs/eval_prompts_500`).
