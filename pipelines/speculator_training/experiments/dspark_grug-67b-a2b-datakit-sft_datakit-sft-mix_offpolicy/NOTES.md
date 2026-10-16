# Serving the Grug DSpark drafter and measuring acceptance length (2026-10-15)

## Recommendation
Serve with `draft_sample_method: probabilistic` and `rejection_sample_method: standard` for sampled decoding (T=1), including RL rollouts:

    vllm serve open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 \
      --speculative-config '{"method":"dspark","model":"<drafter dir or HF repo>","num_speculative_tokens":8,
                             "draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'

For RL, put the same fields in the engine's `speculative_config` block (`pipelines/rl_speculation/configs/*dspark*.yaml` currently omit the two sampling fields).

**Where the setting lives:** not in the drafter checkpoint. From a speculators-format checkpoint, vLLM reads the method (`dspark`), `num_speculative_tokens` (from the first `proposal_methods` entry), the tapped target layers (`aux_hidden_state_layer_ids: [2, 12, 22]`), the verifier name and the weights (including the draft-vocabulary map). It ignores the checkpoint's `proposal_type: greedy` / `default_proposal_method: greedy`
and never reads the sampling modes from it; they default to `draft_sample_method: greedy`, `rejection_sample_method: standard` unless the serving config sets them. State the recommended config in the model card. At T=0 the drafting mode does not matter (verification is greedy).

Basis: vLLM's own DSpark tests and configs (including the acceptance test of a speculators-trained checkpoint) use `probabilistic` at T=1; the DSpark paper (arXiv 2607.05147) and DeepSpec's evaluator measure real decoding at T=1 with sampled drafts, token-wise rejection sampling `min(1, p_t/p_d)`, acceptance length = accepted + 1 per round pooled over all rounds. `rejection_sample_method: block` (used in the Kimi-K3 config) was not tested here.

## Measurements (step-80,813 drafter, 128 held-out validation prompts, T=1, 4096 tokens, DP4 x EP4 on one gb node)
| setting | acceptance length | output tok/s | vs no draft (1,858) |
|---|---|---|---|
| greedy drafts (vLLM default) | 2.660 | 3,136 | 1.69x |
| probabilistic drafts, standard verification | 2.973 | 3,417 | 1.84x |

Same prompts, greedy drafts: 100k-sample drafter 2.140, this run's step-39,066 checkpoint 2.536, step-80,813 2.660 (the full run improves with training). On R2E-Gym first-turn prompts the full-run drafters score about 2.2 (100k drafter 1.86): R2E agent text is harder than the training mix. Single runs; repeats of the same configuration differed by about 0.02.

## Why serving shows less than the trainer's `accept_len`
The trainer's validation `accept_len` (3.785 for step 80,813 on these rows) averages the accepted length of a round started at each of many positions, equally weighted. Serving's acceptance length is the mean over the rounds it actually takes, and rounds are chained: after a long accepted run it jumps over the easy stretch, after a short one it stays in the hard spot. Rounds are therefore weighted toward hard positions. Checks:
- Real vLLM, probabilistic + standard, one round started at 3,047 uniformly random positions in the first reply: **3.585 +/- 0.046**; the trainer's formula with the same weighting: 3.53 to 3.58. Chained decoding on the same prompts: 2.973.
- The trainer's formula applied only to the anchors serving visits: 2.92 (trainer's drafter) / 2.83 (vLLM's drafter), against 2.973 measured; on the anchors serving skips: 3.91.
- Teacher-forced parity (same rows, same anchors, Markov head fed the dataset tokens): the trainer's drafter and vLLM's drafter pick the same token in 97 to 98% of slots; acceptance length on the common anchors 2.642 vs 2.641; the formula applied to each drafter's distribution: 2.91 vs 2.83. No drafter or hidden-state implementation bug was found.
- The trainer's `eal` (greedy, vs the target's argmax) is a T=0 quantity; `accept_len` assumes sampled drafts. The 32k draft vocabulary covers 98% of the training text's supervised tokens but about 91% of the target's R2E outputs.
So the trainer's number is a good measure of drafter quality and tracks serving, but it overstates serving's tokens per step by about 0.6 on the first reply (3.585 vs 2.973). A chained-rounds estimate would be needed for it to predict serving directly.

## Scripts
- `tools/eval_acceptance/build_val_prompts.py`: held-out validation prompts (the trainer's split: the last rows of the dataset) as token-id prompts.
- `tools/eval_acceptance/serve_eval.sh`: serve the target with and/or without drafters on a set of prompts (env `PLAIN`, `ARMS`, `SPEC_EXTRA`, `PROMPTS_DIR`, ...) and report acceptance length and output tokens per second through `run_eval.py`.
- The debug tools used for the investigation (forced-reference decoding hook, parity dump, probes) are not in the repo.
