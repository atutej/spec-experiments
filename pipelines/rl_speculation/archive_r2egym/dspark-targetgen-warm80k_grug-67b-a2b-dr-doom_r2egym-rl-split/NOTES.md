# RL run, split layout, Dr. Doom target + Qwen3-layer targetgen-warm80k drafter (planned, nothing run)

**Scope (user, 2026-10-29): a full run, not a smoke test.** 64 steps (my choice, see the comment at `max_steps` in `rl_dspark.yaml`), 16 prompts x 4 samples, periodic checkpoints every 10 steps (keep 2, untested save path), resume resolved by the launcher per chained job (`resume_mode` left null), 12 h jobs chained with `CHAIN=2` (afterany). Checkpoints of a 67B policy with optimizer state are large: check scratch space before the first save.

Sibling of `dspark-grug-layers_grug-67b-a2b-dr-doom_r2egym-rl-split` (read its NOTES.md for the decisions, the vendor/skyrl findings and the checks). Created 2026-10-29.
Drafter: `runs/eval_ckpts/drdoom_targetgen_warm80k_final` = targetgen (trained on the Dr. Doom target's own replies), warm-started from the itw-80k checkpoint; Qwen3 layers (config has no layer_type), 3 layers, block 8.
Stock vLLM wheel, no overlay. Work dir `runs/rl_speculation/drdoom_targetgen_warm80k_split_20261029`, job `targetgen-warm80k-drdoom-rl-split`.
Same target, split layout (no colocation), same config otherwise (`diff` against the Grug-layer folder's rl_dspark.yaml: drafter path and comments only).
A first version of this folder (`..._itw80k_...`) pointed at the plain itw-80k checkpoint by mistake of reading "warm80k"; it was never submitted and was removed.
