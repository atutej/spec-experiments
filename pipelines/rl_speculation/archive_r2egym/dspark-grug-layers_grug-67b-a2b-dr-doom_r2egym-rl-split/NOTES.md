# RL run, split layout, Dr. Doom target + Grug-layer DSpark drafter (planned, nothing run)

**Scope (user, 2026-10-29): a full run, not a smoke test.** 64 steps (my choice, see the comment at `max_steps` in `rl_dspark.yaml`), 16 prompts x 4 samples, periodic checkpoints every 10 steps (keep 2, untested save path), resume resolved by the launcher per chained job (`resume_mode` left null), 12 h jobs chained with `CHAIN=2` (afterany). Checkpoints of a 67B policy with optimizer state are large: check scratch space before the first save.

Revisit of `dspark_grug-67b-a2b-datakit-sft_r2egym-rl-ab` (2026-10-14) with the new drafter. Created 2026-10-29. Old folders are not modified.

## Decisions (user, 2026-10-29)
- Target: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`.
- Drafter: `runs/eval_ckpts/drdoom_grug_layers_step76600` (Grug layers, itw, Dr. Doom). Needs a vLLM build with branch `grug-dspark-layers` (the pinned wheel has no `layer_type=grug`).
- **No colocated mode.** `placement.colocate_all: false`, engines never sleep (`inference_engine_enable_sleep=False`), which sidesteps the level-2 sleep that wiped the draft (accept 1.000).
  Layout = job 1072882: policy EP=4 on node 1, one DP4 x EP4 engine on node 2, `weight_sync_transport: expert_block`, `kernel_config.moe_backend: triton`, `logprob_chunk_size: 256`, checkpoints off.
- Runtime: `vendor/skyrl` from marin (snapshot `ref/marin_vendor_20261021`, marin 52f7529) if it needs no source changes; fall back to the MarinSkyRL clone and say why.
- Dspark arm only. The plain arm is a separate experiment folder, later. Goal of this stage: find a configuration that runs several steps without bugs, accept length well above 1.

## Known checks from reading the vendored tree (not run)
- Same vLLM pin as our serving (39e62869, cu132); `expert_block` and `inference_placement.py` exist, so the split layout is supported.
- `marinskyrl/speculative_decoding.py` is eagle3-only, but only for the managed `speculative_config`; raw `engine_init_kwargs.speculative_config` passthrough (as before) bypasses it.
  Consequence to verify: after each weight sync the draft's `lm_head` is not refreshed (the DSpark checkpoint has its own lm_head). Watch accept length across steps.
- Open blocker candidate: the launcher (OpenThoughts-Agent 3bd1917e) emits the old Harbor rollout schema, rejected by MarinSkyRL after c96f6d25; the vendored tree has `iris_harbor_environment.py` and a new schema.
  Whether the old YAML keys load there is the first thing to test (config validation only, login-node safe). If it needs source edits, stop and report.
- vLLM overlay for `grug-dspark-layers` must be put on the engine's Python path (the vendored uv lock installs the stock wheel).

## Steps
1. Validate the YAML against vendor/skyrl's config loader offline. 2. Build/locate the vendor env on Vista (aarch64 wheel). 3. Overlay vLLM branch. 4. 2-node job, 3 steps. 5. Gate: first `SpecDecoding` lines accept > 1, and again after step 1's weight sync.
