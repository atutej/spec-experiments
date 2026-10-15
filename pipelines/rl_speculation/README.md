# rl_speculation

Use a finished drafter inside an RL run: does speculative decoding shorten rollouts, and does it leave the learning unchanged?
Replicates Luke Lee's EAGLE-3 / R2E-Gym / MarinSkyRL experiment (marin issues #9079, #9114) with our DSpark drafter
(`speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_offpolicy`) on the Datakit-SFT Grug target, on GB200,
with no draft adaptation. Background and open risks: `docs/rl_dspark_infra.md`.

- **Envs:** `vllm` (serving smoke test), `speculators` (prompt building, the acceptance client). RL stages need a MarinSkyRL uv env with a
  DSpark-capable vLLM; it is not built yet (decided after the serving smoke test).
- **Repos:** `vllm`, `speculators`, `MarinSkyRL`, `OpenThoughts-Agent` (launcher `hpc.launch`; Vista needs `setup/machines/vista/patches/`). `setup/repos.txt`.
- **Reuse:** the acceptance client is `speculator_training/tools/eval_acceptance/run_eval.py`; nothing here copies it.
- **`experiments/<name>/`:** one stage each, same shape as `speculator_training`:
  1. `..._serving-smoke`: no RL, one GB200 node. Serve the target with and without the drafter; engine starts, accept length, tok/s, output parity.
  2. `..._r2egym-rl-smoke` (not created yet): MarinSkyRL, one node, a few steps, `speculative_config` method dspark.
  3. `..._r2egym-rl-ab` (not created yet): 5 nodes per arm, draft vs plain, same seed and tasks.
- **`tools/`:** `build_r2egym_prompts.py`.

Every number in a NOTES.md says whether it is a target or an achieved result, and carries its hardware and config.
