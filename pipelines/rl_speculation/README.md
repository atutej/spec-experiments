# rl_speculation

Use a finished drafter inside an RL run: does speculative decoding shorten rollouts, and does it leave the learning unchanged?
Replicates Luke Lee's EAGLE-3 / R2E-Gym / MarinSkyRL experiment (marin issues #9079, #9114) with our DSpark drafter
(`speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_itw`) on the Datakit-SFT Grug target, on GB200,
with no draft adaptation. Background and open risks: `docs/rl_dspark_infra.md`.

- **Envs:** `vllm` (serving smoke test), `speculators` (prompt building, the acceptance client). RL stages need a MarinSkyRL uv env with a
  DSpark-capable vLLM; it is not built yet (decided after the serving smoke test).
- **Repos:** `vllm`, `speculators`, `MarinSkyRL`, `OpenThoughts-Agent` (launcher `hpc.launch`; Vista needs `setup/machines/vista/patches/`). `setup/repos.txt`.
- **Reuse:** the acceptance client is `speculator_training/tools/eval_acceptance/run_eval.py`; nothing here copies it.
- **`experiments/<name>/`:** the current runs: Reasoning Gym RL (`dspark-drdoom_reasoning-gym-rl-split`) and multi-turn text-to-SQL RL (`dspark-drdoom_text2sql-native-think`), both with the Dr. Doom target and the targetgen warm80k DSpark drafter.
- **`archive_r2egym/`:** the R2E-Gym stages, kept for when R2E-Gym RL resumes (blocked for now by the shared Daytona snapshot quota): serving smoke test, generation-only rollouts, the RL A/B layouts (`..._r2egym-rl-ab`), the two planned Dr. Doom split runs, and the snapshot-registration tools. See `archive_r2egym/README.md`.
- **`tools/`:** `build_r2egym_prompts.py`.

Every number in a NOTES.md says whether it is a target or an achieved result, and carries its hardware and config.
