# RL A/B on 2 GB200 nodes: DSpark draft vs plain, Datakit-SFT Grug target, R2E-Gym, MarinSkyRL (Megatron)

Status 2026-10-14: layouts built and exercised; **no with/without comparison yet**. Full findings: `docs/rl_dspark_infra.md` (sections "RL smoke attempts", "Generating sbatch files on a login node",
"Colocated RL run and the sleep-level finding").

- Layouts (`vista/submit_ab.sh`, `LAYOUT=`): `coloc` (default; policy EP=8, two DP4xEP4 engines sharing all 8 GPUs, sleep between phases) and `split` (policy 4 GPUs EP=4 on node 1, one engine on node 2; ran out of GPU
  memory in its first training phase).
- Run: `bash vista/submit_ab.sh --submit` from a login node submits the DSpark arm, then the plain arm behind it (`ARMS=`, `AFTER=<jobid>`, `TIME_LIMIT=`). 3 RL steps, 16 prompts x 4 samples, 64 concurrent trials,
  `max_turns` unbounded, window 32,768, checkpoints off. Partition gb, account ASC26088, reservation CGAI-gb.
- Needs before submitting (it is all in `setup/machines/vista/patches/`, applied to the workspace clones): the OpenThoughts-Agent Vista patch and the MarinSkyRL sleep-level patch; `tools/patch_vista_sbatch.py`
  then exports `SKYRL_VLLM_SLEEP_LEVEL=1`.
- Check first: after the first rollouts, `SpecDecoding` lines in the job log must show accept length well above 1 (about 2 to 3). 1.0 means the draft did not survive sleep: cancel.
- Measures to compare between arms: step wall time, `vllm/latency_decode_mean` / `_p90` and `vllm/latency_e2e_mean`, `vllm/spec_decode_mean_acceptance_length` (per-step console line "vLLM Stats (step N)" and `WANDB_MIRROR` lines).
  One run per arm: say so, and do not read differences below the noise (accept length varied by about 0.2 between identical passes).
