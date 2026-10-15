# DSpark drafter in MarinSkyRL RL: infrastructure notes (2026-10-14)

**Correction (same day):** the section on the vLLM pin below describes the stale `a51fefd4` checkout. MarinSkyRL `origin/main` (checked 2026-10-14)
already pins `vllm 0.0.0.dev20260929+marin.39e62869693c.cu132` with torch 2.13.0 and an aarch64 wheel, i.e. the build we serve with, and its `vllm` extra
includes `speculators` and `hs-connectors`. No wheel bump is needed for the pitch; the two EAGLE-3-only gates remain (see below). The workspace
clone is pinned to current main; the env is built with `UV_PROJECT_ENVIRONMENT=envs/skyrl uv sync --frozen --extra vllm`.

Goal: replicate Luke's online RL experiment (EAGLE-3 draft, R2E-Gym, MarinSkyRL; marin issues #9079, #9114,
gist bd14e461) with our DSpark drafter and the Datakit-SFT Grug target, on GB200, with no draft adaptation.
Everything below comes from reading code and release metadata; none of it has been run.

## Layers

1. **RL framework (MarinSkyRL `main`):** no code change. `generator.engine_init_kwargs.speculative_config`
   is passed straight to the vLLM engine (see OpenThoughts-Agent
   `hpc/skyrl_yaml/tacc/1node_snowball_eagle3_r2egym_smoke.yaml`). For us: `method: dspark`.
   The `online-eagle` branch (managed draft lifecycle) only allows `eagle3` and is not needed for a frozen draft.
2. **vLLM (the blocker):**
   - MarinSkyRL `main` pins `vllm 0.0.0.dev20260805+marin.fa50698a9a30.cu129` (torch 2.11). No `qwen3_dspark.py`.
   - Our serving pin `39e62869` (torch 2.13/cu132) has DSpark + GrugMoE, but is not a descendant of `fa50698a`
     (3685 commits ahead, 31 behind: GrugMoE pipeline-parallel serving, routed-expert capture, release plumbing).
   - Marin vLLM `main` (2026-10-13) has DSpark + GrugMoE and publishes GB200-validated wheels
     (`marin-vllm-gpu-20261013-ac3b5ea5e851`, torch 2.14.1/cu132). Moving MarinSkyRL there means a trainer torch bump.
   - Two gates are EAGLE-3 only, in the pin and in Marin vLLM `main`:
     `draft_requires_dp_sync` (`method != "eagle3" or is_moe`) and `_co_locate_eagle3_layers`
     (`method == "eagle3"`). Luke's equivalents: commits a9ae5c9 and 00d81ae on `lukedhlee/grugmoe-eagle3`.
     Risks for DSpark on a DP x EP Grug target: extra per-pass all-reduce (speed), and a 3-layer draft split across
     KV-cache groups (startup assertion). Untested.
3. **Drafter:** speculators-format checkpoint; `vllm/transformers_utils/configs/speculators/algos.py:update_dspark`
   maps `aux_hidden_state_layer_ids` to `eagle_aux_hidden_state_layer_ids` and `target_layer_ids` (i-1).
   `GrugMoe` already implements `SupportsEagle3` and returns the tapped layers. Draft is frozen during RL.

## Luke's patches, for DSpark

| commit | purpose | DSpark |
|---|---|---|
| d84b7c7 SupportsEagle3 for GrugMoE | hidden-state taps | already in pin |
| a9ae5c9 skip DP sync for dense drafters | speed | gated to eagle3; likely needed |
| 00d81ae co-locate draft layers in one KV group | startup | gated to eagle3; possibly needed |
| 4bb838c, 4cfae4e hidden-state connector | training-data extraction | not needed for RL |

## Pitch requirements

1. Small vLLM PR adding `dspark` to the two gates (only if the serving smoke test shows they matter).
2. A MarinSkyRL vLLM wheel bump to a release that includes it (check for an in-flight cu132 / torch 2.14 migration).
3. Evidence: with/without-draft RL arms (Luke: 31 vs 44 min for 3 steps, 1.42x, 5 nodes GH200), parity probes,
   TIS ratio, and a longer run for drift.

## Order

1. One-node GB200 serving smoke test: Datakit-SFT target + DSpark draft, DP x EP.
2. A/B the DP-sync gate; accept length and tok/s against plain decoding.
3. Patch the gates only if needed.
4. RL smoke (1 node), then 5-node with/without arms.

## RL smoke attempts on one GB200 node (2026-10-14, idev; results, not plans)

Setup: MarinSkyRL `5de48d34` (Megatron trainer, vLLM 39e62869, torch 2.13/cu132), env `envs/skyrl-megatron`, OpenThoughts-Agent 3bd1917e with the Vista patch,
Datakit-SFT Grug target, our DSpark drafter (`method: dspark`, 8 spec tokens), engines DP2 x EP2 on GPUs 0,1, Megatron policy EP=2 on GPUs 2,3.

- **Engine side works under RL:** target and drafter load, CUDA graphs for both are captured, engines report ready (about 2 min with warm caches).
- **Launcher/schema drift (found by running it):** MarinSkyRL main after `c96f6d25` (2026-10-09) rejects the Harbor keys OpenThoughts-Agent emits
  (`terminal_bench_config.{trials_dir,model_info,archiving,trace_upload}`, `harbor.name` and 12 more); FSDP keys, `use_tis`/`tis_imp_ratio_cap` (now
  `off_policy_correction: tis`), `async_engine`, `batched` are gone since `bd9a59f2` (FSDP removed, 2026-09-25). `5de48d34` is the last commit with the old rollout schema.
- **Vista environment fixes (needed for any Megatron RL env here):** (1) `CUDA_HOME` from the module points at a huge tree; `transformer_engine` globs it recursively at
  import for each CUDA library (minutes; looks like a hang, process in D state). (2) The pip toolkit alone has no `lib64`/unversioned `libcudart.so`, so FlashInfer's JIT
  fails with `ld: cannot find -lcudart`. Fix: `envs/skyrl_cuda_home` shim (symlinks), applied by `pipelines/rl_speculation/tools/patch_vista_sbatch.py`. (3) The flash-attn
  wheel needs a newer `libstdc++` than the system one (`envs/skyrl_extra_lib`). (4) `OT_AGENT_NO_GPU_GRES=1` (no GPU gres on Vista). `tools/launch_ot_rl.sh` does all of it.
- **Policy does not fit on 2 GPUs with Megatron:** OOM while loading ("Tried to allocate 60.94 GiB", 128 GiB already allocated per rank): bf16 weights about 61 GiB
  per rank, a gradient buffer of the same size, then a third. September's one-node smoke trained Snowball with FSDP2 `cpu_offload: true`, which Megatron does not have.
- **Megatron CPU offload is a dead end here:** TE 2.19 says "Offloading weights is deprecated. Using offload_weights=True does not have any effect", so only activations
  would be offloaded; and enabling `cpu_offloading` fails at layer construction with `TypeError: cannot pickle 'Stream' object when instantiating TransformerLayer`.
  Megatron also rejects `cpu_offloading` together with recompute (set `trainer.gradient_checkpointing=false`; the launcher drops null overrides).
- **Consequence:** training this 67B model with the Megatron trainer needs the policy on 4+ GPUs (EP=4, about 33 GiB of weights per rank) and the engines on another node: 2+ nodes, batch job.
  Not yet run: Daytona sandboxes (no run has reached rollouts), accept length inside RL, TIS ratio.
- Also still open: the smoke config uses a 16-task subset (`runs/rl_speculation/r2egym_tasks_16`) and the sbatch temperature is now 1.0 (it inherited 1.2 from the EAGLE-3 smoke YAML).

## Generating sbatch files on a login node (found 2026-10-14, the first A/B submit)

OpenThoughts-Agent picks the Vista profile (`vista` = gh, 1 GPU/node, or `vista_gb200` = gb, 4 GPUs/node) from `$SLURM_JOB_PARTITION`, which is unset on a login node, so it
fell back to the gh profile: `gpus_per_node 1`, `cpus 72`, `module load cuda/12.8`, a gh node-exclusion list, `-p gh` and the default account. A job generated that way would have
misconfigured Ray and the engine sizing on GB200. `tools/launch_ot_rl.sh` now exports `SLURM_JOB_PARTITION=gb`, and `submit_ab.sh` passes `--partition gb --account ASC26088
--reservation CGAI-gb`. Check a generated `*_rl_config.json` for `cluster_name: vista_gb200` and `gpus_per_node: 4` before submitting. Also fixed: Vista's sbatch wrapper prints a banner
before the job id, so the id used for `--dependency` must be the last number in the output, not the whole output (`AFTER=<jobid>` seeds the chain).

## Colocated RL run and the sleep-level finding (2026-10-14; achieved results, one run each)

**Colocated layout (policy EP=8 over 8 GPUs, two DP4 x EP4 engines on the same GPUs, `colocate_all`, vLLM sleep between phases)**, job 1072360, DSpark arm, 2 GB200 nodes:
all three RL steps completed (step wall times 1,013 s, 553 s, 709 s; policy 9.83B params per rank; weight sync 34.6 to 35.5 s; policy_train 83 s, 66 s, 75 s). The job then failed
in the final `save_checkpoints` (a policy worker became unreachable, `ActorUnavailableError: Socket closed`; cause not established, the worker log on node 2 was not collected). Not a training failure.
The 4+4 split layout had run out of GPU memory in its first training phase (the colocated layout halves per-GPU weights and gradients).

**The DSpark draft was useless in that run:** engine log 57 accepted tokens in 465,002 drafts (accept length 1.000); MarinSkyRL's own per-step metrics agree
(`vllm/spec_decode_mean_acceptance_length` 1.0, accepted 61 / 224 / 366 of 1.07M / 2.12M / 3.16M drafts). Its timings therefore measure the drafter's overhead, not its benefit.
In the split layout (no engine sleep) the same drafter had accept length 2.87 on RL rollouts.

**Cause, tested in isolation** (`tools/sleep_wake_probe.py`, one GPU per variant, 16 R2E-Gym prompts, T=1, 8 draft tokens; job 1072452):
| variant | accept length before | after |
|---|---|---|
| no sleep (control) | 2.08 | 1.89 (run-to-run noise is about 0.2) |
| sleep level 1, wake | 2.03 | 2.36 |
| sleep level 2, wake, reload target only (what the weight sync does) | 2.07 | **1.0002** (1 accepted of 4,364 drafts) |
MarinSkyRL's engine `sleep` defaults to level 2, which discards weights; the policy-to-engine sync refills only the target model; the draft's weights come back uninitialised.
The target is fine after the reload (coherent text). While asleep a one-GPU engine holds 3.7 GiB (levels 1 and 2 alike), against 171 GiB awake.
Fix (local patch, `setup/machines/vista/patches/marinskyrl-sleep-level.patch`): `SKYRL_VLLM_SLEEP_LEVEL=1`, which `tools/patch_vista_sbatch.py` exports. Level 1 keeps the
engine weights in host RAM (about 137 GiB per node here); the level-2 run peaked at 84% of 1,691 GiB host memory, so watch `gpu_mem_*.log` and `fd-monitor` lines.
Any RL setup that colocates a speculative-decoding draft with the policy needs the draft to survive sleep; a non-colocated engine (Luke's layout) never sleeps.
The generation-only comparison (accept 2.44, per-stream 1.3 to 1.9x faster) and the split-layout accept length (2.87) are unaffected.

### Fix options for the sleep-wipes-the-draft bug (proposed; only option 1 is in place)
1. **Sleep level 1 (applied locally).** `SKYRL_VLLM_SLEEP_LEVEL=1`: every weight, draft included, goes to host RAM and comes back on wake. One-line patch; costs about 137 GiB host RAM per node here and
   copies the target weights to host only to be overwritten by the next sync. Verified in isolation (accept 2.36 after wake), not yet in an RL run.
2. **Restore only the draft after wake (proper fix, not written).** Keep level 2, and after `wake_up` have MarinSkyRL's engine worker extension (it already injects a worker class used for weight updates)
   reload the draft from `speculative_config.model` (about 0.5 GiB read from disk per step; no host RAM cost). Needs a worker method that reaches the drafter's `load_weights`; test with a `sleep2 + draft reload` variant of the probe.
3. **In vLLM's sleep/wake.** Allocate the draft's weights under their own tag that always offloads (level-1 behaviour), so no caller needs to know. Best upstream shape; a vLLM change.
4. **Do not colocate** (Luke's layout: separate engine nodes). Avoids the bug, but needs more nodes than the 2-node reservation gives this policy.
Whichever is chosen, check the first `SpecDecoding` lines of any colocated run with a draft: accept length near 1.0 means the draft did not survive.

### Update (2026-10-14 evening): sleep level 1 did NOT fix the colocated run; the cause above is incomplete
Job 1072509 (colocated, DSpark arm, `SKYRL_VLLM_SLEEP_LEVEL=1` exported; host memory peak 1,425 -> 1,597 GiB, consistent with level 1) completed all 3 steps (582 s, 560 s, 650 s; checkpoints off, no crash),
but pooled accept length was again **1.000** (39 accepted of 348,161 drafts in 256 windows). So level 2 discarding the draft is not the whole story. The first probe changed two things between its sleeping variants
(level and wake style), so it cannot say which matters: the collapsing variant also woke by tag and reloaded the target, as RL does. Probe 2 (job 1072660) varies one thing at a time. Until its result is in, options 1 to 3
above are untested guesses about the cause; only "do not colocate" (option 4) is known to avoid the problem (accept 2.87 in the split layout).

### Root cause found (2026-10-14 night): the startup sleep passed an explicit level 2
Probe 2 (job 1072660; one change per variant, accept length before -> after): level 1 + wake by tag, no weight update 2.09 -> 2.02; level 1 + full wake + target reload 2.08 -> 2.15; level 1 + wake by tag + target
reload (the RL flow) 2.31 -> 1.81; level 2 + full wake + target reload 1.83 -> 1.001. Only level 2 kills the draft; neither the by-tag wake nor a target reload does. The RL run with level 1 (job 1072509) still
collapsed because `create_ray_wrapped_inference_engines` (ray_wrapped_inference_engine.py, about line 998) puts each engine to sleep at startup with an explicit `level=2` (the function default `sleep_level=2`).
My first patch changed only the default for calls that pass no level, so the draft was discarded at startup, before the first wake-up, and the later level-1 sleeps preserved a blank draft.
The patch (`setup/machines/vista/patches/marinskyrl-sleep-level.patch`) now lets `SKYRL_VLLM_SLEEP_LEVEL` override every caller and logs the level on each sleep. Check the job log for
"AsyncVLLMInferenceEngine.sleep: level=1" at startup, then accept length well above 1 in the first rollouts. Still unconfirmed in an RL run.

### State of the clones and the next run (2026-10-14 night)
MarinSkyRL clone: reverted to the pinned 5de48d34 (clean). The sleep-level fix is saved, not applied: `setup/machines/vista/patches/marinskyrl-sleep-level.patch`. The only change applied is the diagnostic
`marinskyrl-peak-memory.patch` (prints `[cuda-mem] rank=.. phase=.. peak_alloc_gib=.. peak_reserved_gib=..` at the end of the policy worker's `forward`, `ppo_train` and `broadcast_to_inference_engines`; the same numbers already go
to a telemetry backend that is not wired up outside Marin's infrastructure; needs the native allocator, so no `expandable_segments`). Next run: the split layout, to see whether Luke's layout fits by parameters alone
(`logprob_chunk_size` 256, checkpoints off, GPU memory logger on). The split layout never sleeps the engine, so the draft-survival bug does not apply to it.
Bound used to judge the split layout: it has about 95 GiB per GPU for activations and temporaries (184 - 17 non-PyTorch - 72 for bf16 weights and gradients); the colocated run showed the true peak is at most about 117 GiB.

### Split layout with the peak-memory log (job 1072803, 2026-10-14 night): training fits, the weight sync after it does not
`[cuda-mem]` lines (MarinSkyRL clone with `marinskyrl-peak-memory.patch`), policy on node 1 at EP=4, `logprob_chunk_size` 256, checkpoints off:
| phase | peak allocated (rank 0 / rank 3) | outcome |
|---|---|---|
| first weight sync, before training | 73.5 / 73.5 GiB | success |
| logprob forward, step 1 | 77.1 / about 75 GiB (reserved up to 167 GiB: the cache fills the free space) | success |
| ppo_train, step 1 | 88.8 / 85.6 GiB | success |
| weight sync after step 1 | 76.9 / 77.1 GiB at failure | **OutOfMemoryError** in `extract_weights` -> `megatron_to_hf` |
So the true training peak (about 89 GiB) fits under the roughly 95 GiB the split layout leaves per GPU; the training-side parameter changes (logprob chunk 256, checkpoints off) were enough for training.
The sync after training is what runs out. The GPU logger shows why it looks inexplicable from the process alone: after training the policy processes hold 93 GiB each, but the GPUs read 181 / 146 / 107 / 181 GiB,
so 87 / 52 / 14 / 88 GiB per GPU sit outside any listed process (clean at job start; the engine node is consistent: process 181 GiB, GPU 181 GiB). I did not establish what holds it; the pattern (different amounts per GPU,
growing during the gather) fits a gather that accumulates the HF-format weights.
Next try, configured and validated offline but not run: `generator.weight_sync_transport: expert_block` with `engine_init_kwargs.kernel_config.moe_backend: triton`, which broadcasts each expert matrix from the trainer rank that holds it to the
engine workers that serve it, with no full gather. It changes the engine's MoE kernel (default "auto"), so engine throughput may differ from earlier measurements. The colocated layout did the same sync three times without a problem.

### Split layout with expert_block works (job 1072882, 2026-10-14 23:05): RL step 1 end to end with the DSpark draft intact
Config: split layout (policy node 1 EP=4, one DP4 x EP4 engine on node 2), `weight_sync_transport: expert_block`, engines on the Triton MoE backend, `logprob_chunk_size` 256, checkpoints off, clean MarinSkyRL clone plus the
peak-memory log. Step 1 (16 prompts x 4 samples, 64 trajectories, 1.33M response tokens): training step 600.6 s; the weight sync after training took **1.5 s** (the default broadcast sync ran out of memory twice in this layout and took
35 s in the colocated one); ppo_train peak allocated 84.6 GiB; MarinSkyRL's metrics: `spec_decode_mean_acceptance_length` 2.543, `acceptance_rate` 0.193, `latency_decode_mean` 0.877 s (p90 1.83 s), `latency_e2e_mean` 0.927 s,
median generation throughput 733 tok/s on 4 engine ranks. Steps 2 and 3 pending. Not yet a speed comparison: the plain arm in the same layout is needed (the engine uses the Triton MoE backend here, unlike the earlier "auto" runs).
