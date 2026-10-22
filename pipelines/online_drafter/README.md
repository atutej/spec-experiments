# online_drafter

Train a speculative-decoding drafter ONLINE from the target's own rollouts with [Aurora](https://github.com/togethercomputer/aurora) (trainer side), while the Marin vLLM fork serves the target and captures hidden states. Aurora supports SGLang only; Grug has no SGLang support, so the server is ours.

```
rollout_client.py ──POST /rollout──▶ vllm_capture_server.py (Marin vLLM fork + capture_ext.py)
   (prompts)                           │ hidden states ──▶ Mooncake ◀── Aurora trainer (python -m aurora.train_entry)
                                       │ ◀── POST /update_weights_from_disk (new draft every N steps) ──┘
                                       └──▶ POST /push_sample (keys + shapes) ──▶ Aurora callback server
```

- `tools/` — `capture_ext.py` (vLLM worker extension), `vllm_capture_server.py`, `rollout_client.py`, `lib.sh` (process helpers, `forever_watch`), data converters, `compare_accept_curves.py`.
- `experiments/aurora-repro_qwen3-8b-external-with-draft/` — Aurora's own example on Qwen3-8B / ShareGPT (reproduction).
- `experiments/aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_eagle3-online/` — Snowball (Dr. Doom) EAGLE-3 from a random draft on Datakit prompts. Its `run.sh` also runs the DSpark experiment (`INIT_DRAFT`, `METHOD`, `SPEC_TOKENS`, `CONFIG`).
- `experiments/aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_dspark-online/` — continue one of our DSpark checkpoints (e.g. `drdoom_targetgen_warm80k_final`) online: config + `run_long.sbatch`.
- Each experiment: `run.sh <NEW run dir> <data>` (smoke or `FOREVER=1`), `run_long.sbatch` (12 h gb job, runs until Slurm ends it).
- The accept length that is comparable to the Aurora paper's plots is `accept_length` in `rollouts.jsonl` (measured while decoding), not the trainer's `train/simulated_acc_len` (a product of per-position training accuracies, without the bonus token).

## Deviations from upstream (Aurora, vLLM, Marin)

Rule we follow: patch Aurora (our fork, branch `vllm-engine`), do NOT change Marin-owned repos (vLLM fork, marin, MarinSkyRL). Everything below is a local choice made for that reason or for simplicity; "upstream change" is what removing the workaround would take.

| # | What we do locally | Why | Upstream change that would replace it |
|---|---|---|---|
| 1 | Mooncake publishing in `capture_ext.py` + `vllm_capture_server.py`; the fork's `OnlineEagleCapture` is used unmodified | no edits to the vLLM fork; the fork writes safetensors, Aurora reads Mooncake keys | a Mooncake sink in `online_eagle.py` and a streaming drain RPC |
| 2 | vLLM runs as a separate process | the fork needs transformers >= 5.10.4, Aurora pins 4.57.1 | none (a dependency conflict, not a patch) |
| 3 | `vllm_capture_server.py` implements `/update_weights_from_disk` (and a `/rollout` endpoint) | Aurora's external mode expects an SGLang-style server | a `VllmEngine` in Aurora |
| 4 | DSpark capture: `capture_ext.py` `ac_begin` calls the fork's resolver with `"eagle3"` hard-coded and sets `runner.online_eagle_capture` (DSpark runs on the fork's model runner V2, whose capture hooks and aux-layer lookup already handle it) | the fork's `resolve_online_eagle_capture_config` rejects any method but `eagle3`. Spiked 2026-10-22 on Dr. Doom + `warm80k`: windows, aux layers [2, 12, 22], draft reload all work | accept `"dspark"` in that resolver (one line) plus a capture test |
| 5 | Aurora fork commits: optional SGLang import (`2a32ce6`), `loss_mask` accepted by the pretokenized loader (`bf633cd`), `load_target_config` fallback for `grug_moe` (`9b5ba3d`) | Aurora needs SGLang at import; Datakit rows carry `loss_mask`; unknown model type | upstream PRs to Aurora |
| 6 | Prompt rendering and last-turn cutting in `rollout_client.py` (conversations through the last user turn; token rows cut at the start of the last supervised span) | one data file serves both the client and Aurora's trainer | none needed |
| 7 | Client throttle `--max-pool` (default 96) and `--mooncake-segment-gb` (default 16) | in external mode nothing slows the server down; a fast server overran the 6 GB Mooncake pool and the trainer lost its oldest samples (Qwen long job 1088903) | backpressure on `/push_sample` or a Mooncake pool sized from `max_sample_pool_size` in Aurora |
| 8 | Client logs accept length, tokens/s, draft updates to a second W&B run in the trainer's group | the accept length is measured by the client/server, the W&B run belongs to the trainer | log it from Aurora's controller loop |
| 9 | `forever_watch` copies the trainer's draft-sync dir to `drafts/step_N` after each sync (temp dir + rename; keeps the last 5 and every 1000th) | Aurora overwrites that directory in place, so a kill mid-write can tear it | atomic draft saves in Aurora |
| 10 | The run scripts decide when to stop (marker or deadline) / run until Slurm ends the job | in external mode Aurora never exits ("Max steps reached, ... Waiting for more data") | an exit option in Aurora's external mode |
| 11 | Single node, server TP=1 on one GPU, no DP/EP | Mooncake across nodes failed earlier (`put_tensor -800`, undiagnosed) | diagnose the cross-node Mooncake path |
| 12 | DSpark in Aurora (fork branch `vllm-engine`, commits e68d24e and 3495a95): `models/draft/dspark.py` (port of the speculators DSpark/DFlash model, our parameter names), `models/dspark.py` (`DSparkModel`: Eagle3Model's interface, anchored blocks, Aurora's forward KL), `training/dspark_trainer.py` (`DSparkTrainer(Eagle3Trainer)`, confidence head frozen, TODO to revisit), 2 hooks in `Eagle3Trainer.init_model`, a dispatch on `decode.speculative_algorithm: DSPARK`, DSpark checkpoints in `AutoEagle3DraftModel.from_pretrained` (they have no `model_type`) | `speculators` needs transformers 5.x, Aurora pins 4.57.1, so the model is ported rather than imported; the loss is Aurora's, not our CE+TV; RoPE is computed on the fly (a buffer broke Aurora's per-rank init) | a DSpark model and trainer upstream in Aurora |

Keep this table current when a deviation is added or removed.

## Known limits

- The trainer's `train/simulated_acc_len` and the server's accept length are different quantities (see above).
- Capture slows generation about 4x on Qwen3-8B (2.5k vs 9.7k tok/s); cause not investigated.
- The draft-copy watcher (`forever_watch`) and the client's W&B logging were tested only against stubs; the real runs had not exercised them when this was written.
