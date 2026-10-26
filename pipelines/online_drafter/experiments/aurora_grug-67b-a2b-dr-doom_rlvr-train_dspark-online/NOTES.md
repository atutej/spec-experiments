# Aurora online DSpark drafter on the RLVR TRAIN prompts: scratch vs warm start from itw 80k (Dr. Doom target)

**Target: Dr. Doom** (`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`; `run.sh` serves it for the rollouts and hidden states, the configs name it as `target_model_path`). Two arms, one 12 h reservation node each, run side by side:
| arm | start | optimizer (as the Datakit runs) | sbatch |
|---|---|---|---|
| scratch | random init (the Datakit "scratch control") | lr 1e-4 constant, no warmup, 8 samples per optimizer step | `run_long_scratch.sbatch`, `config_scratch.yaml` |
| itw80k | `runs/eval_ckpts/drdoom_itw_step80813` (the Dr. Doom itw 80k drafter) | lr 1e-5, 30 steps of warmup, 32 samples per optimizer step | `run_long_itw80k.sbatch`, `config_itw80k.yaml` |
Both: Aurora's forward KL with its 0.8**i slot weights (NOT the TV loss; the Datakit warm-start run showed this costs about 0.1-0.15 accept length against a TV-trained start), confidence head frozen, DSpark 3 layers / block 8 / draft vocabulary 32,000.

## What differs from the Datakit runs
The prompts. Rollout prompts are the RLVR train rows (`open-athena/Nemotron-RL-Ultra-RLVR1-Cleaned-v3-2026-09-30`, 84,403 rows): ALL rows, including those with prefilled assistant / tool turns, rendered with the target's chat template as the RL runs send them (tools from
`request_json`, default `Reasoning: /think`), prompts up to 4,096 tokens; the eval prompts (the 1,024-row file `runs/nemotron_rlvr_prompts_1024`, of which the eval uses the first 256) are removed. Built by the eval prompt builder (`tools/eval_acceptance/build_nemotron_rlvr_prompts.py`, all rows) and `make_prompts.py`.
Completions are capped at **4,096 new tokens** (the RL eval's mean is about 15.5k, up to 49,152, so the drafter sees only the start of each completion), server window 8,192, trainer and Mooncake length 8,320, client Mooncake pool 96 GB with a throttle at 64 unfetched samples.
Expected volume (estimate, not measured on RLVR): about 1,200-1,900 generated tokens/s with capture, so roughly 12-20k rollouts per 12 h (about a fifth of the prompts once); about 2k optimizer steps for scratch, about 550 for itw80k; draft sync every 30 / 8 steps (about 10 min), full checkpoints every 500 / 125 steps.
The trainer's `<data file>` is still a Datakit shard (`data-00000-of-00215.arrow`): it only supplies the scratch arm's vocabulary mapping and the step count; the itw80k arm takes its mapping from the checkpoint.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/online_drafter/experiments/aurora_grug-67b-a2b-dr-doom_rlvr-train_dspark-online
# 1. the prompts (the eval builder's output is in runs/nemotron_rlvr_train_prompts_20261024/all_rows)
python make_prompts.py --all /scratch/09749/atutej/marin_speculator/runs/nemotron_rlvr_train_prompts_20261024/all_rows --eval /scratch/09749/atutej/marin_speculator/runs/nemotron_rlvr_prompts_1024 \
    --out /scratch/09749/atutej/marin_speculator/runs/nemotron_rlvr_train_prompts_20261024/prompts_token_ids.jsonl
# 2. smoke (one reservation node, about 30-60 min), then both arms
sbatch run_smoke.sbatch ; sbatch run_long_scratch.sbatch ; sbatch run_long_itw80k.sbatch
```
W&B: project `marin_aurora`, groups `dspark-rlvr-scratch-<jobid>` / `dspark-rlvr-itw80k-<jobid>`. Evaluate each arm's last draft (`drafts/step_*` of the run dir, copy to `runs/eval_ckpts/<name>/`: config.json, config.py, model.safetensors from a checkpoint or the HF-style draft) on the usual Dr. Doom evals (RLVR 256 prompts; Datakit multi-turn 1,024),
against Dr. Doom itw 80k (RLVR 2.689, Datakit pooled 2.932). Aurora saves `pytorch_model.bin` + `config.json` (the same key set as our checkpoints); vLLM loads such a directory directly as the draft (checked on the random-init scratch draft), `model.safetensors` is not required.

## Open points
- Only the first 4,096 completion tokens are seen; late positions of long completions (where most eval tokens are) are not trained on.
- The scratch arm gets about 2k optimizer steps, far fewer than the Datakit scratch control (23k steps of 8 short samples); the token count is similar (about 50-80M generated tokens per 12 h).
- Not run yet: both arms.

## Smoke test (2026-10-24, itw80k arm, 128 RLVR train prompts, 4,096 new tokens, 4 optimizer steps, job 1096921; a first smoke, 1096872, found the sync problem below)
Rollouts: 32 requests per batch, about 94k generated tokens, 84-102 s per batch (1,030-1,140 tokens/s with capture), accept length 2.48-2.78 for the itw 80k drafter on these prompts. Mooncake sizing and 8.3k-token samples through the DSpark trainer: fine
(loss 2.6-3.2, trainer acc_len 1.5-1.8 on the first steps). **Found and fixed:** Aurora gives `/update_weights_from_disk` 60 s, and the server only applies an update between rollout batches; with 90 s batches the first sync of smoke 1 timed out. The endpoint now copies
the trainer's draft directory, answers at once and applies the newest copy between batches (`tools/draft_updater.py`; `draft_updates` in `/health` counts applied updates): in smoke 2 both syncs (steps 2 and 4) were applied, no failure. The smoke runs were stopped by hand
after step 4 (the non-FOREVER script waits up to 30 min for a trainer marker that does not come).
