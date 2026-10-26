# itw DSpark drafter for the DR. DOOM target with a 3x wider FFN, unannealed ("ffn3x")

**Target: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom` (Dr. Doom), not the 09.21 model.** `settings.sh` sets `MODEL` to it outright after sourcing the 09.21 itw settings and stops if it does not end in `-dr-doom`;
`vista/run_train.sbatch` prints `target model:` and the draft config's FFN width at the top of every job log. `MODEL` is the model vLLM serves for the hidden states, the trainer's `--verifier-name-or-path` and the one the card names.

Same as the Dr. Doom itw run ([`../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/): data = the Datakit SFT mixture's own text, no regeneration, 32,768-token rows, lr 3e-4, loss 0.1 CE + 0.9 TV, 12,288 anchors,
block 8, 3 layers, 32k draft vocabulary, seed) with ONE change: the drafter's FFN is `intermediate_size` 11,520 (3 x 3,840), 457,203,201 parameters instead of 280,256,001 (`draft_config.json`; verified identical to the itw one except that field).

## Schedule: unannealed, as long as we can afford it
The learning rate decays linearly over the FULL epoch (382,747 steps), as in the itw runs, so it is still about 80% of peak at step 80,813. The run is not stopped by a budget: it goes on for as many 12 h jobs as are submitted, as the itw runs did (80k steps, then
160k when there was time). Comparison points are the saved steps: the itw runs' own last steps are 80,813 (211 x 383) and 160,860 (420 x 383), and the checkpoint cadence (every 0.1% of the epoch = 383 steps) hits both. `settings.sh` keeps the last 8 checkpoints (`SAVE_TOTAL_LIMIT`); `vista/run_train.sbatch`
runs a small keeper that hard-links `checkpoint-80813` and `checkpoint-160860` into `runs/grug_dspark_itw_drdoom_ffn3x/keep/` as soon as they exist (`KEEP_STEPS`), so they survive the rotation (the trainer's rotation only removes its own names; checked with a test directory).

## Cost and what is known
Measured on the 09.21 version of this drafter (`../dspark-ffn3x_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw`, stopped at step 18,058, never served): 622 ms/step against 523 ms for the 280M drafter (+19%), so 80,813 steps need about 14 h, more than one 12 h job (two jobs for the 80k point, about 5 for 160k).
At equal steps its train accept length was +0.10 to +0.11 above the baseline's (3.909 against 3.797 at steps 17-18k); serving was not evaluated, so whether that carries over is open. Reference for this target: the Dr. Doom itw 80k drafter has train accept length 4.201 over its last 500 steps, serving acceptance 2.689 (RLVR) / 2.932 (Datakit pooled), 160k: 2.785 / 2.995.

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark-ffn3x_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
bash vista/submit_chain.sh --dry-run ; bash vista/submit_chain.sh         # one 12 h job on 2 gb nodes (CGAI-gb); JOBS=n adds n-1 afterany follow-ups; the gb QOS allows 3 submitted jobs
```
Training resumes from `runs/grug_dspark_itw_drdoom_ffn3x/checkpoints`; to go on, submit again (or `drive_chain.sh` of the itw folder). Fresh work dir `runs/grug_dspark_itw_drdoom_ffn3x`, data view `.../data` (the Datakit dataset is read only), W&B run = this folder's name in `atutej/marin_speculator`.
Evaluate as for the other Dr. Doom drafters (`tools/eval_acceptance`, RLVR 256 prompts and Datakit multi-turn 1,024, target Dr. Doom): copy the checkpoint into `runs/eval_ckpts/<name>/` first (config.json, config.py, model.safetensors); upload to the Hub only by hand (`HF_REPO=auto bash vista/run.sh upload` of the itw folder), public, name = experiment + date.

## Result
Ran one 12 h job (2026-10-24/25) to step 69,323 (181 x 383), evaluated at that step (Dr. Doom target): acceptance length 2.746 (RLVR) / 3.043 (Datakit first turn) / 2.580 (later turn), 3,681 / 4,220 / 2,555 tokens/s, about 1.83x / 1.97x / 1.57x over the earlier-session plain runs. Itw 80k: 2.689 / 2.983 / 2.522; itw 160k: 2.785 / 3.047 / 2.587. Card and raw results: `hf_card/`, `docs/results/dspark_eval/drdoom_ffn3x`. Not uploaded to the Hub; no job queued after the first.
