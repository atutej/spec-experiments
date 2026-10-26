# itw DSpark drafter for the DR. DOOM target: 3x wider FFN AND twice the layers ("ffn3x-layers2x"), unannealed

**Target: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom` (Dr. Doom).** `settings.sh` sets `MODEL` to it outright and stops if the name does not end in `-dr-doom`; `vista/run_train.sbatch` prints `target model:` and the draft shape at the top of every job log.

Same as [`../dspark-ffn3x_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw`](../dspark-ffn3x_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw/) (which is the Dr. Doom itw recipe with a 3x FFN: Datakit text, 32,768-token rows, lr 3e-4, loss 0.1 CE + 0.9 TV, 12,288 anchors,
block 8, 32k draft vocabulary, seed, full-epoch linear learning-rate schedule, no stop step) with ONE more change: `num_hidden_layers` 6 instead of 3 (`draft_config.json` differs from that folder's only in this field). Shapes: hidden 2,560, FFN 11,520, 20 heads / 5 KV heads, 6 layers, tapped target layers 2, 12, 22.

Trainable parameters, by arithmetic from the checkpoint shapes (not instantiated): per layer about 104.9M (attention 16.4M + MLP 88.5M), 6 layers 629M, plus about 142.5M outside the layers (fc, lm_head over 32k, Markov head): about **772M**, against 457M for the 3-layer ffn3x drafter and 280M for the itw one.
Step time, MEASURED in job 1099865 (2026-10-25, near step 1,090): 1.15 s/step (forward 323 ms, backward 772 ms, optimizer 40 ms), against 523 ms for the itw drafter and 622 ms for the 3-layer ffn3x one: about 37k steps per 12 h job, so about 26 h (three jobs) to step 80,813 and about 51 h (five jobs) to 160,860. No out-of-memory with twice the layers at 12,288 anchors per 32k row.

## Schedule
UNANNEALED, as the other itw runs: the rate decays over the full epoch (382,747 steps) and the run goes on as long as jobs are submitted (`JOBS=n`, or resubmit; training resumes from `runs/grug_dspark_itw_drdoom_ffn3x_l6/checkpoints`). Checkpoints every 383 steps, the last 8 kept, and the comparison steps 80,813 and 160,860 are hard-linked into `.../keep/` by the wrapper (`KEEP_STEPS`).

## Run it (from a LOGIN node)
```
cd spec-experiments/pipelines/speculator_training/experiments/dspark-ffn3x-layers2x_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
bash vista/submit_chain.sh --dry-run ; bash vista/submit_chain.sh      # one 12 h job on 2 gb nodes (CGAI-gb); JOBS=n adds afterany follow-ups (the gb QOS allows 3 submitted jobs)
```
Evaluate as the other Dr. Doom drafters (`tools/eval_acceptance`: RLVR 256 prompts, Datakit multi-turn 1,024), after copying a checkpoint into `runs/eval_ckpts/<name>/` (config.json, config.py, model.safetensors). References (acceptance length, target Dr. Doom): itw 80k 2.689 / 2.932 (RLVR / Datakit pooled), itw 160k 2.785 / 2.995,
ffn3x (3 layers) at step 69,323 2.746 / 2.989.
