# targetgen drafter for the Dr. Doom target, warm-started from the itw checkpoint at step 80,813

The targetgen pipeline ([`../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen`](../dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen/): the target regenerates its own replies to Datakit conversations, then the drafter trains on them) with three differences:

1. **The target is Dr. Doom** (`open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`), NOT 09.21: the rollouts (regeneration), the hidden states during training, and the `lm_head` / gated final norm all come from Dr. Doom.
2. **Warm start** from the Dr. Doom-trained itw drafter at step 80,813 (`runs/eval_ckpts/drdoom_itw_step80813`, 3 layers, FFN 3,840, 280M parameters; on the RL prompts and the Dr. Doom target: acceptance length 2.697, 1.81x; trainer accept_len 4.201 over its last 500 steps). Training loads its weights (`--from-pretrained`) with a fresh optimizer and a fresh learning-rate schedule.
3. **Rows**: 250,000 conversations sampled uniformly from the Datakit rows of the itw run's steps **80,813 to 160,860** (about 1.37M rows), none of which its steps 0 to 80,813 saw (the pool is built from the trainer's batch order, `tools/eval_acceptance/itw_seen_rows.py --start-steps`, and checked disjoint from `runs/itw_seen_rows/rows_step80813.npy`).

## Stages (`vista/submit_chain.sh`, run from a login node)
`regen` (2 gb nodes, 12 h: the batch node first builds the row pool, then each node serves Dr. Doom and regenerates its half of the 250k rows; resumable) -> optionally a second `regen` job (`REGEN_JOBS=2`, the default; it exits quickly when nothing is left) -> `train` (2 gb nodes, 12 h, afterok regen; first the merge of the parts into the dataset, then the itw folder's `run_2node_ddp.sh train` with this folder's settings).

## The draft vocabulary (`d2t` / `t2d`), and why a warm start must keep the checkpoint's
The drafter predicts over 32,000 of the target's 128,256 tokens (the most frequent supervised tokens of its training data). `d2t` (32,000 entries) lists which target token each draft index is; `t2d` (128,256 booleans) marks the tokens that are in the vocabulary. The `lm_head` rows, the Markov head and the learned weights are all indexed by draft index.
The trainer builds these maps from the training data's token frequencies, and with `--from-pretrained` it still passes the maps built from the NEW data into the loaded model, not the checkpoint's. The regenerated rows give a different top 32,000 (for 09.21's targetgen data the vocabulary differed from the itw one in 1,269 of 32,000 tokens, and the arrays did not line up), so the loaded weights would be misaligned.
The merge step therefore writes the CHECKPOINT's own `d2t` / `t2d` (read from its `model.safetensors`, identical in values and dtype to the trainer's cache files, checked) into the data directory (`regenerate_token_rows.py merge --vocab-from-checkpoint`), where the trainer finds them as `d2t-32000.npy` / `t2d-32000.npy`. Nothing is copied by hand, and the merge prints how many tokens the vocabulary built from Dr. Doom's replies would have differed in, and how much of the new text the kept vocabulary covers.

## Time (an estimate, not a measurement for Dr. Doom)
On 09.21: 270k rows took about 8.7 h to regenerate on 2 nodes (4,413 generated tokens per row), and training on them about 2.6 h. Dr. Doom's replies are longer (about 1.3x on the RL prompts), so for 250k rows: about 9.5 to 11 h regeneration + about 2.4 h training. A little over 12 h in total is expected, and a regeneration that overruns one job is why `REGEN_JOBS=2` exists.

## Training recipe
The targetgen recipe unchanged: lr 3e-4 peak with 1% warmup and linear decay to zero over the epoch over the regenerated rows (about 16k steps), loss 0.1 CE + 0.9 TV, 12,288 anchors, 32,768-token rows, 0.5% of the rows held out for validation. The itw checkpoint ended at about 2.4e-4, so a 3e-4 start is not a large jump; lower it in `settings.sh` (`LR`, from the itw settings) if the first steps look disturbed.

## Guards
`settings.sh` assigns every fresh path outright after sourcing the itw settings (work dir `runs/grug_dspark_drdoom_targetgen_warm80k`, which did not exist) and recomputes `LOGGER_ARGS`. Nothing writes into any other run's directory; the merge refuses a directory it did not make. The warm-start checkpoint and the Datakit dataset are read only.
NOT yet tested: the warm start itself (`--from-pretrained` with this data), the pool stage, and the merge with `--vocab-from-checkpoint`. A short smoke run would check all three (see below).

## Evaluate afterwards
Copy the final checkpoint into a new `runs/eval_ckpts/<name>/`; serve on **Dr. Doom** with the 256 RL prompts and the usual settings. References on the same target and prompts: the itw drafter at step 80,813 (2.667-2.697 acceptance length, 1.79-1.81x), the targetgen drafter trained on 09.21 (2.465, 1.62x), and the itw drafter at step 160,860 (not yet evaluated: the control for "just more itw training"). Single runs differ by about 0.02.
