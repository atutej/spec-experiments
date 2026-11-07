# itw DSpark drafter for Dr. Doom: 6 Grug layers (FFN 7,936), FIVE taps (2, 7, 12, 17, 22), block size 8, 32,000-token draft vocabulary (one epoch of Datakit SFT)

Target: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom`. Data: the full Datakit SFT mixture, itw (the mixture's own replies, no regeneration), ONE epoch (382,747 steps, about 46B tokens), learning rate 3e-4 decayed linearly over the whole epoch. Written 2026-11-06 as a design step towards the drafter that maximises acceptance length (AL); settings come from the parameter-matched parent (`../dspark-grug-layers_..._itw`) through `settings.sh`, as in the 10-layer run.

## Why these choices
- **6 layers:** the 6-layer run (47,875 steps) beat the 10-layer run (34,853 steps) on the Datakit held-out set at 5 tokens (3.011 against 2.958) and costs less per step (0.89 s measured for the 6-layer run; about 1.5 s estimated for 10 layers).
- **Block 8:** the 6- and 10-layer runs used 5 and lose the positions 6 to 8, worth about 0.17 to 0.20 AL for the 8-token baselines (itw80k: 2.939 at 8 tokens, 2.765 at 5; grug 3-layer step 76,600: 2.991 and 2.796).
- **Five taps, 2 / 7 / 12 / 17 / 22 (+ final):** the 6- and 10-layer runs tapped 23, 24, 25 (late layers, redundant with the appended final output) and gained less than their size suggests. The earlier drafters tapped 2, 12, 22. This folder adds two layers in between, a superset of those taps.

## Configuration
| | 6-layer run (earlier) | this run |
|---|---|---|
| layers (Grug) / FFN | 6 / 7,680 | 6 / 7,936 |
| tapped target layers | 23, 24, 25 (+ final) | **2, 7, 12, 17, 22 (+ final)** |
| draft vocabulary | full (128,256) | 32,000 |
| block size | 5 | **8** |
| Markov head rank (vanilla head) | 256 | **1024** |
Everything else (data, seed, loss 0.1 CE + 0.9 TV, 12,288 anchors, 32,768-token rows, Markov head, `--draft-layer-type grug --grug-qk-mult 1.75`) as the parent.

## Size and latency (estimates from the 10-layer notes, not measured)
Layers 78.7M each, 6 layers about 472M; Markov head 164.1M at rank 1024 (W1 128,256 x 1024 = 131.3M, a table lookup, no per-step cost; W2 1024 x 32,000 = 32.8M, read every step; rank 256 was 41.0M); fc 32.8M (5 taps x 2,560 x 2,560; 19.7M with 3 taps): 669.1M trainable, counted by building the model on the meta device (the same count reproduces the 10-layer run's 847.7M). Frozen: embedding 328.3M, draft head 81.9M, verifier head copy 81.9M, 492.2M together; total 1,161.3M. The two extra taps add 13.1M and the larger W2 24.6M, together about 6% of the weights a draft step reads (about 620M). The trainer's hidden-state traffic grows with the number of slots (the 5 taps plus the final output).

## Markov head
Rank 1024 (user's choice, 2026-11-06). The DSpark paper (arXiv 2607.05147) uses rank 256 by default and, as far as I read, gives no rank sweep, so 1024 is untested ground. Head type stays `vanilla`: the paper reports the RNN head gives only marginal gains over the Markov head, mainly at longer proposal lengths, at higher implementation complexity; `gated` (bias gated by the backbone hidden state) has no published ablation.

## Reading the result
No run in this folder's family isolates the extra taps: the closest cleaner comparison is a 6-layer, block-8 run with taps 2, 12, 22 (not created). Against the earlier 6-layer run compare AL at 5 tokens (count the first 5 positions of `acceptance.csv`); that comparison mixes taps, block size, vocabulary and FFN width.

## Run it (from a LOGIN node)
```
cd .../experiments/dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw
bash vista/submit_chain.sh --dry-run
JOBS=1 bash vista/submit_chain.sh        # first job alone: check its first steps before queueing more
```
Work dir `runs/grug_dspark_itw_drdoom_grug-layers6-taps5` (override `GRUG_L6T5_WORK_DIR`). Training goes through the base itw experiment's `vista/run_2node_ddp.sh`. Evaluate with `tools/eval_acceptance/eval_matrix.sh` (`label=<checkpoint dir>@dspark@8+overlay`).

## Risks
- **Five taps are partly untested:** the model builds with 5 taps (fc input = number of taps x hidden size, checked on the meta device). Still to be confirmed by the first job: that the hidden-state server extracts layers 2, 7, 12, 17, 22 (+ final), and that vLLM takes the five ids from the checkpoint's `aux_hidden_state_layer_ids`.
- Grug-layer code is uncommitted in speculators and the vLLM fork (see the 10-layer notes).
- Checkpoint retention (`SAVE_TOTAL_LIMIT=3`, one milestone per 12 h job) will not keep checkpoints at fixed step counts; raise it before a scaling comparison.
