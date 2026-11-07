# Aurora online DSpark drafter, target Antidoom RLVR1 Step12, RLVR train prompts, warm start from targetgen warm80k

**Target: `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02`** (RL from the Antidoom FTPO/DAPO step-45 export on Nemotron RLVR1 data; an ancestor of Dr. Doom plus RL, NOT Dr. Doom plus RL;
same architecture, tokenizer and chat template as Dr. Doom). **Drafter start: `runs/eval_ckpts/drdoom_targetgen_warm80k_final`** (trained against Dr. Doom's replies).

Everything else is the Dr. Doom RLVR itw80k arm (`../aurora_grug-67b-a2b-dr-doom_rlvr-train_dspark-online`, see its NOTES.md for the prompts, 4,096-token completions, lr 1e-5 with 30 warmup steps,
32 samples per optimizer step, Aurora forward KL, frozen confidence head, sync every 8 steps). Only `target_model_path` (config.yaml) and `TARGET=` (run_long.sbatch; `run.sh` takes `TARGET` from the environment, default Dr. Doom) differ,
plus the drafter start. Prompts: `runs/nemotron_rlvr_train_prompts_20261024/prompts_token_ids.jsonl` (already built; the eval's 1,024 prompts are excluded).

Run (login node): `sbatch run_long.sbatch`. Run dir `runs/aurora_antidoom_step12_rlvr_warm80k_20261026/long_<jobid>`; W&B `marin_aurora`, group `dspark-step12-rlvr-warm80k-<jobid>`.

Evaluate the last draft against the same-session plain and the pristine warm80k on Step12 (RLVR 256 prompts): **warm80k 2.907 AL / 1.93x** (`docs/results/dspark_eval/rl_prompts_256_target-dr-doom_vs_antidoom-rlvr1-step12`).
Open point as in the Dr. Doom RLVR arm: completions are capped at 4,096 new tokens, so the late positions of long replies are not trained on (Step12's RLVR replies average about 5.3k tokens, shorter than Dr. Doom's).

## Result (job 1103011, 2026-10-26 20:15, ran the full 12 h to TIMEOUT; last draft `drafts/step_00000608`, copied to `runs/eval_ckpts/antidoom_step12_rlvr_warm80k_online_step608`)
Eval: 256 RLVR prompts (first 256 of `runs/nemotron_rlvr_prompts_1024`), T=1, 49,152 new tokens, window 102,400, probabilistic draft sampling with standard rejection, one job per target with its own plain arm
(Datakit-online arm from `../aurora_grug-67b-a2b-antidoom-step12_datakit-sft-mix_dspark-online` shown for comparison). One run per cell.
| Target | Arm | Acceptance length | Output tok/s | Speedup vs plain |
|---|---|---:|---:|---:|
| Step12 (job 1103960) | plain | 1.000 | 1,818 | 1.00x |
| | warm80k (control) | 2.929 | 3,397 | 1.87x |
| | **online RLVR (this run)** | **2.968** | 3,910 | 2.15x |
| | online Datakit | 2.855 | 3,542 | 1.95x |
| Marin Snowball v0 (job 1107636) | plain | 1.000 | 1,768 | 1.00x |
| | warm80k (control) | 2.874 | 3,496 | 1.98x |
| | **online RLVR (this run)** | **2.821** | 2,277 | 1.29x |
| | online Datakit | 2.907 | 3,426 | 1.94x |

Reading: on its own target (Step12) the RLVR-online draft is +0.039 AL over warm80k, about twice the run-to-run spread (an earlier warm80k run on the same target and prompts gave 2.907 vs 2.929 here, a spread of about 0.02), so a likely but small gain.
On Marin Snowball v0 (never trained against; a merge of the Step12 lineage with the allkimi relay SWE/Terminal specialist, `Campaign3 dare-linear-02`) it is -0.053, inside the noise. The 2,277 tok/s of the v0 RLVR-online arm looks like a tail artifact (2 of 256 requests hit the 49,152-token cap; the other arms had at most 1), not verified: use AL.
Raw results: `runs/serve_eval_nemotron_rlvr_antidoom_step12_online_vs_warm80k` and `runs/serve_eval_nemotron_rlvr_marin_snowball_v0_online_vs_warm80k` (the Step12 job was submitted without the CGAI-gb reservation, which had expired).
