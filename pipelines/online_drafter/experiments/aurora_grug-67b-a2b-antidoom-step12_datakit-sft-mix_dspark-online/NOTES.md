# Aurora online DSpark drafter, target Antidoom RLVR1 Step12, Datakit SFT prompts, warm start from targetgen warm80k

**Target: `open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02`**; **drafter start: `runs/eval_ckpts/drdoom_targetgen_warm80k_final`**. Same as `../aurora_grug-67b-a2b-dr-doom_datakit-sft-mix_dspark-online`
(Datakit shard 0 prompts, 256 new tokens, lr 1e-5 with 300 warmup steps, 32 samples per optimizer step, sync every 10 steps) with the target swapped (`target_model_path` in config.yaml, `TARGET=` in run_long.sbatch).

Run (login node): `sbatch run_long.sbatch`. Run dir `runs/aurora_antidoom_step12_datakit_warm80k_20261026/long_<jobid>`; W&B `marin_aurora`, group `dspark-step12-warm80k-lr1e-5-datakit-online-<jobid>`.

Baseline still to measure: pristine warm80k on Step12 with the Datakit multi-turn 1,024 val prompts (`build_val_prompts.py`, as `serve_eval_val_multiturn1024_*`); the Dr. Doom number was 2.595 later-turn / 3.091 first-turn.
Not measured yet: this draft on Datakit prompts. Only the RLVR-prompt eval below has been run.

## Result (job 1103012, 2026-10-26 20:15, ran the full 12 h to TIMEOUT; last draft `drafts/step_00007650`, copied to `runs/eval_ckpts/antidoom_step12_datakit_warm80k_online_step7650`)
Eval on RLVR prompts: 256 prompts (first 256 of `runs/nemotron_rlvr_prompts_1024`), T=1, 49,152 new tokens, window 102,400, probabilistic draft sampling with standard rejection, one job per target with its own plain arm
(RLVR-online arm from `../aurora_grug-67b-a2b-antidoom-step12_rlvr-train_dspark-online` shown for comparison). One run per cell.
| Target | Arm | Acceptance length | Output tok/s | Speedup vs plain |
|---|---|---:|---:|---:|
| Step12 (job 1103960) | plain | 1.000 | 1,818 | 1.00x |
| | warm80k (control) | 2.929 | 3,397 | 1.87x |
| | online RLVR | 2.968 | 3,910 | 2.15x |
| | **online Datakit (this run)** | **2.855** | 3,542 | 1.95x |
| Marin Snowball v0 (job 1107636) | plain | 1.000 | 1,768 | 1.00x |
| | warm80k (control) | 2.874 | 3,496 | 1.98x |
| | online RLVR | 2.821 | 2,277 | 1.29x |
| | **online Datakit (this run)** | **2.907** | 3,426 | 1.94x |

Reading: on RLVR prompts with its own target (Step12) the Datakit-online draft is -0.074 AL against warm80k, larger than the run-to-run spread (about 0.02; an earlier warm80k run gave 2.907 vs 2.929), so likely a real loss:
adapting to Datakit prompts does not help on RLVR prompts. On Marin Snowball v0 (never trained against; `Campaign3 dare-linear-02` merge) it is +0.033, inside the noise, so the sign flips between targets and neither number supports a claim.
The v0 RLVR-online tok/s anomaly is discussed in the RLVR experiment's NOTES.md. Raw results: `runs/serve_eval_nemotron_rlvr_antidoom_step12_online_vs_warm80k` and `runs/serve_eval_nemotron_rlvr_marin_snowball_v0_online_vs_warm80k`.
