# RL-prompt acceptance: Dr. Doom vs Antidoom RLVR1 Step12 as target (2026-10-26)

256 RL prompts (first 256 of `runs/nemotron_rlvr_prompts_1024`, Nemotron RLVR1 Cleaned-v3), T=1, 49,152 max new tokens, 102,400 context,
concurrency 16, probabilistic draft sampling with standard rejection, 8 speculative tokens. One run per cell; each job has its own plain arm.

Targets: `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom` and
`open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02` (RL from the Antidoom FTPO/DAPO step-45 export, an ancestor of Dr. Doom; not Dr. Doom plus RL).
Drafters (both trained against Dr. Doom): itw-80k (`drdoom_itw_step80813`) and targetgen-warm80k (`drdoom_targetgen_warm80k_final`).

| Target | Drafter | Acceptance length | Output tok/s | Speedup vs same-job plain | Mean tokens/reply | Hit cap |
|---|---|---:|---:|---:|---:|---:|
| Dr. Doom | plain | 1.000 | 1,996 | 1.00x | | |
| Dr. Doom | targetgen-warm80k | 2.853 | 3,924 | 1.97x | ~15,900 | 51/256 |
| Step12 | plain | 1.000 | 1,913 | 1.00x | | |
| Step12 | targetgen-warm80k | 2.907 | 3,699 | 1.93x | ~5,300 | 0/256 |
| Step12 (itw80k job) | plain | 1.000 | 1,841 | 1.00x | ~5,200 | 1/256 |
| Step12 (itw80k job) | itw-80k | 2.859 | 3,638 | 1.98x | ~4,900 | 0/256 |

Dr. Doom with itw-80k (earlier session, `../drdoom_itw_vs_warm80k/rlvr/session1`): AL 2.689, 1.77x.

Caveats: the three Step12/Dr. Doom jobs ran in different sessions (plain varies about 15% between sessions), so compare acceptance length across jobs and
speedup only within a job. Step12 replies are about a third as long as Dr. Doom's, which may affect acceptance. Single run per cell. Only `acceptance.csv`
and `results.json` are kept; completions stay in `runs/serve_eval_nemotron_rlvr_{drdoom_warm80k_20261026,antidoom_step12_warm80k,antidoom_step12_itw80k}`.
