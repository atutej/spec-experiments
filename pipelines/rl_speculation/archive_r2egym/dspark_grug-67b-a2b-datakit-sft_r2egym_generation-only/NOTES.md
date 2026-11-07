# Generation-only R2E-Gym rollouts with the DSpark drafter (no trainer)

Stage between the serving smoke test and the RL A/B. Real Terminus-2 agent rollouts on R2E-Gym tasks in Daytona sandboxes, Datakit-SFT Grug target, MarinSkyRL
`5de48d34` entrypoint `terminal_bench_generate`, one Vista GB200 node (4 GPUs, one vLLM engine DP4 x EP4, vLLM 39e62869), 2026-10-14. Configs:
`../../configs/1node_grug_datakit_{dspark,plain}_r2egym_generate.yaml` (identical except `speculative_config`); launched with `tools/launch_ot_rl.sh <yaml> <job> --run`.
Setup: 16 R2E-Gym tasks x 2 samples = 32 trials, 16 concurrent, T=1.0, at most 12 turns, request window 32768 with 8192 new tokens per turn. Unadapted drafter
(`runs/grug_dspark_100k/checkpoints/0`, 8 draft tokens). One run per arm, no seeds, no repeats: all numbers below are single measurements.

## Results (achieved)
| | dspark | plain |
|---|---|---|
| trials finished | 32/32 | 32/32 |
| stop reasons | 2 complete, 30 turn cap | 4 complete, 28 turn cap |
| batch wall time | 280.5 s | 194.4 s |
| approx generated tokens (engine logs, 5 s windows) | 141,944 | 147,444 |
| engine-active rank-windows (5 s each) | 68 | 97 |
| per-stream decode, 1 running request on the rank | 250 tok/s | 165 tok/s (1.52x) |
| per-stream decode, 2 running | 274 tok/s | 146 tok/s (1.88x) |
| per-stream decode, 3 running | 181 tok/s | 117 tok/s (1.55x) |
| per-stream decode, 4 or more running | 123 tok/s (3 windows) | 95 tok/s (1.29x) |

- Pooled accept length (dspark, 68 windows, 58,311 drafts): **2.438**; per-position acceptance 0.523, 0.327, 0.211, 0.142, 0.091, 0.064, 0.046, 0.034 (`tools/parse_specdec.py`).
  Individual 5 s windows ranged from 1.15 to 3.56.
- Daytona sandboxes, Terminus-2 turns and the verifier worked end to end (first run to reach rollouts on this machine). All rewards were 0 at the 12-turn cap; this is a speed measurement, not a quality one.

## How to read this
- Decode itself is faster with the drafter (per-stream speed 1.3x to 1.9x at matched concurrency, and about 30% fewer engine-active windows for a similar token total), but the
  **batch wall time was longer** (280 s vs 194 s, plain faster by 31%). The wall time includes sandbox start, tool execution and verification, which the engine cannot speed up,
  and the two arms sampled different trajectories (30 vs 28 trials hit the turn cap). With two waves of 16 trials and one run each, the wall-time difference may be noise from those
  parts; this run cannot tell. Do not quote 280 s vs 194 s as a speed-up or a slow-down of the method.
- Concurrency per engine rank was only 1 to 3 requests (16 streams over 4 DP ranks), so the 1.5x to 1.9x per-stream figures are low-batch numbers, where drafting helps most.
  Luke's per-stream numbers (1.5x to 1.61x) were at 16 streams on a node as well but with one engine per node and a different drafter and target (issue #9114, Jupiter GH200).
- The first plain launch ran the full 16,202-trial dataset because only the DSpark config had been switched to the 16-task subset; it was stopped about a minute in. Up to 16 Daytona
  sandboxes may have been left running from it (the Daytona SDK in the launcher env could not list them: check the dashboard).

## Next
2 or more nodes for the RL A/B (`../dspark_grug-67b-a2b-datakit-sft_r2egym-rl-ab/vista/submit_ab.sh`); repeat this measurement with more trials and 2 to 3 repeats per arm if a wall-time claim is needed.
