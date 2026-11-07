# R2E-Gym stages (archived)

R2E-Gym RL is on hold: the shared Daytona org's snapshot quota (40, holding 48) blocked registering the 11 per-repo snapshots, and declarative builds are not allowed. These folders are kept for when it resumes. The runs we have results for are in `../experiments/` (Reasoning Gym, text-to-SQL).

| Folder | What it is | Status |
|---|---|---|
| `dspark_grug-67b-a2b-datakit-sft_r2egym_serving-smoke` | Serve the Datakit-SFT target with and without the drafter, no RL | run once, 2026-10-14 |
| `dspark_grug-67b-a2b-datakit-sft_r2egym_generation-only` | Real agent rollouts on R2E-Gym tasks with the drafter, no trainer | run, 2026-10-14 |
| `dspark_grug-67b-a2b-datakit-sft_r2egym-rl-ab` | 2-node RL, drafter vs plain, colocated and split layouts | layouts exercised, no with/without comparison |
| `dspark-grug-layers_grug-67b-a2b-dr-doom_r2egym-rl-split` | Planned 64-step split RL, Dr. Doom target, Grug-layer drafter (needs a vLLM overlay) | planned, not run |
| `dspark-targetgen-warm80k_grug-67b-a2b-dr-doom_r2egym-rl-split` | Planned 64-step split RL, Dr. Doom target, targetgen warm80k drafter | planned, not run |
| `dspark-drdoom_r2egym-async-replica` | Snapshot-registration tool, image list and digests for the reference setup's 11 per-repo images | registration failed on the quota |

The shared R2E-Gym configs stay in `../configs/`, the prompt builder in `../tools/build_r2egym_prompts.py`.
