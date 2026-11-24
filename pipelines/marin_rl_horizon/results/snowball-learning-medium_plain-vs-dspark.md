# snowball learning-medium: plain vs frozen DSpark draft

25 RL updates of Marin's `learning-medium` Snowball preset on Horizon, without and with the frozen DSpark draft (marin-community/marin#11975).
All numbers below are **achieved**, one run per arm, measured 2026-11-24. Both arms ran at the same time on different nodes, with the same runtime archive
(plain archive + the vLLM 20261118 overlay), the same seed and the same prompt order.

**Config** (`launchers/learning-medium-draft` is the DSpark arm and `launchers/learning-medium-plain` the control; both carry the recipe): Grug-67B-A2B Datakit-SFT initializer (`fe666478`), 10 nodes x 4 GB200
(8 policy nodes Megatron TP1/PP2/EP8 + 2 engine nodes, one DP8/EP8 vLLM engine), 128 prompts x 4 replies per update, replies stop naturally (up to 8192 new tokens),
zero-reward groups filtered, no evaluations (eval-lite is in Marin object storage), no checkpoints, 4 draft proposals, `gpu_memory_utilization` 0.75, up to 1024 concurrent
tasks. Prompts are the stand-in mix. W&B: `atutej/marin_rl_new`, runs `attempt-20261124-114647` (plain, job 105248) and `attempt-20261124-122709` (DSpark, job 105301).

| steps 2-25 | plain | DSpark |
|---|---|---|
| total time | 51.0 min | 28.1 min |
| median / mean step | 106 s / 127 s | 57 s / 70 s |
| steps limited by rollouts (over 50 s) | 16 of 24 | 13 of 24 |
| mean of those steps | 174 s | 99 s |
| median of the other steps | 34 s | 35 s |
| step 1 (includes startup) | 562 s | 341 s |
| mean acceptance length at the end | n/a | 2.75 tokens per round |

**Why the gain is not uniform.** Rollouts overlap training, so a step takes the longer of the learner (about 32 s: 25 s policy training + 6 s log-probs, the same every step)
and the wait for rollout groups (`timing/step_wall/group_admission`). In the plain run a 314 s step spent 278 s waiting; the 34 s steps waited about 0. The draft only helps
the waiting steps, so learner-bound steps are the same in both arms.

**Load the engines saw** (sampled vLLM log lines, so a rough picture, not a sweep): running requests per engine median 27 (plain) / 25 (DSpark), 90th percentile 51 / 49;
generation speed per active engine median 2.2k / 4.0k tokens/s. This is the range where a one-layer EAGLE3 draft on the Mochi long loop fell behind plain decoding
(penfever's measurement on marin-community/marin#11975); the DSpark draft did not.

**Reward:** mean reward rises in both arms (mean of the first five and last five updates: 0.57 -> 0.69 plain, 0.58 -> 0.68 DSpark); 86% of groups have identical rewards in both.
Rewards are on the stand-in prompts and not comparable to Marin's.

**Startup was slow and uneven** in both arms (about 20 and 23 minutes to engines ready, against about 5 in a solo run), because two 10-node jobs read the 17.5 GB runtime archive
and the 67B weights from the shared NFS `/scratch` at once. Slowest weight load seen: about 116 s (plain) vs about 857 s (DSpark). A first DSpark attempt (job 105243) hung for 22
minutes with four engines on one node in disk wait and was cancelled; the cause (contention or that node) was not isolated. Fixes since: `sbcast` of the archive, a per-node
read preflight and a 1800 s engine init timeout (`setup/machines/horizon/NOTES.md`). They passed the backend's tests but had not run in a real job at the time of these runs.

**Not covered:** one run per arm, so the exact ratio is uncertain; one concurrency range; the stand-in prompts; no evaluation of the resulting policy.

**Reproducing it now:** a runtime built from a marin lockfile at or after 2026-11-24 needs no overlay (see the pipeline README). Then `launchers/learning-medium-plain/launch.sh` and `launchers/learning-medium-draft/launch.sh` (no `--drafter`: the preset's own drafter), with `WANDB_ENTITY` and `WANDB_PROJECT` exported to log to W&B, each with `--validate` first. The numbers above are from the runs as they were, on the overlay archive.
