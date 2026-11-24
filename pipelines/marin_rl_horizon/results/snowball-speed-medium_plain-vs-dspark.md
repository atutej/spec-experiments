# snowball speed-medium: plain vs frozen DSpark draft

Update time of Marin's `speed-medium` Snowball preset on Horizon, without and with the frozen DSpark draft (marin-community/marin#11975).
All numbers below are **achieved**, one run per arm, measured 2026-11-24.

**Config** (`launchers/speed-medium-draft` is the DSpark arm and `launchers/speed-medium-plain` the control; both carry the recipe): Grug-67B-A2B Datakit-SFT initializer
(`fe666478`), 10 nodes x 4 GB200 (8 policy nodes Megatron TP1/PP2/EP8 + 2 engine nodes, one DP8/EP8 vLLM engine), 10 RL updates, 128 prompts x 4
replies, every reply forced to 4096 tokens (`ignore_eos`), no zero-reward-group filter, no evaluations, no checkpoints, 4 draft proposals,
`gpu_memory_utilization` 0.75. Prompts are the stand-in mix (see the pipeline README), so rewards are meaningless here: every group had the same reward,
which is by design for a throughput preset.

| | plain | DSpark |
|---|---|---|
| job | 105043 | 105114 |
| median update, steps 2-10 | 76 s | 50 s (about 34% lower) |
| step 1 (includes startup) | 309 s | 261 s |
| job wall time | 25 min 10 s | 22 min 6 s |
| mean acceptance length / rate | n/a | 2.5 tokens per round / 0.38 |

The ratio matches the H100 numbers in marin-community/marin#11975 (85.3 s -> 56.5 s, 34%).

**Not controlled:** the plain arm ran on the old runtime archive and marin code from before #11975; the DSpark arm ran on the archive with the vLLM
20261118 overlay and #11975 applied. The overlay also brings a tuned GB200 MoE kernel config, so part of the gap could come from it. The learning-medium
comparison (next folder) uses the same archive in both arms.

**Reproducing it now:** a runtime built from a marin lockfile at or after 2026-11-24 needs no overlay (see the pipeline README). Then `launchers/speed-medium-plain/launch.sh` and `launchers/speed-medium-draft/launch.sh` (no `--drafter`: the preset's own drafter), each with `--validate` first. The numbers above are from the runs as they were.
