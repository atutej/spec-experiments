# Naming: itw and targetgen (renamed 2026-10-16)

The drafter experiments were first called *offline* / *off-policy* and *online* / *on-policy*. Those words describe the training pipeline poorly, so they are renamed:

| New | Meaning | Old names |
|---|---|---|
| **itw** | in-the-wild: the drafter trains on text that already exists (a dataset's own replies, no regeneration) | offline, off-policy, `_offpolicy` |
| **targetgen** | target-generated: the drafter trains on the target model's own regenerated replies | online, on-policy, `_onpolicy` |

Not renamed, because they are different things: online hidden-state extraction and online training over Mooncake, W&B/HF offline modes, Luke's online EAGLE RL work.

## What was renamed

**Experiment folders** (`pipelines/speculator_training/experiments/XXX`):

| Old | New |
|---|---|
| `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_offpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_itw` |
| `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_offpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw` |
| `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_onpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen` |
| `dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy` | `dspark_qwen3_0_6b_nemotron-terminal-corpus_itw` |
| `dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy` | `dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen` |
| `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_offpolicy` | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_itw` |
| `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_onpolicy` | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_targetgen` |

**W&B runs** (project `atutej/marin_speculator`; run ids and URLs unchanged; the old name is kept in each run's notes):

| Run id | Old name | New name |
|---|---|---|
| 9z6aoy4y | `dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy` | `dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen` |
| whv5r2d4 | `dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy` | `dspark_qwen3_0_6b_nemotron-terminal-corpus_itw` |
| u9o0s9uf | `dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy_rollout_eval_0` | `dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen_rollout_eval_0` |
| ulugwbxq | `dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy_rollout_eval_0` | `dspark_qwen3_0_6b_nemotron-terminal-corpus_itw_rollout_eval_0` |
| m18iw5hf | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_offpolicy` | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_itw` |
| 876awhuu | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_offpolicy_rollout_eval_0` | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_itw_rollout_eval_0` |
| a3yo88jk | `dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy_rollout_eval_0_on_sft_target` | `dspark_qwen3_0_6b_nemotron-terminal-corpus_itw_rollout_eval_0_on_sft_target` |
| 3xwqx9my | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_onpolicy` | `dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_targetgen` |
| om7n1ryh | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_offpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_itw` |
| jgsst5k0 | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_offpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw` |
| 71rm9vy8 | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_offpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw` |
| ij7oistz | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_offpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw` |
| 6wnqmuwi | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_onpolicy` | `dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_targetgen` |

**Hugging Face repos** (renamed with HF's move, so the old URLs redirect; the label text in each model card was updated):

| Old | New |
|---|---|
| `atutej/dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_offpolicy-20261012` | `atutej/dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_itw-20261012` |
| `atutej/dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy-20261012` | `atutej/dspark_qwen3_0_6b_nemotron-terminal-corpus_targetgen-20261012` |
| `atutej/dspark_qwen3_0_6b_nemotron-terminal-corpus_offpolicy-20261012` | `atutej/dspark_qwen3_0_6b_nemotron-terminal-corpus_itw-20261012` |
| `atutej/dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_onpolicy-20261013` | `atutej/dspark_qwen3_0_6b-sft-nemotron-terminal-corpus_nemotron-terminal-corpus_targetgen-20261013` |
| `atutej/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_offpolicy-20261014` | `atutej/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-100k_itw-20261014` |

## Left as they are (old names still appear)

- On-disk run directories and checkpoints (`runs/grug_dspark_full`, `runs/grug_dspark_onpolicy`, `runs/nemotron_qwen3_0_6b_*_offpolicy_*` and `*_online_*`, `runs/eval_ckpts/{full_step80813,onpolicy_final,...}`): renaming them would break resume and the paths of finished runs. Scripts and docs keep these paths verbatim.
- Slurm job names and logs of jobs that already ran (`logs/slurm/...<jobid>.out`) and git history. Jobs submitted from now on use the new names (`dspark-itw-*`, `dspark-targetgen-*`).
- Permalinks pinned to commit `020c7f5` of this repo point to the tree as it was, with the old folder names.
- Future W&B and HF names follow the folder name (`RUN_NAME`, `HF_REPO=auto`), so they now end in `_itw` / `_targetgen`.
