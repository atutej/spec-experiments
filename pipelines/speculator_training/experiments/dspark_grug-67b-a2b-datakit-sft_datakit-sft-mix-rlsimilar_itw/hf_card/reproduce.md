## Reproduce

Code and commands are in [`atutej/spec-experiments`](https://github.com/atutej/spec-experiments/tree/{GH_REF}) at commit `{GH_REF}`:

- Training: [`rlsimilar` experiment folder](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix-rlsimilar_itw): settings in `settings.sh`, the data job (`vista/build_data.sbatch`) and the training job (`vista/run_train.sbatch`); `NOTES.md` has the plan and the run commands.
- Data selection: [`tools/select_similar_rows/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/tools/select_similar_rows) (embed, select, write) on the Datakit dataset built by the scripts in the [`itw` experiment's `data/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data).
- Trainer: [`atutej/speculators`](https://github.com/atutej/speculators/commit/99a50bf3977776a4a0f5548e5a25e28e6cdd396a) (branch `grug-dspark-gate`).
- Serving evaluation: [`tools/eval_acceptance/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance). The write-up of the itw and targetgen drafters (methodology, other targets and prompt sets): [marin-community/marin#10227](https://github.com/marin-community/marin/issues/10227).
