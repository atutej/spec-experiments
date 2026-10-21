## Reproduce

Code and commands are in [`atutej/spec-experiments`](https://github.com/atutej/spec-experiments/tree/{GH_REF}) at commit `{GH_REF}`:

- Training: [`itw` experiment folder for the Dr. Doom target](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw): the itw recipe with the Dr. Doom target and fresh paths (`settings.sh`, `NOTES.md`); the run was resumed from its step-80,813 checkpoint for a second 12 h.
- Data: the Datakit dataset is built by the scripts in [`data/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft_datakit-sft-mix_itw/data) (see the manifest above).
- Trainer: [`atutej/speculators`](https://github.com/atutej/speculators/commit/99a50bf3977776a4a0f5548e5a25e28e6cdd396a) (branch `grug-dspark-gate`).
- Serving evaluation: [`tools/eval_acceptance/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance).
