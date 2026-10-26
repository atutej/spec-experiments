## Reproduce

Code and commands are in [`atutej/spec-experiments`](https://github.com/atutej/spec-experiments/tree/{GH_REF}) at commit `{GH_REF}`:

- Training: [experiment folder](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark-ffn3x_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw): `settings.sh`, `draft_config.json`, the training job (`vista/submit_chain.sh`) and `NOTES.md`. Same data view and recipe as the [Dr. Doom itw folder](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw).
- Trainer: [`atutej/speculators`](https://github.com/atutej/speculators/commit/99a50bf3977776a4a0f5548e5a25e28e6cdd396a) (branch `grug-dspark-gate`).
- Serving evaluation: [`tools/eval_acceptance/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance).
