## Reproduce

Code and commands are in [`atutej/spec-experiments`](https://github.com/atutej/spec-experiments/tree/{GH_REF}) at commit `{GH_REF}`:

- Training and regeneration: [targetgen experiment folder](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen): settings, the regeneration, merge and training jobs (`vista/submit_chain.sh`), and `NOTES.md`.
- Regeneration and merge tool: [`tools/regenerate_token_rows.py`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/regenerate_token_rows.py).
- Trainer: [`atutej/speculators`](https://github.com/atutej/speculators/commit/99a50bf3977776a4a0f5548e5a25e28e6cdd396a) (branch `grug-dspark-gate`).
- Serving evaluation: [`tools/eval_acceptance/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance).
