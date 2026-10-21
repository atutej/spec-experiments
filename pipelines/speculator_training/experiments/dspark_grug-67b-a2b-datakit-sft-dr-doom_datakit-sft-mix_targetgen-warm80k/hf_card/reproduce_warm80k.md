## Reproduce

Code and commands are in [`atutej/spec-experiments`](https://github.com/atutej/spec-experiments/tree/{GH_REF}) at commit `{GH_REF}`:

- Training and regeneration: [warm-start targetgen experiment folder](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/experiments/dspark_grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_targetgen-warm80k): settings, the regeneration, merge and training jobs, and `NOTES.md` (including why a warm start has to keep the checkpoint's draft vocabulary).
- Regeneration and merge tool: [`tools/regenerate_token_rows.py`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/regenerate_token_rows.py); the row pool of steps 80,813 to 160,860: [`tools/eval_acceptance/itw_seen_rows.py`](https://github.com/atutej/spec-experiments/blob/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance/itw_seen_rows.py).
- Trainer: [`atutej/speculators`](https://github.com/atutej/speculators/commit/99a50bf3977776a4a0f5548e5a25e28e6cdd396a) (branch `grug-dspark-gate`).
- Serving evaluation: [`tools/eval_acceptance/`](https://github.com/atutej/spec-experiments/tree/{GH_REF}/pipelines/speculator_training/tools/eval_acceptance).
