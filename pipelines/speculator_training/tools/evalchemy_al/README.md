# Acceptance length on Marin eval-policy tasks (Evalchemy)

Measures the speculative-decoding acceptance length (AL) of a DSpark drafter on the Dr. Doom target over the tasks of the Marin eval policy (campaign
`eval-campaign-09-25-unlabeled`, Marin `75a5690b0f`), using the policy's own Evalchemy client at the pinned Evalchemy commit `c1d5c62cdc`. No generation code of ours:
`run_evalchemy_al.sbatch` serves the target with the drafter, then runs the policy's `marin/evaluation/evalchemy/client.py` (unmodified) once per task and records the
server's `/metrics` counters before and after (`../agentic_eval/spec_metrics.py`).

- `build_evalchemy_env.sbatch`: the pinned Evalchemy client environment (CPU only).
- `make_client_config.py`: policy task file -> the client's JSON config (chat template on, the task file's thinking setting, gen_kwargs, batch size, seed).
- `run_evalchemy_al.sbatch` / `submit_evalchemy_al.sh`: one drafter, one or more tasks, one or more seeds, one server per job.
- `collect_results.py`: result dirs -> `RESULTS.md` and `results.csv`.

Scores are blinded on purpose (Marin eval policy #10197): only the counters are kept; each task's Evalchemy output and log are deleted right after the task, and nobody reads
them. Sampling is whatever the policy sets (the task file's `gen_kwargs`, else the task's own default); `TEMPERATURE=` overrides it for every task.
The "policy" seed is the task file's own seed (none if it sets none); the second run uses `--seed 43`. GPQA-Diamond needs a Hugging Face token with access to the gated
dataset: it is read from `HF_TOKEN` in the submit environment, withheld from the vLLM server and given only to the client of the tasks in `HF_TOKEN_TASKS` (default gpqa-diamond).

Skipped on purpose: FinanceBench (needs a judge API key) and MRCR (long context).
