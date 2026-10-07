# speculator_training

Train speculative-decoding drafters with the `speculators` fork: regenerate on-policy responses
with vLLM, build hidden-state data, train online over Mooncake.

- **Envs:** `vllm` (serving: regeneration, hidden-state server) and `speculators` (data prep,
  training, analysis). Recipes in `setup/envs/`.
- **Repos:** `speculators`, `vllm` (Marin fork, built from source on Vista). See `setup/repos.txt`.
- **`experiments/<name>/`:** `genai/run.sh` is the genai launch script; `vista/` has Vista smoke tests
  and sbatch scripts; `NOTES.md` has settings, learnings and reference results.
- **`tools/`:** `export_registry_dataset.py` (seeded sample of a dataset preset as JSONL).
- **`notebooks/`:** `dspark_from_scratch*.ipynb`, walkthroughs of the drafter and its training.

Experiments: `dspark_qwen3_0_6b_nemotron_terminal`, `dspark_qwen3_0_6b_perfectblend`.
