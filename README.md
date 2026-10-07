# spec-experiments

Experiments on speculative decoding and the models around it, run on the UT ECE `genai` machines
and TACC Vista. The repo holds the glue (scripts, notebooks, setup tooling); the libraries live in
sibling repos.

## Layout

```
env.sh                 detects the machine, sources its settings, sets caches; sourced by every script
setup/                 rebuild the workspace: repos.txt, conda env recipes, locks, setup.sh
  machines/<name>/       one folder per machine: env.sh, NOTES.md, slurm/ (env-build wrappers)
pipelines/             one folder per family of experiments
  speculator_training/   train drafters (DSpark etc.) on on-policy data with the speculators fork
    experiments/<name>/  NOTES.md, genai/run.sh, vista/ (smoke tests, sbatch)
    tools/  notebooks/
  <future family>/       same shape, e.g. SFT of a base model
common/                helpers shared by more than one family
docs/                  repo-wide guides (vista_setup.md: porting to Vista)
.claude/skills/        workspace-setup skill for Claude Code
```

Workspace (this repo is a sibling of the others; `env.sh` takes `PROJECT_ROOT` as its parent):

    <workspace>/spec-experiments/  speculators/  vllm/  envs/  cache/  runs/  logs/  tmp/

## Start

```bash
git clone git@github.com:atutej/spec-experiments.git <workspace>/spec-experiments
bash <workspace>/spec-experiments/setup/setup.sh --check   # report; drop --check to build
```

Run outputs go to `<workspace>/runs/<name>/` (their logs in `runs/<name>/logs/`), and logs of setup, Slurm jobs
and smoke tests go to `<workspace>/logs/{setup,slurm,smoke}/`. Neither goes into this repo. Read a family's `README.md`
for the envs it needs, and an experiment's `NOTES.md` for its settings and reference results.
