---
name: workspace-setup
description: Set up, repair, or extend the marin_speculator workspace (repos, conda envs, caches) on any machine (genai, TACC Vista gg/gh/gb nodes). Use on a fresh machine, after a $SCRATCH purge, when an env is broken or an import fails, before running an experiment on a machine for the first time, or whenever the project gains a dependency (a new repo, package, or build step).
---

# marin_speculator workspace setup

The workspace (`PROJECT_ROOT`) is a plain directory holding sibling repos plus caches and
run outputs. `spec-experiments` holds everything needed to rebuild it:

| File | Holds |
|---|---|
| `env.sh` | Per-machine settings (`MACHINE`, `CONDA_ROOT`, env location, caches, modules). Sourced by every script. |
| `setup/repos.txt` | Repo dependencies: `dir url ref [upstream]`. |
| `setup/envs/<env>.sh` | One recipe per conda env: `env_build` (install steps per `MACHINE`) and `env_check` (verify). |
| `setup/setup.sh` | Idempotent driver: clones missing repos, builds missing or failing envs, checks everything. |
| `setup/check_gpu.sh` | GPU smoke test of the serving stack (needs a GPU node). |
| `setup/locks/<machine>/<env>.txt` | Exact package lists of a known-good env (`pip freeze`). |
| `docs/vista_setup.md` | Machine notes and how the Vista setup was worked out. |

## Rebuild or check a workspace

```bash
# fresh machine or after a purge (on Vista: PROJECT_ROOT=$SCRATCH/marin_speculator)
git clone git@github.com:atutej/spec-experiments.git $PROJECT_ROOT/spec-experiments
bash $PROJECT_ROOT/spec-experiments/setup/setup.sh          # clone, build, check

bash setup/setup.sh --check     # report only; changes nothing
bash setup/setup.sh --gpu       # also the GPU smoke test (run on a GPU node)
```

`env.sh` exports `MACHINE` (genai or vista), `NODE_KIND` (Vista: `gg` CPU-only, `gh` 1× H100,
`gb` 4× GB200, else `gpu`, `cpu` or `login`) and `NUM_GPUS`. Check them first, and never assume
`nvidia-smi` works. Everything except `--gpu` runs on CPU-only nodes, including env builds.
Build on a compute node (gg is fine), never a login node. On Vista one env set must serve
both gh (`sm_90`) and gb (`sm_100`). `env_check` verifies torch's compiled archs
(`REQUIRED_CUDA_ARCHS`), and `--gpu` must pass on a gh **and** a gb node. `setup.sh`
never modifies a repo with local changes. `--update` only fast-forwards clean repos.
It ends with `READY` or `NOT READY: <items>`.

On Vista the user works in two kinds of session. Match the work to the node:
- **CPU session** (`gg` or login): code edits, commits (with permission), `--check`, and env
  builds (on `gg`, not login). No GPU work.
- **GPU idev session** (`gh` 1× H100 or `gb` 4× GB200, at most 2 h): `--gpu` and smoke
  tests. Check the time left (`squeue -u $USER`). **Full runs always go through `sbatch`**,
  possibly multi-node; their configuration is agreed with the user.

## Add or change a dependency

Do this whenever the project needs something new, so the next rebuild includes it:

1. **New repo:** add a line to `setup/repos.txt`. If the code needs it at a path, use
   `$PROJECT_ROOT/<dir>`.
2. **New package or build step:** add it to `env_build` in `setup/envs/<env>.sh`, in *every*
   `MACHINE` branch it applies to, and pin a version when it matters. Make `env_check`
   verify it, with an import or command plus a version or feature check, so a stale env
   fails the check instead of failing mid-experiment.
3. **New env:** copy an existing recipe to `setup/envs/<name>.sh`. `setup.sh` picks it up
   automatically.
4. **New machine:** add a branch in `env.sh` and in each recipe's `case "$MACHINE"`.
5. Apply it with `setup/setup.sh --rebuild <env>`, then `setup/setup.sh --check`.
6. Record the result with `setup/setup.sh --freeze`, which updates `setup/locks/<machine>/`.
7. Note anything non-obvious (workarounds, why a pin exists) in the recipe comments or
   `docs/vista_setup.md`.
8. Commit to `spec-experiments`, **asking the user first**.

## Rules

- Keep machine-specific paths and modules in `env.sh` and recipes, never in experiment scripts.
- Prefer fixing a recipe over hand-installing into an env. A hand fix is lost at the next purge.
- If a recipe step fails, report the error and options to the user. Don't swap in a different
  package source (for example upstream vLLM instead of the Marin fork) without asking.
