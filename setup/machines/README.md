# Machines

One folder per machine: `setup/machines/<name>/`. Adding a cluster means adding a folder here, with no
edits to shared code. Currently `genai` and `vista`.

## What a machine folder holds

| Path | Holds |
|---|---|
| `env.sh` | The machine's settings. Sourced by the root `env.sh` (so by every script). |
| `NOTES.md` | Access, filesystems, purge policy, scheduler and node types, gotchas. Optional. |
| `slurm/` | Wrappers to build envs or run jobs on that machine (only where there's a scheduler). |
| `envs/<env>.sh` | `env_build`: the install steps for that conda env on this machine. (`env_check` is shared, in `setup/envs/<env>.sh`.) |
| `locks/<env>.txt` | `pip freeze` of the known-good env, written by `setup.sh --freeze`. |

## The `env.sh` contract

The root `env.sh` picks the machine: an explicit `MACHINE=<name>` wins; otherwise the first machine
whose `MACHINE_HOSTNAME_REGEX` matches `hostname -f`; otherwise the one with `MACHINE_FALLBACK=1`
(`genai`). It then sources that machine's `env.sh` and fails with a message if a required variable is
missing.

Required:

| Variable | Meaning |
|---|---|
| `MACHINE` | The folder name. Recipes branch on it. |
| `CONDA_ROOT` | Directory of the conda installation. |
| `REQUIRED_CUDA_ARCHS` | CUDA archs torch must be compiled for, e.g. `"sm_90 sm_100"`. `env_check` verifies it. |
| `MACHINE_HOSTNAME_REGEX` **or** `MACHINE_FALLBACK=1` | How the machine is detected. Exactly one machine may be the fallback. |

Optional, with a default: `NODE_KIND` (what kind of node this is: Vista sets `gg`, `gh`, `gb`; empty
falls back to `gpu`, `cpu` or `login`), `PIP_CACHE_DIR`, `CONDA_ENVS_PATH`, and any compiler or
module settings the machine needs (Vista sets `CC=gcc CXX=g++`).

Keep the file cheap and free of side effects beyond exports: the root `env.sh` sources it more than
once while detecting the machine.

## Add a machine

1. `mkdir setup/machines/<name>` and write `env.sh` with the variables above.
2. Test: `MACHINE=<name> bash -c 'source env.sh; env | grep -E "MACHINE|CONDA_ROOT"'`.
3. Write `envs/<env>.sh` with an `env_build` for each env in `setup/envs/` (copy the closest machine), then
   `setup/setup.sh --check`, `--rebuild <env>` and `--freeze`.
