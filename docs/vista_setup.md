# Setting up the marin_speculator workspace on TACC Vista

Instructions for a Claude Code agent on TACC Vista. Together with the user, you'll recreate
the `marin_speculator` workspace there: clone the repos, build the environments, and get the
experiments in `spec-experiments` running under Slurm. The workspace was developed on the UT
ECE genai machines (x86, several H100s per machine, no scheduler). This is a long-lived
project, so expect more repos and dependencies over time, and keep the setup easy to extend.

Read this whole guide before running anything. Work through the phases in order. Each
phase ends with a **checkpoint**: stop, report what you found, and agree the next step with
the user.

## First: know which node you are on

The user works with you on Vista in two ways, and what you can do depends on the node:

| Session | Node | Use it for | Don't |
|---|---|---|---|
| **CPU session** | `gg` (Grace CPU only), or a login node | Editing code, commits (with permission), `setup/setup.sh --check`, building envs (on `gg`, not login) | Run anything that needs a GPU. `nvidia-smi` may be missing or fail, and that's normal here. |
| **GPU idev session** (up to 2 h) | `gh` (1× H100 per node) or `gb` (4× GB200 per node) | Smoke tests, `setup/setup.sh --gpu` | Start a full run. **Full runs always go through `sbatch`**, as do jobs that can't finish in the session's time left. |

**`sbatch` is refused on compute nodes** (gg, gh, gb, including idev sessions: "sbatch not available on compute nodes. Use a login node."). Ask the user to submit from a login node, then watch with `squeue -u $USER` and the job's log. Job accounting: use the account in uppercase (`#SBATCH -A CCR24067`), or Slurm errors out.

At the start of every session, run `source <PROJECT_ROOT>/spec-experiments/env.sh; echo
$NODE_KIND $NUM_GPUS $SLURM_JOB_PARTITION`, plus `squeue -u $USER` for the time left.
`env.sh` sets `NODE_KIND` to `gg`, `gh` or `gb` from the Slurm partition, otherwise to
`gpu`, `cpu` or `login`, and sets `NUM_GPUS` without failing where `nvidia-smi` is
unavailable. Never assume a GPU is present. If a task needs a GPU and you're on a CPU node,
say so and stop.

**One env set for both GPU node types.** The aim is a single `vllm` and a single
`speculators` env that work on gh (H100, `sm_90`) and gb (Blackwell, `sm_100`), and that
can be built on any node, gg included. Building needs no GPU. `env.sh` sets
`REQUIRED_CUDA_ARCHS="sm_90 sm_100"` on Vista, and each `env_check` verifies that torch was
compiled for both, which works without a GPU. Still run `setup/setup.sh --gpu` on **both** a
gh and a gb node before calling an env good. If one env really can't cover both, explain
why to the user before splitting into per-node-type envs.

## Ground rules

- **Ask before deciding.** The user wants an interactive workflow. Ask before choosing
  layouts or tooling, changing an experiment's settings, or launching anything bigger than a
  smoke test. Never launch a full run without an explicit go.
- **Don't commit or push** without asking.
- **Keep heavy work off the login nodes.** Building environments, downloading or processing
  datasets, and anything using a GPU go in an `idev` session or an `sbatch` job.
- **Check, don't assume.** The Vista facts below are from memory and marked *verify*. Confirm
  them, and tell the user where they were wrong.
- Before editing a file or launching a job, check that another session isn't already doing
  the same (`squeue -u $USER`, file mtimes). The user sometimes runs several Claude sessions
  at once.
- **Generalize.** Prefer machine-specific settings in one place (`env.sh` or a per-machine
  variant) over hardcoded paths in scripts. Record what you set up in this guide, so the next
  setup is easier.

## The workspace

`marin_speculator/` is a plain directory, not a repo. It holds the repos as siblings, plus
the data and outputs that don't belong in version control:

```
marin_speculator/                 # PROJECT_ROOT
├── spec-experiments/             # pipelines, env.sh, setup, docs (this repo)
├── speculators/                  # speculators fork (library)
├── <future repos>/               # more dependencies as the project grows
├── cache/  tmp/                  # HF, vLLM, torch caches; tmp/ is TMPDIR except on Vista (node-local /tmp)
├── logs/{setup,slurm,smoke}/     # env builds and GPU checks, sbatch output, smoke tests
└── runs/<run name>/              # all outputs of one run (its logs in runs/<run name>/logs/)
```

`spec-experiments/env.sh` is sourced at the start of every script. It sets
`PROJECT_ROOT` to the parent of the repo, picks `MACHINE` by hostname (or an explicit
`MACHINE=<name>`) and sources `setup/machines/<machine>/env.sh` (the Vista settings are in
`setup/machines/vista/env.sh`; contract in `setup/machines/README.md`), points every cache and
`TMPDIR` into `PROJECT_ROOT`, and initializes conda. Scripts reach other repos as
`$PROJECT_ROOT/<repo>`.

**Rebuilding the workspace is automated, and you keep it that way.** Use the
`workspace-setup` skill (`spec-experiments/.claude/skills/workspace-setup/SKILL.md`; read it
now):
- `setup/repos.txt` lists the repos.
- `setup/envs/<env>.sh` holds each env's check; `setup/machines/<machine>/envs/<env>.sh` its build steps.
- `setup/setup.sh` is the idempotent driver (with `--check`, `--rebuild ENV`, `--freeze` and
  `--gpu`).
- `setup/machines/<machine>/locks/` holds exact package lists of known-good envs.

On genai, all of this was tested. On Vista, the recipes were `TODO(vista)` stubs, and
writing them was the main job of Phase 3 (done; see "Vista status"). Vista's `$SCRATCH` is purged, so a fix made by hand
inside an env is lost. Put every install step in a recipe.

### Repositories

`setup/repos.txt` is the source of truth. Keep it and this table current as dependencies
are added.

| Directory | Remote | Ref | Role |
|---|---|---|---|
| `spec-experiments` | `git@github.com:atutej/spec-experiments.git` | `main` | Pipelines (experiments by family), `env.sh`, workspace setup, docs |
| `speculators` | `git@github.com:atutej/speculators.git` (`upstream`: `marin-community/speculators`) | `nemotron-terminal-preset` | Speculative-decoding library: training, `regenerate-responses`, `prepare-data`, `scripts/launch_vllm.py`, plus the `hs_connectors` sub-package in `speculators/hs_connectors/` |

The `speculators` branch is upstream `42ed6ff` plus the user's commits. It currently has one
extra commit, which adds the `nemotron-terminal` dataset preset. Its working tree should stay
clean apart from intended changes.

### Environments (as on genai)

There are two conda envs, because vLLM and speculators pin different dependency versions.
Exact package lists are in `setup/machines/genai/locks/{vllm,speculators}.txt`.

| Env | Used for | Key packages on genai (x86_64, Python 3.12.14) |
|---|---|---|
| `vllm` | Serving: generation, the hidden-state server | **Custom Marin vLLM build** `0.0.0.dev20260929+marin.39e62869693c` (cu132), `torch` 2.13.0+cu132, `transformers` 5.18.0, `hs_connectors` (editable), `mooncake-transfer-engine-cuda13` 0.3.13.post1 |
| `speculators` | Data prep, training, analysis | `speculators` 0.9.0.dev32 (editable from `speculators/`), `hs_connectors` (editable), `torch` 2.13.0, `transformers` 5.16.1 (speculators pins `>=5.0,<5.17`), `datasets` 5.0.1, `pyarrow` 25.0.1, `mooncake-transfer-engine-cuda13` (provides `mooncake_master`), jupyterlab |

The genai driver is 595.84, with nvcc 12.8. The Marin vLLM was installed from a local x86-only
wheel, `vllm-0.0.0.dev20260929+marin.39e62869693c.cu132-cp38-abi3-manylinux_2_28_x86_64.whl`.
That's Marin's vLLM fork at commit `39e62869693c`. Ask the user where it came from.

## How Vista differs (verify every item)

- **CPU architecture:** all node types use NVIDIA Grace CPUs, so they're **aarch64 (ARM)**.
  No x86 env or wheel carries over. Every package needs an aarch64 build, and the custom vLLM
  is the hardest one.
- **Node types:** `gg` is Grace CPU only. `gh` is 1× H100 (about 96 GB) per node. `gb` has 4×
  GB200 (Blackwell) per node. *Verify* the GPU model, memory and driver (`nvidia-smi`) and the
  CPU memory (`free -g`) on a gh and a gb node. The driver decides the newest CUDA a single
  env can use on both.
- **GPU count changes the design:** a **gb** node has 4 GPUs, like the genai machines, so a
  smoke test of a genai 4-GPU script may run on one gb node almost unchanged. A **gh** node
  has 1 GPU, so smoke tests there need a one-GPU variant. Full runs are `sbatch` jobs that may
  use **several nodes** of either type, depending on model size. Their configuration is
  decided with the user (see Phase 4).
- **Slurm:** check partition names (e.g. `gg`, `gh`, `gh-dev`, `gb`, ...) and time limits with
  `sinfo -s` and `scontrol show partition`. Nodes are exclusive, so no GPU sharing with other
  users. idev sessions last up to 2 h; longer work goes through `sbatch`.
- **Filesystems:** `$HOME` (small), `$WORK` (medium, not purged) and `$SCRATCH` (large, purged
  when files go unused). **The user's decision is to put the whole workspace in `$SCRATCH`:
  `PROJECT_ROOT=$SCRATCH/marin_speculator`, with repos, `cache/`, `tmp/` and `runs/` all
  inside it.** Check the purge policy and quotas, and tell the user what's at risk:
  - The repos are on GitHub and can be re-cloned.
  - Envs live in `$PROJECT_ROOT/envs` (the user's choice), using the conda installation in
    `$WORK/miniconda3`. `env.sh` sets `CONDA_ENVS_PATH`, so `conda activate vllm` works by
    name. After a purge, `setup/setup.sh` rebuilds them.
  - Run outputs such as checkpoints and final results can't be recovered. Propose a way to
    copy what matters to `$WORK` or elsewhere once a run finishes.

  If envs installed on Lustre `$SCRATCH` import slowly (many small files), tell the user
  before moving them anywhere.
- **Internet from compute nodes:** *verify*. If there's none, model, dataset and pip
  downloads must happen on a login node or another node with access.
- **Modules:** CUDA, GCC and Python versions from `module avail` decide which PyTorch and vLLM
  builds can work.

## Phases

### Phase 1: Orientation

Run this on whatever node you're on, and summarize it for the user. Repeat the GPU line
on a gh and a gb node when the user has those sessions:

```bash
hostname; uname -m; nproc; free -g | head -2
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "no GPU visible"
echo "job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION nodes=$SLURM_NNODES"
sinfo -s | head -20
echo "HOME=$HOME WORK=$WORK SCRATCH=$SCRATCH"; df -h $HOME $WORK $SCRATCH
module list 2>&1 | tail -5; module avail cuda python gcc 2>&1 | head -40
which conda mamba micromamba uv python3 2>&1; python3 --version
curl -sI --max-time 10 https://huggingface.co | head -1
timeout 10 ssh -T -o BatchMode=yes git@github.com 2>&1 | tail -1
```

**Checkpoint:** report the facts, correct anything in "How Vista differs", and confirm
`$SCRATCH/marin_speculator` as `PROJECT_ROOT` with envs in `$PROJECT_ROOT/envs`. Check that
`$WORK/miniconda3` exists.

### Phase 2: Workspace and code

1. Clone `spec-experiments` into `$SCRATCH/marin_speculator/`, then run
   `bash spec-experiments/setup/setup.sh --check`. That clones nothing; run it without
   `--check` to clone the repos in `setup/repos.txt`. The env steps will fail at the
   `TODO(vista)` stubs, which is expected until Phase 3. If GitHub SSH doesn't work on Vista,
   ask the user, because the repos may be private.
2. Fill in `setup/machines/vista/env.sh`: confirm `CONDA_ROOT=$WORK/miniconda3`, and add the
   module loads Phase 3 needs. Keep the genai branch working.

**Checkpoint:** show the layout and the `env.sh` change.

### Phase 3: Environments

Build the `vllm` and `speculators` envs for aarch64, matching the genai versions where
possible:

- **PyTorch** with CUDA for aarch64 (sbsa), in a version compatible with the CUDA modules and
  driver.
- **vLLM from the Marin fork at `39e62869693c`.** Options, in order of preference:
  - an existing aarch64 wheel of that build (ask the user),
  - building from source on a compute node (slow; use `MAX_JOBS` and ccache),
  - upstream vLLM, *only* if the user agrees.

  The speculators hidden-state tooling (`launch_vllm.py`, `hs_connectors`) relies on the
  fork's `extract_hidden_states` method and hidden-state KV connectors, which upstream may lack.
- **`speculators`** (editable, from the clone) and **`hs_connectors`** (editable, from
  `speculators/hs_connectors`) in **both** envs.
- **Mooncake:** check for an aarch64 `mooncake-transfer-engine` build. If there isn't one,
  `launch_vllm.py` already includes a `file` hidden-states backend as a fallback. Switching
  changes how experiments run, so ask the user.

Building can happen on gg. Validate the parts that need no GPU anywhere, and the rest on a gh
**and** a gb node:
- imports of `vllm`, `speculators` and `hs_connectors`,
- `vllm serve Qwen/Qwen3-0.6B` passing its `/health` check,
- one chat completion with `"return_token_ids": true` that returns `prompt_token_ids`, which
  `regenerate-responses` needs,
- `mooncake_master --help` if using Mooncake.

Put every working step into `setup/machines/vista/envs/<env>.sh`, and make `env_check` (in `setup/envs/<env>.sh`)
verify it. Then confirm `setup/setup.sh` ends with `READY` (on any node) and `setup/setup.sh --gpu` ends
with `READY` on both a gh and a gb node, and run
`setup/setup.sh --freeze` to record `setup/machines/vista/locks/`. A good test is a rebuild from
scratch into a fresh `CONDA_ENVS_PATH`, because that's what happens after a purge.

**Checkpoint:** report versions, differences from genai, and workarounds.

### Phase 4: Experiments

The experiments are under `spec-experiments/pipelines/<family>/experiments/<name>/` (`genai/run.sh`). Each one was written for genai,
as a single bash pipeline that assumes several GPUs on one machine. For each experiment the
user wants on Vista:

1. Read its script and any notes about it (see "Experiment notes" below).
2. Run its steps as **smoke tests** at small scale, comparing with genai reference results
   where they exist.
3. Agree the full-run design with the user, then write sbatch scripts (for example in
   `<experiment>/vista/`) that reuse the experiment's settings. Full runs are
   always `sbatch` jobs, possibly multi-node on gh or gb. The node type, node count and
   per-stage layout are open, so decide them with the user and record the result in this
   experiment's notes below. Show the scripts to the user before submitting. Patterns that
   may help:
   - Embarrassingly parallel generation as a **job array**: shard the input, run one server
     per node, `--resume` so timed-out shards continue, then merge.
   - Later stages as **dependent jobs** (`--dependency=afterok:`).
   - Restartability across time limits, using the trainer's checkpoints.
4. Launch only with the user's explicit go.

**Checkpoint after each experiment's smoke test**, and again before any full run.

## Vista status (2026-10-07)

Phase 3 is done for `gg` and `gb`; **`gh` is not validated yet.**

Built on `gg`: `vllm` (Marin fork `39e62869693c` compiled for sm_90 + sm_100, wheel cached in
`$WORK/wheels`, via `setup/machines/vista/slurm/build_vllm_env.sbatch`, about 1 h on a full gg node) and
`speculators` (`setup/machines/vista/slurm/build_speculators_env.sh`). `setup.sh --gpu` passes on **gb**
(4x GB200, driver 590.48.01).

Workarounds that are now in the recipes and `env.sh` (don't undo them):
- `env.sh` sets `CC=gcc CXX=g++`: the default `nvidia` module sets nvc/nvc++, which torch
  inductor can't use (`-Wno-psabi`).
- The vllm env has its own `cuda-home` (pip `cuda-toolkit==13.2.1` plus symlinks, and a link-time
  libcuda stub from module cuda/13.1), activated by `conda activate vllm`. Reason: pip's toolkit
  has no unversioned `.so` files or lib64 (CMake failed), and FlashInfer JITs with the `nvcc` on
  PATH, which would be the nvidia module's 12.5.
- `torchaudio==2.11.0+cpu` in `speculators` (no cu132 build exists; PyPI's cu130 one won't import).
- `FLASHINFER_WORKSPACE_BASE` points into `$PROJECT_ROOT/cache` (small `$HOME` quota).

**To test on gh (1x H100, sm_90) when we get there:**
1. `setup/setup.sh --gpu` ends with `READY`. FlashInfer JIT-compiles for sm_90a there, which hasn't
   been exercised; check the `cuda-home` nvcc handles it.
2. Check the driver supports CUDA 13.2 (gb's 590 does).
3. Anything that assumes several GPUs per node (see the experiment notes).

Not tested on any node: `mooncake_master` actually running (only the PATH check), Mooncake
transfers between processes, `speculators` training and `prepare-data`, multi-node.

## Experiment notes

Per-experiment notes (settings, what was learned on genai, reference results, Vista smoke-test
log) live next to the experiment: `pipelines/<family>/experiments/<name>/NOTES.md`. Add one when
an experiment is set up or changed. Currently:

- `pipelines/speculator_training/experiments/dspark_qwen3_0_6b_nemotron-terminal-corpus_onpolicy/NOTES.md`

## Open questions for the user

- Where the Marin vLLM wheel came from, and whether an aarch64 build exists.
- Whether the repos are public or private.
- SU budget, partition, and how many nodes to use at once.
- Mooncake versus the `file` hidden-state backend on Vista.
- Which experiments to set up first.
