# marin_rl_horizon

Run Marin's Snowball RL presets (67B Grug MoE, single-turn, non-agentic) on TACC Horizon through the Slurm launcher backend of
the `marin` fork, with and without the frozen DSpark draft. It answers: does the draft shorten RL updates on GB200, and does it
leave the learning unchanged? (Same question as `rl_speculation`, which uses the MarinSkyRL + OpenThoughts-Agent launcher instead.)

- **Layout:** 10 nodes x 4 GB200 = 8 policy nodes (Megatron TP1/PP2/EP8, the 32-GPU P40 layout on four-GPU nodes) + 2 engine nodes
  (one DP8/EP8 vLLM engine). Whole nodes, `-p gpu`.
- **Repos and envs:** the `marin` fork checkout at `$PROJECT_ROOT/marin` on the branch that holds the Slurm backend (`vendor/skyrl/cloud/slurm/`),
  plus rjpower's open marin-community/marin#11975 (the `dspark` method and the default draft on Snowball presets) until it merges.
  `$PROJECT_ROOT/marin_env/$RUNTIME_NAME` (default `skyrl-megatron`) is the frozen runtime built by `setup/machines/horizon/slurm/build_skyrl_env.sbatch`;
  `$PROJECT_ROOT/marin_env/marin-plan` is the uv env that imports the marin presets to print recipes. Neither is in `setup/repos.txt` yet:
  the fork branch has not been pushed.
- **Inputs:** the policy and the drafter are Hugging Face references pinned to a commit (`ORG/REPO@<40-hex commit>`), resolved through the Hub cache (`$HF_HOME`, default `$PROJECT_ROOT/cache/hf`) and downloaded once on first use (login node, network). The prompts are
  `$PROJECT_ROOT/data/marin_rl_horizon/standin-tasks/train.parquet` (public stand-in prompts: the real `snowball-100k` selection needs Marin object storage; see below).
- **Outputs:** `runs/marin_rl_horizon/<run name>/` (rendered job scripts, Slurm logs, Ray logs, run records), never in this repo.

## Set up (once)

Everything lives under `$PROJECT_ROOT` (the workspace root, parent of this repo; `env.sh` exports it). Run the `sbatch` and download commands from a login node.

```bash
export PROJECT_ROOT=<workspace>; cd $PROJECT_ROOT
git clone <marin fork> marin                      # the branch with the Slurm backend (cloud/slurm/); keep it where it is, the runtime records this path

# 1. the frozen SkyRL runtime and its archive (about 10 min, one node); see the header of the script for MARIN_REPO / RUNTIME_NAME
sbatch -o $PROJECT_ROOT/logs/slurm/%x-%j.log spec-experiments/setup/machines/horizon/slurm/build_skyrl_env.sbatch

# 2. the environment that prints the presets' recipes (CPU only). This is how it was built here: a frozen sync of the marin workspace.
cd $PROJECT_ROOT/marin && UV_PROJECT_ENVIRONMENT=$PROJECT_ROOT/marin_env/marin-plan uv sync --frozen --no-dev

# 3. the weights need no step of their own: the launchers resolve the policy and the drafter from the Hub on first use (see Run)

# 4. the stand-in prompts (public sources; see "Stand-in prompts")
cd $PROJECT_ROOT/marin
for source in openscience reasoning_gym_generated; do
  PYTHONPATH=. $PROJECT_ROOT/marin_env/marin-plan/bin/python -m experiments.post_training.task_curation.driver \
      --mode quick --source $source --run --cache-dir $PROJECT_ROOT/data/marin_rl_horizon/curation/cache --output-root $PROJECT_ROOT/data/marin_rl_horizon/curation/out
done
cd $PROJECT_ROOT/spec-experiments/pipelines/marin_rl_horizon/tools && $PROJECT_ROOT/marin_env/${RUNTIME_NAME:-skyrl-megatron}/bin/python build_standin_tasks.py 4096
```

A Weights & Biases key is needed only for runs with `--wandb-*`: export `WANDB_API_KEY` in the login profile.

## Run

Each launcher folder under `launchers/` is self-contained (`launch.sh` and its `recipe.yaml`): `launchers/<preset>-plain/launch.sh` is the control, `launchers/<preset>-draft/launch.sh [--drafter ORG/REPO@COMMIT]` adds a drafter (the preset's own without `--drafter`). `--policy ORG/REPO@COMMIT` picks the model RL starts from (default: the preset's 09.21 initializer, 134 GB, downloaded on first use). Add `--validate` to check a config without submitting. The commands below are what a launcher does, for running a new recipe.

```bash
export PROJECT_ROOT=<workspace>            # the root env.sh exports it too
cd pipelines/marin_rl_horizon/tools
PY=$PROJECT_ROOT/marin_env/${RUNTIME_NAME:-skyrl-megatron}/bin/python

# 1. recipe from the preset (no config of its own is hand-edited); save it as the recipe.yaml of a launcher folder
PYTHONPATH=$PROJECT_ROOT/marin $PROJECT_ROOT/marin_env/marin-plan/bin/python dump_recipe.py --preset learning-medium > <recipe.yaml>
# 2. launch config; the policy and the drafter resolve from the Hub on first use (login node); --draft none is the control arm
$PY make_config.py --recipe <recipe.yaml> --name-prefix my-run [--policy ORG/REPO@COMMIT] [--drafter ORG/REPO@COMMIT | --draft none] \
    [--wandb-entity <entity> --wandb-project <project>]        # prints the run folder, runs/marin_rl_horizon/my-run-...
./launch.sh <run folder> --validate    # renders and checks, submits nothing
./launch.sh <run folder>               # submits and waits in the background
```

W&B is off unless both `--wandb-*` flags are given, and the entity is always explicit: Marin's own Iris launcher defaults
`WANDB_ENTITY` to another account. `WANDB_API_KEY` must be exported in the login profile (`launch.sh` uses a login shell); it is forwarded
to the nodes by name and never written to disk.

## Tools

| File | What it does |
|---|---|
| `dump_recipe.py` | prints a preset's SkyRL recipe with the P40 plan narrowed to four-GPU nodes (`speed-medium`, `learning-medium`) |
| `make_config.py` | recipe -> `launch.yaml` / `launch.prepare.yaml`; `--policy` and `--drafter` take `ORG/REPO@COMMIT` or a folder; `--draft none` makes the control arm; the Horizon-specific changes are listed in its docstring |
| `horizon_config.py`, `paths.py` | the shared Slurm/Ray/artifact sections and the workspace paths |
| `hf_ref.py` | resolves `ORG/REPO@COMMIT` to a local folder through the Hub cache, downloading once if it is missing |
| `launch.sh` | submit a run folder's config through a login shell, or `--validate` |
| `build_standin_tasks.py` | samples the stand-in prompts from the task_curation driver's output (`python -m experiments.post_training.task_curation.driver --mode quick --source openscience --run --cache-dir C --output-root O`, then the same for `reasoning_gym_generated`) |
| `make_smoke_config.py`, `make_smoke_data.py` | one-node Qwen3-0.6B smoke test of the launcher, the cheapest check after a change |
| `run_backend_tests.sbatch` | the Slurm backend's tests on a compute node (login nodes cannot run them) |

## vLLM build

The runtime's vLLM comes from the marin lockfile (`vendor/skyrl/uv.lock`), so there is nothing to patch: the `GrugDSparkModel` loader that Grug-style drafters need (marin-community/vllm#128) has been in the pinned build since 2026-11-24
(`vllm 0.0.0.dev20261124+marin.7e9b413a5daf`, aarch64 cu132 wheel, torch 2.14.1). Build the runtime from a marin checkout at or after that lock. The first two experiments below ran earlier, on the 20261113 build with the
newer vLLM files copied over it; their notes say so.

## Stand-in prompts

`snowball-100k` is built from Marin object storage, which this account cannot read, so the runs use 4096 prompts sampled in equal parts from
two public sources (OpenScience, Reasoning Gym), fixed seed. Families missing against the real mix: competitive coding, structural,
calendar. `eval-lite` is also in object storage, so the learning preset runs without evaluations. Rewards here are not comparable to Marin's.

## Results

Write-ups only; the launchers that produced them are in `launchers/`.

| Write-up | Question | Result |
|---|---|---|
| `results/snowball-speed-medium_plain-vs-dspark.md` | update time, 10 updates, every reply forced to 4096 tokens | DSpark median update 50 s vs 76 s |
| `results/snowball-learning-medium_plain-vs-dspark.md` | 25 updates, natural stop, rewards logged | steps 2-25 take 28 min vs 51 min; reward curves match |

Every number in a write-up says whether it is a target or an achieved result, and carries its hardware and config.
