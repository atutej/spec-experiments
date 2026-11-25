"""Workspace paths for the Horizon Snowball RL tools.

Every default hangs off PROJECT_ROOT, the workspace root that holds spec-experiments/, marin/, envs/ and runs/ (the root
env.sh exports the same variable). Each path can be overridden by an environment variable of the same name.
"""

import os
from pathlib import Path


def _path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default)


PIPELINE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _path("PROJECT_ROOT", PIPELINE.parents[2])

MARIN_REPO = _path("MARIN_REPO", PROJECT_ROOT / "marin")  # checkout with the Slurm backend; the runtime records its path
MARIN_ENV = _path("MARIN_ENV", PROJECT_ROOT / "marin_env")  # holds the runtime env, activation file and archives
RUNTIME_NAME = os.environ.get("RUNTIME_NAME", "skyrl-megatron")  # the name build_skyrl_env.sbatch was given
RUNTIME_ENV = MARIN_ENV / RUNTIME_NAME
ACTIVATION = MARIN_ENV / f"{RUNTIME_NAME}.runtime.env"
RUNTIME_ARCHIVE = MARIN_ENV / f"{RUNTIME_NAME}-site-packages.tar"

DATA = _path("MARIN_RL_DATA", PROJECT_ROOT / "data" / "marin_rl_horizon")
RUNS = _path("MARIN_RL_RUNS", PROJECT_ROOT / "runs" / "marin_rl_horizon")
CACHE = _path("MARIN_RL_CACHE", PROJECT_ROOT / "cache" / "marin_rl_horizon")

MODELS = _path("MODELS_DIR", PROJECT_ROOT / "models")  # local model folders (the Qwen3-0.6B smoke test); Hub models resolve through HF_HOME
STANDIN_TASKS = DATA / "standin-tasks"
HF_HOME = _path("HF_HOME", PROJECT_ROOT / "cache" / "hf")  # model references resolve through the Hub cache here
# What the launchers run when --policy / --drafter are not given: the Dr. Doom policy (the 09.21 initializer after Dr. Doom training) and the
# 6-layer Grug DSpark drafter trained on it (layer_type grug, taps 2/7/12/17/22, block 8). Both are pinned to a commit.
DEFAULT_POLICY = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom@fd4a272b78d4c224851aaeae7f51987c5d19649c"
DEFAULT_DRAFTER = "atutej/dspark_grug-layers6-taps5-grug-67b-a2b-datakit-sft-dr-doom_datakit-sft-mix_itw-20261108@06501b01b28657f570229f387605c870df6a871a"
