"""Pieces of a Horizon launch config shared by the Snowball and smoke builders.

Each one answers a Horizon fact (see setup/machines/horizon/NOTES.md): whole four-GPU nodes, a per-user thread limit that
Ray's default worker count exceeds, a shared NFS /scratch that stalls under concurrent compile and weight reads, and no
CoreWeave InfiniBand device names.
"""

import getpass
from pathlib import Path

import paths

USER = getpass.getuser()
GPUS_PER_NODE = 4
TIMEOUT = 7200  # collective and placement-group deadlines; startup on a cold node is long
ENGINE_INIT_TIMEOUT = 1800  # startup takes minutes; a hung node should fail the job, not hold it for hours
RAY_NUM_CPUS = 64  # Ray starts one idle worker per CPU; 144 of them exceeded the per-user thread limit
NODE_LOCAL = "/tmp"
NODE_DATA = f"{NODE_LOCAL}/skyrl-{USER}-data"  # each node copies its inputs here
NODE_CACHE = f"{NODE_LOCAL}/skyrl-{USER}-cache"  # compile caches on shared /scratch stalled with stale file handles


def slurm_section(
    run_dir: Path,
    time_limit: str,
    *,
    partition: str = "gpu",
    preflight_files: list[str] | None = None,
    wandb_entity: str | None = None,
    exclude_nodes: list[str] | None = None,
) -> dict:
    env = {
        "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
        "XDG_CACHE_HOME": str(paths.CACHE),
        "TRITON_CACHE_DIR": f"{NODE_CACHE}/triton",
        "TORCHINDUCTOR_CACHE_DIR": f"{NODE_CACHE}/inductor",
        "FLASHINFER_WORKSPACE_BASE": str(paths.CACHE / "flashinfer"),
    }
    section = {
        "run_dir": str(run_dir), "time_limit": time_limit, "runtime_env": str(paths.RUNTIME_ENV),
        "activation_file": str(paths.ACTIVATION), "runtime_archive": str(paths.RUNTIME_ARCHIVE), "node_local_dir": NODE_LOCAL,
        "partition": partition, "poll_interval": 15, "env": env,
    }
    if preflight_files:
        section["preflight_files"] = preflight_files
    if exclude_nodes:
        section["extra_sbatch"] = [f"--exclude={','.join(exclude_nodes)}"]
    if wandb_entity:
        # An explicit entity: Marin's own Iris launcher defaults WANDB_ENTITY to another account.
        env["WANDB_ENTITY"] = wandb_entity
        section["forward_env"] = ["WANDB_API_KEY"]  # forwarded by name from the submitting shell, never written to disk
    return section


def ray_section(run: Path, *, num_cpus: int | None) -> dict:
    ray = {"spill_backend": "local", "spill_dir": "/tmp/skyrl-ray-spill",
           "rendezvous_dir": str(run / "rendezvous"), "log_dir": str(run / "ray-logs")}
    if num_cpus is not None:
        ray["num_cpus"] = num_cpus
    return ray


def artifacts_section(art: Path) -> dict:
    return {"checkpoint_root": str(art / "checkpoints"), "export_root": str(art / "exports"),
            "attempts_root": str(art / "attempts"), "resolved_config_uri": str(art / "resolved.yaml"),
            "terminal_manifest_uri": str(art / "terminal.json")}
