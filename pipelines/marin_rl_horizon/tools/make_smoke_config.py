"""Build the one-node launch config for Marin's own Qwen3-0.6B smoke recipe: the cheapest test of the Slurm launcher.

Starts from vendor/skyrl/cloud/iris/configs/qwen_megatron_smoke.yaml in the marin checkout and changes only what Horizon
requires: four GPUs on one node, FlashInfer attention (Blackwell), local model and data, and a `slurm` section.
Needs models/Qwen3-0.6B (and Qwen3-0.6B.revision with its commit) and `make_smoke_data.py` output under data/.
Writes <out>/launch.yaml and <out>/launch.prepare.yaml.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

import horizon_config as hc
import paths


def build(submission: str, attempt_id: str, out: Path) -> dict:
    recipe = yaml.safe_load((paths.MARIN_REPO / "vendor/skyrl/cloud/iris/configs/qwen_megatron_smoke.yaml").read_text())
    recipe["trainer"]["placement"]["policy_num_gpus_per_node"] = hc.GPUS_PER_NODE
    recipe["generator"]["num_inference_engines"] = hc.GPUS_PER_NODE
    recipe["generator"]["vllm_attention_backend"] = "FLASHINFER"
    # Startup on a node with cold compile caches takes far longer than the 180 s to 30 min defaults.
    recipe["trainer"].setdefault("distributed", {}).update(
        placement_group_timeout_seconds=hc.TIMEOUT, worker_collective_timeout_seconds=hc.TIMEOUT
    )
    recipe["generator"].update(engine_init_timeout_seconds=hc.TIMEOUT, weight_sync_pause_timeout_seconds=hc.TIMEOUT)
    model = paths.MODELS / "Qwen3-0.6B"
    data = paths.DATA / "smoke"
    run = out / "run" / attempt_id
    return {
        "schema_version": 1,
        "run": {"id": "horizon-smoke", "attempt_id": attempt_id, "seed": 42, "mode": "train",
                "submission": submission, "export_hf": False},
        "runtime": {"profile": "megatron", "entrypoint": "skyrl_train.entrypoints.main_base",
                    "experiments_dir": str(paths.MARIN_REPO / "experiments")},
        "iris": {"job_name": "horizon-smoke",
                 "allocation": {"num_nodes": 1, "gpus_per_node": hc.GPUS_PER_NODE, "gpu_variant": "GB200"}},
        "slurm": hc.slurm_section(out / "slurm", "01:00:00",
                                  preflight_files=[str(model / "model.safetensors")]),
        "ray": hc.ray_section(run, num_cpus=None),
        "artifacts": hc.artifacts_section(out / "artifacts" / attempt_id),
        "inputs": {"model": {"uri": str(model), "identity": (paths.MODELS / "Qwen3-0.6B.revision").read_text().strip(),
                             "local_path": str(model)},
                   "data_kind": "parquet",
                   "train_data": [{"kind": "directory", "uri": str(data), "identity": "smoke-arithmetic-v1",
                                   "local_path": str(data), "relative_path": "train.parquet"}],
                   "validation_data": []},
        "skyrl": recipe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=paths.RUNS / "horizon-smoke")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    attempt_id = f"attempt-{time.strftime('%Y%m%d-%H%M%S')}"
    for submission, name in (("prepare", "launch.prepare.yaml"), ("wait", "launch.yaml")):
        (args.out / name).write_text(yaml.safe_dump(build(submission, attempt_id, args.out), sort_keys=False))
    print(f"{attempt_id}: wrote {args.out}/launch.yaml and launch.prepare.yaml")


if __name__ == "__main__":
    sys.exit(main())
