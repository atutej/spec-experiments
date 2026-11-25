"""Build the Horizon launch config (and its validate-only twin) for a Snowball recipe.

    python make_config.py --recipe recipe.yaml --name-prefix horizon-learning-medium-draft \
        [--policy ORG/REPO@COMMIT] [--drafter ORG/REPO@COMMIT | --draft none] [--wandb-entity E --wandb-project P]

The policy and the drafter are Hugging Face references pinned to a commit (or absolute folders); hf_ref.py resolves them
through the Hub cache, downloading once if needed (login node). Changes only what Horizon requires: FlashInfer attention
(Blackwell), local weights and stand-in prompts, a Ray CPU cap
under the per-user thread limit, no CoreWeave-only NCCL setting, W&B only on request, and a `slurm` section. Writes
<out>/launch.yaml (submission: wait) and <out>/launch.prepare.yaml (validates, submits nothing) and prints <out>.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml

import hf_ref
import horizon_config as hc
import paths

SEED = 20261019


def data_source(file: str) -> dict:
    return {"kind": "directory", "uri": str(paths.STANDIN_TASKS), "identity": "snowball-standin-v2",
            "local_path": f"{hc.NODE_DATA}/standin-tasks", "relative_path": file}


def recipe_draft_ref(recipe: dict) -> str:
    """The drafter the recipe names, as ORG/REPO@COMMIT."""
    model = recipe["generator"]["speculative_decoding"]["model"]
    assert model["source_uri"].startswith("hf://"), model
    return f"{model['source_uri'].removeprefix('hf://')}@{model['source_identity']}"


def build(args: argparse.Namespace, attempt_id: str) -> dict:
    recipe = yaml.safe_load(args.recipe.read_text())
    placement = recipe["trainer"]["placement"]
    placement["ref_num_gpus_per_node"] = hc.GPUS_PER_NODE  # the recipe was authored for eight-GPU nodes
    generator = recipe["generator"]
    engine_gpus = (generator["num_inference_engines"] * generator["inference_engine_data_parallel_size"]
                   * generator["inference_engine_tensor_parallel_size"] * generator["inference_engine_pipeline_parallel_size"])
    assert engine_gpus % hc.GPUS_PER_NODE == 0, engine_gpus
    num_nodes = placement["policy_num_nodes"] + engine_gpus // hc.GPUS_PER_NODE
    generator["vllm_attention_backend"] = "FLASHINFER"
    generator["engine_init_timeout_seconds"] = hc.ENGINE_INIT_TIMEOUT
    recipe["trainer"].setdefault("distributed", {}).update(
        placement_group_timeout_seconds=hc.TIMEOUT, worker_collective_timeout_seconds=hc.TIMEOUT
    )
    if args.draft == "none":
        generator.pop("speculative_decoding", None)
    else:
        if "speculative_decoding" not in generator:
            raise ValueError("--draft keep, but the recipe has no speculative_decoding block (marin checkout lacks #11975?)")
        speculative = generator["speculative_decoding"]
        # compute nodes read the resolved local snapshot, not the Hub
        speculative["model"] = {"source_uri": args.drafter_path, "source_identity": args.drafter_identity}
        if args.proposals:
            speculative["num_speculative_tokens"] = args.proposals
    # The preset also logs to Marin's W&B project, which needs credentials this account does not have.
    loggers = [name for name in recipe["trainer"]["logger"] if name != "wandb"]
    if args.wandb_entity:
        loggers.append("wandb")
        recipe["trainer"]["project_name"] = args.wandb_project
    recipe["trainer"]["logger"] = loggers
    # NCCL_IB_HCA=ibp names CoreWeave's InfiniBand devices; Horizon's differ, so let NCCL detect them.
    task_env = {key: value for key, value in recipe.pop("extra_env").items() if key != "NCCL_IB_HCA"}

    root = args.out
    run = root / "run" / attempt_id
    first_shard = sorted(Path(args.policy_path).glob("*.safetensors"))[0]
    return {
        "schema_version": 1,
        "run": {"id": args.name, "attempt_id": attempt_id, "seed": SEED, "mode": "train", "submission": "wait",
                "export_hf": False},
        "runtime": {"profile": "megatron", "entrypoint": "", "experiments_dir": str(paths.MARIN_REPO / "experiments"),
                    "task_env": task_env},
        "iris": {"job_name": args.name, "max_retries": 0,
                 "allocation": {"num_nodes": num_nodes, "gpus_per_node": hc.GPUS_PER_NODE, "gpu_variant": "GB200"}},
        "slurm": hc.slurm_section(root / "slurm", args.time_limit, partition=args.partition,
                                  preflight_files=[str(first_shard)], wandb_entity=args.wandb_entity,
                                  exclude_nodes=args.exclude_node),
        "ray": hc.ray_section(run, num_cpus=hc.RAY_NUM_CPUS),
        "artifacts": hc.artifacts_section(root / "artifacts" / attempt_id),
        "inputs": {"model": {"uri": args.policy_path, "identity": args.policy_identity, "local_path": args.policy_path},
                   "data_kind": "parquet", "train_data": [data_source("train.parquet")], "validation_data": []},
        "skyrl": recipe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--name", help="run id, also the Slurm job name and the folder under runs/")
    parser.add_argument("--name-prefix", help="without --name: the run id is <prefix>-<policy commit>[-<drafter commit>] (first 8 hex)")
    parser.add_argument("--policy", default=paths.PRESET_POLICY,
                        help="the model RL starts from, ORG/REPO@<40-hex commit> or an absolute folder (default: the preset's initializer)")
    parser.add_argument("--draft", choices=("keep", "none"), default="keep",
                        help="keep the recipe's speculative decoding block or remove it (the control arm)")
    parser.add_argument("--drafter", help="with --draft keep: ORG/REPO@<40-hex commit> or an absolute folder instead of the drafter the recipe names")
    parser.add_argument("--proposals", type=int, help="speculative tokens per round; default keeps the recipe's")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--exclude-node", action="append", default=[], help="Slurm node to avoid; repeatable")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    parser.add_argument("--attempt-id", default=f"attempt-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if bool(args.wandb_entity) != bool(args.wandb_project):
        parser.error("--wandb-entity and --wandb-project go together")
    if bool(args.name) == bool(args.name_prefix):
        parser.error("give exactly one of --name and --name-prefix")
    if args.draft == "none" and (args.drafter or args.proposals):
        parser.error("--drafter and --proposals need --draft keep")
    recipe = yaml.safe_load(args.recipe.read_text())
    args.policy_path, args.policy_identity = hf_ref.resolve(args.policy, "--policy")
    tags = [args.policy_identity[:8]]
    if args.draft == "keep":
        args.drafter_path, args.drafter_identity = hf_ref.resolve(args.drafter or recipe_draft_ref(recipe), "--drafter", real_files=True)
        tags.append(args.drafter_identity[:8])
    args.name = args.name or "-".join([args.name_prefix, *tags])
    args.out = args.out or paths.RUNS / args.name
    args.out.mkdir(parents=True, exist_ok=True)
    for submission, name in (("prepare", "launch.prepare.yaml"), ("wait", "launch.yaml")):
        config = build(args, args.attempt_id)
        config["run"]["submission"] = submission
        (args.out / name).write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"{args.attempt_id}: wrote {args.out}/launch.yaml and launch.prepare.yaml", file=sys.stderr)
    print(args.out)  # the run folder, for the launcher


if __name__ == "__main__":
    sys.exit(main())
