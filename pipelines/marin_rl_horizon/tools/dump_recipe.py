"""Print the SkyRL recipe of a Snowball preset with its training role plan narrowed to four-GPU nodes.

    PYTHONPATH=$MARIN_REPO marin-plan/bin/python dump_recipe.py --preset learning-medium > recipe.yaml

Any preset of snowball_rl.SNOWBALL works. Follows rl_compose.run_spec without evaluations, checkpoints, export or publishing
(none of them can run here: eval-lite and the dashboards are in Marin object storage), so no storage is read and no task input
is needed. The policy keeps the GPU count its plan gives it, spread over four-GPU nodes: P40's four eight-GPU nodes become
eight four-GPU nodes, and the engine's eight GPUs span two more.
The recipe carries the preset's default draft (DSpark on the pinned initializer) when the marin checkout includes
marin-community/marin#11975; make_config.py --draft none removes it.
"""

import argparse
import sys
from dataclasses import replace

import yaml
from skyrlrecipe import RecipePatch, SkyRLRecipe, sections

from experiments.post_training import rl_compose as c
from experiments.post_training.snowball_rl import SNOWBALL

GPUS_PER_NODE = 4


def build_recipe(name: str) -> tuple[dict, int]:
    # A preset may evaluate (eval-lite, a task set, Harbor) and save, export or publish checkpoints; none of that is available here.
    preset = replace(SNOWBALL.by_name[name], evaluations=(), harbor=None, checkpoints="none", checkpoint_every=None, export="none", publishes=False)
    config = c._training_config(preset, (), preset.model)
    workload = preset.recipe.layout.workload
    _, in_flight = workload.batch_parameters(config)
    config = config.merge(RecipePatch.from_document({"trainer": {"rollout_buffer": {"max_in_flight": in_flight}}}))
    preset = preset.with_updates(config.trainer.max_steps)
    config = config.merge(RecipePatch(trainer=sections.Trainer(callbacks=workload.callbacks(preset, None, {}, 0, None))))
    recipe = SkyRLRecipe.from_document(config.to_skyrl()).to_skyrl()
    policy_gpus = workload.plan.policy_num_nodes * workload.plan.policy_num_gpus_per_node
    assert policy_gpus % GPUS_PER_NODE == 0, policy_gpus
    plan = replace(workload.plan, policy_num_nodes=policy_gpus // GPUS_PER_NODE, policy_num_gpus_per_node=GPUS_PER_NODE)
    recipe["entrypoint"] = plan.entrypoint
    for dotted, value in plan.config_values().items():
        node = recipe
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return recipe, in_flight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(SNOWBALL.by_name), required=True)
    recipe, in_flight = build_recipe(parser.parse_args().preset)
    print(yaml.safe_dump(recipe, sort_keys=False))
    print("# in_flight", in_flight, "max_steps", recipe["trainer"]["max_steps"], file=sys.stderr)


if __name__ == "__main__":
    main()
