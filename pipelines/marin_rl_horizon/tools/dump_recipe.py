"""Print the SkyRL recipe of a Snowball preset with its P40 training role plan narrowed to four-GPU nodes.

    PYTHONPATH=$MARIN_REPO marin-plan/bin/python dump_recipe.py --preset learning-medium > recipe.yaml

Follows rl_compose.run_spec for a preset without evaluations, so no Marin object storage is read. P40 uses four
eight-GPU nodes for the policy; the same 32 GPUs are eight four-GPU nodes here, and the engine's eight GPUs span two more.
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
PRESETS = ("speed-medium", "learning-medium")


def build_recipe(name: str) -> tuple[dict, int]:
    preset = SNOWBALL.by_name[name]
    if name == "learning-medium":
        # The preset evaluates on eval-lite (Marin object storage) and saves checkpoints; neither is available here.
        preset = replace(preset, evaluations=(), checkpoints="none", checkpoint_every=None, export="none", publishes=False)
    assert not preset.evaluations
    config = c._training_config(preset, (), preset.model)
    workload = preset.recipe.layout.workload
    _, in_flight = workload.batch_parameters(config)
    config = config.merge(RecipePatch.from_document({"trainer": {"rollout_buffer": {"max_in_flight": in_flight}}}))
    preset = preset.with_updates(config.trainer.max_steps)
    config = config.merge(RecipePatch(trainer=sections.Trainer(callbacks=workload.callbacks(preset, None, {}, 0, None))))
    recipe = SkyRLRecipe.from_document(config.to_skyrl()).to_skyrl()
    plan = replace(workload.plan, policy_num_nodes=32 // GPUS_PER_NODE, policy_num_gpus_per_node=GPUS_PER_NODE)
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
    parser.add_argument("--preset", choices=PRESETS, required=True)
    recipe, in_flight = build_recipe(parser.parse_args().preset)
    print(yaml.safe_dump(recipe, sort_keys=False))
    print("# in_flight", in_flight, "max_steps", recipe["trainer"]["max_steps"], file=sys.stderr)


if __name__ == "__main__":
    main()
