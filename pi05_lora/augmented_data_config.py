"""LeRobotLiberoDataConfig subclass that pushes TexturePerturbation onto the
repack_transforms group. Kept out of openpi's own source tree entirely --
this only composes on top of the stock config via inheritance, so the
upstream (already-verified) pipeline is untouched.
"""

from __future__ import annotations

import dataclasses

from openpi.training import config as train_config

from texture_perturbation import TexturePerturbation

# Exact LIBERO prompt string for the Track1-graded tomato-sauce task. Must
# match meta/tasks.jsonl in the training dataset verbatim (task_index 20 in
# libero_plus_openpi_subset320) -- do NOT match "alphabet soup and the
# tomato sauce" (task_index 37), which is a different, ungraded task.
TOMATO_TASK_PROMPT = "pick up the tomato sauce and place it in the basket"


@dataclasses.dataclass(frozen=True)
class AugmentedLeRobotLiberoDataConfig(train_config.LeRobotLiberoDataConfig):
    augment_prob: float = 0.15
    boosted_prompts: tuple[str, ...] = (TOMATO_TASK_PROMPT,)
    boosted_prob: float = 0.9
    boosted_dark_weight: float = 0.7

    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        augmented_repack = base.repack_transforms.push(
            inputs=[
                TexturePerturbation(
                    prob=self.augment_prob,
                    boosted_prompts=self.boosted_prompts,
                    boosted_prob=self.boosted_prob,
                    boosted_dark_weight=self.boosted_dark_weight,
                )
            ]
        )
        return dataclasses.replace(base, repack_transforms=augmented_repack)
