"""LeRobotLiberoDataConfig subclass that pushes PerturbationRobustnessAugmentation
onto the repack_transforms group. Sibling to augmented_data_config.py (which
wires in the texture/dark-scene-specific TexturePerturbation) -- kept as a
separate class so the two augmentation hypotheses can be selected
independently via train_pi05_lora.py's --standard-augment flag.
"""

from __future__ import annotations

import dataclasses

from openpi.training import config as train_config

from standard_augmentation import PerturbationRobustnessAugmentation


@dataclasses.dataclass(frozen=True)
class StandardAugmentedLeRobotLiberoDataConfig(train_config.LeRobotLiberoDataConfig):
    augment_prob: float = 0.5

    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        augmented_repack = base.repack_transforms.push(
            inputs=[PerturbationRobustnessAugmentation(prob=self.augment_prob)]
        )
        return dataclasses.replace(base, repack_transforms=augmented_repack)
