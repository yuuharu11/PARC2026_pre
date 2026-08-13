"""Training-time perturbation robustness, modeled directly on the four
perturbation axes the competition's own eval harness defines
(pipeline/config.py's PerturbationConfig: robot_init_pos_noise,
object_pos_noise, camera_view_shift, action_noise, observation_noise).
Track1's registered PerturbationConfig is all-zeros (no perturbation), but
compe/t1/README.md states the real production evaluation uses a different,
non-public task set -- if that set (or any harder Track) turns those knobs
on, a policy that never saw analogous variation during training would be
brittle to it. This is a different hypothesis from texture_perturbation.py
(which targets one specific known visual gap, dark scenes, and was found to
underperform training on real data alone at every tested strength): instead
of fixing a known gap, this adds generic robustness along the same axes the
harness itself perturbs, uniformly across all tasks.

Four independent perturbations, applied via openpi's repack_transforms
(training data only -- per config.py's own comment, repack_transforms never
run at inference/eval time):
  - state noise      -> analog of robot_init_pos_noise / object_pos_noise
                         (the policy's belief about robot/object state)
  - camera view shift -> analog of camera_view_shift (small crop/pan)
  - image pixel noise -> analog of observation_noise (sensor noise)
  - action noise       -> analog of action_noise (actuation imprecision);
                         perturbs the training *target*, not just the input,
                         so the policy learns a slightly noise-tolerant
                         action distribution rather than overfitting to
                         exact demonstrated trajectories.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms


@dataclasses.dataclass(frozen=True)
class PerturbationRobustnessAugmentation(transforms.DataTransformFn):
    prob: float = 0.5
    image_keys: tuple[str, ...] = ("observation/image", "observation/wrist_image")
    state_key: str = "observation/state"
    actions_key: str = "actions"

    # camera_view_shift analog: crop scale range (fraction of original H/W kept)
    crop_scale: tuple[float, float] = (0.9, 1.0)
    # observation_noise analog: additive Gaussian pixel noise, std in 0-1 scale
    # (LeRobot stores images as float32 CHW already normalized to [0, 1] --
    # see openpi/src/openpi/policies/libero_policy.py's _parse_image, which
    # does the uint8/HWC conversion downstream of this transform)
    image_noise_std: float = 6.0 / 255.0
    # robot_init_pos_noise / object_pos_noise analog: additive Gaussian noise
    # on the normalized proprioceptive state vector
    state_noise_std: float = 0.02
    # action_noise analog: additive Gaussian noise on the action targets
    action_noise_std: float = 0.02

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        out = dict(data)

        for key in self.image_keys:
            if key not in out:
                continue
            img = np.asarray(out[key]).astype(np.float32)
            # CHW: crop the last two axes (H, W), keep channel axis (0) intact.
            h, w = img.shape[-2:]

            # camera_view_shift analog: random crop; downstream ResizeImages
            # (image_tools.resize_with_pad) accepts arbitrary input sizes and
            # resizes/pads back to the model's target size.
            scale = np.random.uniform(*self.crop_scale)
            ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
            top = np.random.randint(0, h - ch + 1)
            left = np.random.randint(0, w - cw + 1)
            img = img[:, top : top + ch, left : left + cw]

            # observation_noise analog: sensor-noise-like pixel jitter.
            img = img + np.random.normal(0.0, self.image_noise_std, size=img.shape).astype(np.float32)
            out[key] = np.clip(img, 0.0, 1.0).astype(np.float32)

        if self.state_key in out:
            state = np.asarray(out[self.state_key]).astype(np.float32)
            out[self.state_key] = state + np.random.normal(0.0, self.state_noise_std, size=state.shape)

        if self.actions_key in out:
            actions = np.asarray(out[self.actions_key]).astype(np.float32)
            out[self.actions_key] = actions + np.random.normal(0.0, self.action_noise_std, size=actions.shape)

        return out
