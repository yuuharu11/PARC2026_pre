"""P​ARC adapter for the official single-checkpoint pi0.5-LIBERO policy."""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np


def _axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(quat[3]) / denominator)).astype(np.float32)


def _prepare_image(image: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the official LIBERO client: rotate, then resize with padding."""
    from openpi_client import image_tools

    rotated = np.ascontiguousarray(np.asarray(image, dtype=np.uint8)[::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated, size, size))


class Pi05Policy:
    """One pi0.5-LIBERO checkpoint shared by every instruction and task."""

    def __init__(self):
        # The development tree may vendor a different SmolVLA-era ``lerobot``
        # beside this file. openpi needs its pinned package. The final pi0.5
        # archive vendors the compatible package (identified by ``common``),
        # in which case the submission directory must remain importable.
        here = Path(__file__).resolve().parent
        if not (here / "lerobot" / "common").is_dir():
            sys.path[:] = [
                entry
                for entry in sys.path
                if Path(entry or os.getcwd()).resolve() != here
            ]
        from openpi.policies import policy_config
        from openpi.training import config as train_config

        checkpoint = Path(
            os.environ.get(
                "PI05_CHECKPOINT",
                here / "pi05_weights",
            )
        ).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"pi0.5-LIBERO checkpoint not found at {checkpoint}")

        variant = os.environ.get("PI05_VARIANT")
        if variant is None:
            # Training writes the manifest beside its numbered checkpoints;
            # final packaging copies it into the checkpoint root.
            variant = (
                "lora"
                if (checkpoint / "training_manifest.json").is_file()
                or (checkpoint.parent / "training_manifest.json").is_file()
                else "base"
            )
        variant = variant.lower()
        if variant == "lora":
            from openpi.models import pi0_config

            lora_model = pi0_config.Pi0Config(
                pi05=True,
                action_horizon=10,
                discrete_state_input=False,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            )
            config = train_config.TrainConfig(
                name="pi05_libero_lora_inference",
                model=lora_model,
                data=train_config.LeRobotLiberoDataConfig(
                    repo_id="physical-intelligence/libero",
                    assets=train_config.AssetsConfig(
                        assets_dir=str(checkpoint / "assets"),
                        asset_id="physical-intelligence/libero",
                    ),
                    base_config=train_config.DataConfig(prompt_from_task=True),
                    extra_delta_transform=False,
                ),
            )
        elif variant == "base":
            config = train_config.get_config("pi05_libero")
        else:
            raise ValueError(f"unsupported PI05_VARIANT={variant!r}; expected 'base' or 'lora'")

        self._policy = policy_config.create_trained_policy(config, checkpoint)
        self._instruction = ""
        self._queue: deque[np.ndarray] = deque()
        self._execute = int(os.environ.get("PI05_ACTION_CHUNK", "5"))

        # JAX compilation takes longer than the 10-second request limit. Compile
        # before uvicorn exposes /health; load + compile remains within the
        # competition's 120-second startup allowance on the measured A100.
        if os.environ.get("PI05_WARMUP", "1") not in {"0", "false", "False"}:
            self._policy.infer(
                {
                    "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
                    "observation/state": np.zeros(8, dtype=np.float32),
                    "prompt": "do something",
                }
            )

    def reset(self, instruction: str = "") -> None:
        self._instruction = instruction
        self._queue.clear()

    def _input(self, obs: dict[str, np.ndarray]) -> dict:
        state = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                _axis_angle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            ]
        )
        return {
            "observation/image": _prepare_image(obs["agentview_image"]),
            "observation/wrist_image": _prepare_image(obs["robot0_eye_in_hand_image"]),
            "observation/state": state,
            "prompt": self._instruction,
        }

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if not self._queue:
            actions = np.asarray(self._policy.infer(self._input(obs))["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"pi0.5 returned invalid action chunk shape {actions.shape}")
            self._queue.extend(actions[: self._execute])
        action = self._queue.popleft()
        if not np.isfinite(action).all():
            raise ValueError("pi0.5 returned a non-finite action")
        return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
