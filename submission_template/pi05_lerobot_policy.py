"""PARC adapter for the LeRobot/PyTorch pi0.5-LIBERO checkpoint."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import numpy as np


def _install_transformers_replacements(bundle_root: Path) -> None:
    """Install the small compatibility layer required by the OpenPI PyTorch port."""
    import transformers

    if transformers.__version__ != "4.53.2":
        raise RuntimeError(
            "LeRobot pi0.5 requires transformers==4.53.2, "
            f"but {transformers.__version__} is installed"
        )
    source = bundle_root / "transformers_replace"
    if not source.is_dir():
        raise FileNotFoundError(f"Transformers compatibility bundle not found: {source}")
    destination = Path(transformers.__file__).resolve().parent
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Match LeRobot's LIBERO quaternion-to-axis-angle conversion."""
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(1.0 - w * w, 0.0))
    if denominator <= 1e-10:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(w) / denominator)).astype(np.float32)


def _build_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )


def _image_tensor(image: np.ndarray, *, flip: bool = False):
    """Convert a raw LIBERO image like LeRobot's environment processor."""
    import torch

    array = np.asarray(image, dtype=np.uint8)
    if flip:
        array = array[::-1, ::-1]
    array = np.ascontiguousarray(array)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255.0)


class LeRobotPi05Policy:
    """One PyTorch pi0.5-LIBERO policy shared by every task instruction."""

    def __init__(self):
        here = Path(__file__).resolve().parent
        _install_transformers_replacements(here)

        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self._torch = torch
        requested_device = os.environ.get("LEROBOT_PI05_DEVICE", "auto")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LEROBOT_PI05_DEVICE requests CUDA, but PyTorch cannot access a GPU")
        self.device = torch.device(requested_device)
        self.flip_images = os.environ.get("LEROBOT_PI05_FLIP_IMAGES", "0").lower() not in {
            "0",
            "false",
            "no",
        }

        bundled_checkpoint = here / "pi05_lerobot_weights"
        self.checkpoint = Path(
            os.environ.get(
                "LEROBOT_PI05_CHECKPOINT",
                bundled_checkpoint
                if bundled_checkpoint.is_dir()
                else "/work/PARC2026_models/pi05_libero_finetuned_v044",
            )
        ).resolve()
        if not (self.checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(f"LeRobot pi0.5 checkpoint not found: {self.checkpoint}")

        config = PreTrainedConfig.from_pretrained(self.checkpoint)
        config.device = str(self.device)
        config.compile_model = False
        config.gradient_checkpointing = False
        config.n_action_steps = int(os.environ.get("LEROBOT_PI05_ACTION_CHUNK", "5"))
        if not 1 <= config.n_action_steps <= config.chunk_size:
            raise ValueError(
                f"LEROBOT_PI05_ACTION_CHUNK must be in [1, {config.chunk_size}], "
                f"got {config.n_action_steps}"
            )

        self.image_keys = [
            key
            for key, feature in config.input_features.items()
            if str(getattr(feature.type, "value", feature.type)).upper() == "VISUAL"
        ]
        if len(self.image_keys) < 2:
            raise ValueError(f"pi0.5 exposes {len(self.image_keys)} cameras; expected at least two")
        self.image_shapes = {key: tuple(config.input_features[key].shape) for key in self.image_keys}

        self.policy = PI05Policy.from_pretrained(
            self.checkpoint,
            config=config,
            local_files_only=True,
            # The v0.4.4 checkpoint stores the tied PaliGemma embedding once
            # under lm_head.weight. PyTorch reports the alias as a missing key
            # in strict mode even though both modules share the same storage.
            strict=False,
        )
        paligemma = self.policy.model.paligemma_with_expert.paligemma
        embeddings_are_tied = (
            paligemma.lm_head.weight.data_ptr()
            == paligemma.language_model.embed_tokens.weight.data_ptr()
        )
        if not embeddings_are_tied:
            raise RuntimeError("PaliGemma input and output embeddings are not tied")
        self.policy.eval()

        tokenizer_override = {}
        bundled_tokenizer = self.checkpoint / "tokenizer"
        tokenizer_path = os.environ.get("LEROBOT_PI05_TOKENIZER")
        if tokenizer_path:
            tokenizer_override = {"tokenizer_processor": {"tokenizer_name": tokenizer_path}}
        elif bundled_tokenizer.is_dir():
            tokenizer_override = {
                "tokenizer_processor": {"tokenizer_name": str(bundled_tokenizer)}
            }

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                **tokenizer_override,
            },
        )
        self.instruction = ""

    def reset(self, instruction: str = "") -> None:
        self.instruction = instruction
        self.policy.reset()

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        batch = {
            "observation.state": torch.from_numpy(_build_state(obs)),
            "task": self.instruction,
            self.image_keys[0]: _image_tensor(obs["agentview_image"], flip=self.flip_images),
            self.image_keys[1]: _image_tensor(
                obs["robot0_eye_in_hand_image"], flip=self.flip_images
            ),
        }
        for image_key in self.image_keys[2:]:
            batch[image_key] = torch.zeros(self.image_shapes[image_key], dtype=torch.float32)

        batch = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(batch))

        action = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        if action.shape != (7,):
            raise ValueError(f"LeRobot pi0.5 returned action shape {action.shape}; expected (7,)")
        if not np.isfinite(action).all():
            raise ValueError("LeRobot pi0.5 returned a non-finite action")
        return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
