"""Restore a pi0.5 LoRA checkpoint and run one finite-action inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, default=Path("/tmp/openpi"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/work/PARC2026_training/checkpoints/pi05_libero_lora/smoke_20step/19"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    sys.path.insert(0, str(args.openpi_root.resolve() / "src"))

    from openpi.models import pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as train_config

    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    config = train_config.TrainConfig(
        name="pi05_libero_lora_inference",
        model=model,
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
    policy = policy_config.create_trained_policy(config, checkpoint)
    output = policy.infer(
        {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "pick up the tomato sauce and place it in the basket",
        }
    )
    actions = np.asarray(output["actions"])
    if actions.shape != (10, 7):
        raise ValueError(f"unexpected action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("checkpoint produced non-finite actions")
    print(f"checkpoint={checkpoint}")
    print(f"actions.shape={actions.shape} min={actions.min():.6f} max={actions.max():.6f}")
    print("first_action=" + np.array2string(actions[0], precision=6))


if __name__ == "__main__":
    main()
