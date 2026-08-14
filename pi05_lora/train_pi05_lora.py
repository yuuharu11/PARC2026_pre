"""Conservative LoRA fine-tuning for the official pi0.5-LIBERO checkpoint.

The script deliberately starts from the already evaluated ``pi05_libero``
checkpoint, freezes the dense Gemma weights, and trains only the LoRA and
small non-LLM parameters.  Dataset files and checkpoints live outside the git
repository by default so a training run cannot bloat a submission branch.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, default=Path("/tmp/openpi"))
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("/tmp/openpi-data/openpi-assets/checkpoints/pi05_libero"),
    )
    parser.add_argument(
        "--dataset-home",
        type=Path,
        default=Path("/work/PARC2026_data/lerobot"),
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="physical-intelligence/libero",
        help="LeRobot repo id below --dataset-home (use libero-smoke for the partial smoke set)",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/work/PARC2026_training/checkpoints"),
    )
    parser.add_argument("--exp-name", default="smoke_20step")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--peak-lr", type=float, default=1e-5)
    parser.add_argument(
        "--jerk-loss-weight",
        type=float,
        default=0.0,
        help="training-only weight for the reconstructed-action second-difference penalty (default: off)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="apply TexturePerturbation (dark/blue-tint/desaturate) to training images only; "
        "the tomato-sauce task gets boosted, dark-skewed augmentation (see augmented_data_config.py)",
    )
    parser.add_argument(
        "--augment-prob",
        type=float,
        default=0.15,
        help="augmentation probability for non-boosted tasks (boosted tasks use --boosted-prob)",
    )
    parser.add_argument(
        "--boosted-prob",
        type=float,
        default=0.9,
        help="augmentation probability for the tomato-sauce task specifically",
    )
    parser.add_argument(
        "--boosted-dark-weight",
        type=float,
        default=0.7,
        help="fraction of boosted-task augmentations that use dark mode specifically",
    )
    parser.add_argument(
        "--standard-augment",
        action="store_true",
        help="apply StandardImageAugmentation (random crop + brightness/contrast/saturation "
        "jitter) instead of TexturePerturbation; mutually exclusive with --augment "
        "(see standard_augmentation.py)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="enable wandb logging (requires WANDB_API_KEY env var or prior `wandb login`)",
    )
    return parser.parse_args()


def load_train_module(openpi_root: Path):
    train_path = openpi_root / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_train_entrypoint", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import openpi trainer from {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    if not math.isfinite(args.jerk_loss_weight) or args.jerk_loss_weight < 0:
        raise ValueError("jerk-loss-weight must be a finite, non-negative number")

    openpi_root = args.openpi_root.resolve()
    base_checkpoint = args.base_checkpoint.resolve()
    dataset_home = args.dataset_home.resolve()
    dataset_root = dataset_home / Path(args.dataset_repo_id)
    hf_cache = dataset_home.parent / "huggingface"
    if not (openpi_root / "src" / "openpi").is_dir():
        raise FileNotFoundError(f"openpi source not found: {openpi_root}")
    if not (base_checkpoint / "params").is_dir():
        raise FileNotFoundError(f"base checkpoint not found: {base_checkpoint}")
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"LIBERO dataset not found: {dataset_root}")
    if args.augment and args.standard_augment:
        raise ValueError("--augment and --standard-augment are mutually exclusive")

    os.environ["HF_LEROBOT_HOME"] = str(dataset_home)
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    if not args.wandb:
        os.environ.setdefault("WANDB_MODE", "disabled")
    sys.path.insert(0, str(openpi_root / "src"))

    import flax.nnx as nnx

    from openpi.shared import nnx_utils
    from openpi.training import config as train_config
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from augmented_data_config import AugmentedLeRobotLiberoDataConfig
    from jerk_loss import JerkPi0Config
    from standard_augmented_data_config import StandardAugmentedLeRobotLiberoDataConfig

    model = JerkPi0Config(
        pi05=True,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        jerk_loss_weight=args.jerk_loss_weight,
    )
    warmup_steps = min(100, max(1, args.steps // 10))
    if args.augment:
        data_config_cls = AugmentedLeRobotLiberoDataConfig
    elif args.standard_augment:
        data_config_cls = StandardAugmentedLeRobotLiberoDataConfig
    else:
        data_config_cls = train_config.LeRobotLiberoDataConfig
    data_config_kwargs = dict(
        repo_id=args.dataset_repo_id,
        assets=train_config.AssetsConfig(
            assets_dir=str(base_checkpoint / "assets"),
            asset_id="physical-intelligence/libero",
        ),
        base_config=train_config.DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
    )
    if args.augment:
        data_config_kwargs["augment_prob"] = args.augment_prob
        data_config_kwargs["boosted_prob"] = args.boosted_prob
        data_config_kwargs["boosted_dark_weight"] = args.boosted_dark_weight
    elif args.standard_augment:
        data_config_kwargs["augment_prob"] = args.augment_prob
    config = train_config.TrainConfig(
        name="pi05_libero_lora",
        exp_name=args.exp_name,
        model=model,
        data=data_config_cls(**data_config_kwargs),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(base_checkpoint / "params")),
        # Freeze every dense parameter.  The stock low-memory recipe also
        # updates vision/projection weights (~467M params); for this already
        # strong checkpoint we train only LoRA matrices (~50M params) to
        # minimize catastrophic forgetting on unseen LIBERO-plus tasks.
        freeze_filter=nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*lora.*")),
        ),
        batch_size=args.batch_size,
        num_workers=2,
        num_train_steps=args.steps,
        log_interval=1 if args.steps <= 20 else 10,
        save_interval=min(args.save_interval, args.steps),
        keep_period=min(args.save_interval, args.steps),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=max(args.steps, warmup_steps + 1),
            decay_lr=args.peak_lr / 10,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        checkpoint_base_dir=str(args.checkpoint_root.resolve()),
        seed=args.seed,
        overwrite=args.overwrite,
        wandb_enabled=args.wandb,
    )

    run_dir = config.checkpoint_dir
    manifest = {
        "config": config.name,
        "exp_name": config.exp_name,
        "base_checkpoint": str(base_checkpoint),
        "dataset": f"{args.dataset_repo_id}@v2.0",
        "dataset_root": str(dataset_root),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "peak_lr": args.peak_lr,
        "jerk_loss_weight": args.jerk_loss_weight,
        "seed": args.seed,
        "augment": args.augment,
        "augment_prob": args.augment_prob if args.augment else None,
        "standard_augment": args.standard_augment,
        "standard_augment_prob": args.augment_prob if args.standard_augment else None,
        "model": dataclasses.asdict(model),
    }
    train_module = load_train_module(openpi_root)
    train_module.main(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
