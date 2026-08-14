"""Average ("model soup") LoRA params across multiple pi0.5-LIBERO checkpoints
trained with the same recipe/dataset, to reduce reliance on any single
checkpoint's run-to-run reproducibility noise (this project has observed
swings of +/-11pt on the local Track1 proxy eval between nominally-identical
reruns of the same seed/recipe -- see pi05_lora/README.md).

Runs on CPU (JAX_PLATFORMS=cpu) since restore_params loads the *entire*
model (frozen dense weights included, not just the small LoRA matrices) --
averaging 2-3 full copies plus a float32 upcast for the average itself can
exceed a single A100's 40GB, while the host's ~1TB system RAM handles it
trivially. No GPU/training required.

Example:
    python soup_checkpoints.py \\
        --checkpoint /path/to/ckpt_a/700 \\
        --checkpoint /path/to/ckpt_b/750:2.0 \\
        --checkpoint /path/to/ckpt_c/775 \\
        --out-dir /path/to/soup_output

Each --checkpoint is CHECKPOINT_STEP_DIR[:WEIGHT] (weight defaults to 1.0).
assets/ and a training_manifest.json documenting the soup composition are
copied/written into --out-dir; --assets-from picks which candidate's assets
to reuse (defaults to the first candidate; norm_stats should be identical
across candidates trained on the same dataset/base checkpoint).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def parse_checkpoint_arg(raw: str) -> tuple[Path, float]:
    if ":" in raw and not raw.startswith(("gs:", "s3:")):
        path_str, _, weight_str = raw.rpartition(":")
        return Path(path_str), float(weight_str)
    return Path(raw), 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        dest="checkpoints",
        help="CHECKPOINT_STEP_DIR[:WEIGHT], repeatable (each dir must contain a params/ subdir)",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--assets-from",
        type=int,
        default=0,
        help="index (0-based, in --checkpoint order) of which candidate's assets/ to copy",
    )
    parser.add_argument("--openpi-root", type=Path, default=Path("/tmp/openpi"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"--out-dir already exists: {args.out_dir}")

    candidates = [parse_checkpoint_arg(c) for c in args.checkpoints]
    for path, _ in candidates:
        if not (path / "params").is_dir():
            raise FileNotFoundError(f"no params/ subdir in checkpoint: {path}")
    for path, weight in candidates:
        if not math.isfinite(weight):
            raise ValueError(f"non-finite weight for {path}: {weight}")
    total_weight = sum(w for _, w in candidates)
    if not math.isfinite(total_weight) or total_weight == 0:
        raise ValueError(f"sum of --checkpoint weights must be finite and nonzero, got {total_weight}")
    if not 0 <= args.assets_from < len(candidates):
        raise ValueError(f"--assets-from={args.assets_from} out of range for {len(candidates)} checkpoints")
    assets_src = candidates[args.assets_from][0] / "assets"
    if not assets_src.is_dir():
        raise FileNotFoundError(
            f"--assets-from={args.assets_from} ({candidates[args.assets_from][0]}) has no assets/ directory"
        )

    sys.path.insert(0, str(args.openpi_root.resolve() / "src"))
    import jax
    import numpy as np
    import orbax.checkpoint as ocp
    from openpi.models import model as _model

    print("Restoring params:")
    param_trees = []
    for path, weight in candidates:
        param_trees.append(_model.restore_params(path / "params", restore_type=np.ndarray))
        print(f" - {path} weight={weight}")

    print("Averaging...")
    weights = [w for _, w in candidates]

    def avg_leaf(*xs: np.ndarray) -> np.ndarray:
        dtype = xs[0].dtype
        acc = sum(x.astype(np.float32) * w for x, w in zip(xs, weights)) / total_weight
        return acc.astype(dtype)

    averaged = jax.tree_util.tree_map(avg_leaf, *param_trees)

    args.out_dir.mkdir(parents=True)
    out_params_dir = args.out_dir / "params"
    print(f"Saving averaged params to {out_params_dir}")
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(out_params_dir, {"params": averaged})

    shutil.copytree(assets_src, args.out_dir / "assets")

    manifest = {
        "config": "pi05_libero_lora",
        "exp_name": args.out_dir.name,
        "note": "Model soup: weighted average of LoRA params across the checkpoints listed below.",
        "soup_candidates": [
            {"checkpoint": str(path), "weight": weight} for path, weight in candidates
        ],
    }
    (args.out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Done. Wrote soup checkpoint to {args.out_dir}")


if __name__ == "__main__":
    main()
