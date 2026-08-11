"""Build an openpi-compatible (old lerobot v2.0 layout, flat keys) LeRobot
dataset from the .npz episodes produced by stage1_decode_libero_plus.py.

Must run under openpi's own pinned lerobot (/tmp/openpi/.venv/bin/python) --
that old version is what train_pi05_lora.py / pi05_policy.py actually load
datasets with, and it cannot read lerobot/libero_plus's newer v3.0 layout
directly (hence the stage1/stage2 split).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, default=Path("/tmp/openpi"))
    parser.add_argument("--staged-dir", type=Path, required=True)
    parser.add_argument("--dataset-home", type=Path, default=Path("/work/PARC2026_data/lerobot"))
    parser.add_argument("--repo-id", default="local/libero_plus_openpi")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="add episodes to an existing dataset at --repo-id instead of creating fresh",
    )
    parser.add_argument("--image-writer-threads", type=int, default=0)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument(
        "--episode-list-file",
        type=Path,
        default=None,
        help="optional .npy of explicit episode indices (matched against staged npz filenames)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.openpi_root.resolve() / "src"))

    import os

    # Must be set before importing lerobot_dataset: HF_LEROBOT_HOME is read
    # once at import time into a module-level constant, so setting it after
    # import (as this script used to) silently writes to the default
    # ~/.cache/huggingface/lerobot instead of --dataset-home.
    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_home.resolve())

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = args.dataset_home.resolve() / args.repo_id
    if args.append:
        if not output_path.exists():
            raise FileNotFoundError(f"--append given but {output_path} does not exist")
        dataset = LeRobotDataset(repo_id=args.repo_id, root=output_path)
        if args.image_writer_threads or args.image_writer_processes:
            dataset.start_image_writer(args.image_writer_processes, args.image_writer_threads)
        print(f"appending to existing dataset: {output_path} (currently {dataset.meta.total_episodes} episodes)")
    else:
        if output_path.exists():
            if not args.overwrite:
                raise FileExistsError(f"{output_path} exists; pass --overwrite or --append")
            shutil.rmtree(output_path)

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            robot_type="panda",
            fps=args.fps,
            features={
                "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {
                    "dtype": "image",
                    "shape": (256, 256, 3),
                    "names": ["height", "width", "channel"],
                },
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            },
            image_writer_threads=args.image_writer_threads,
            image_writer_processes=args.image_writer_processes,
        )

    if args.episode_list_file is not None:
        wanted = np.load(args.episode_list_file).tolist()
        npz_paths = [args.staged_dir / f"episode_{i:06d}.npz" for i in wanted]
        missing = [p for p in npz_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing staged episodes: {missing[:5]}...")
    else:
        npz_paths = sorted(args.staged_dir.glob("episode_*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"no staged episodes found under {args.staged_dir}")

    for npz_path in npz_paths:
        with np.load(npz_path, allow_pickle=True) as ep:
            task = str(ep["task"])
            n_frames = ep["image"].shape[0]
            for i in range(n_frames):
                dataset.add_frame(
                    {
                        "image": ep["image"][i],
                        "wrist_image": ep["wrist_image"][i],
                        "state": ep["state"][i],
                        "actions": ep["actions"][i],
                        "task": task,
                    }
                )
            dataset.save_episode()
        print(f"added {npz_path.name} ({n_frames} frames, task={task!r})")

    print(f"done: {output_path} ({len(npz_paths)} episodes)")


if __name__ == "__main__":
    main()
