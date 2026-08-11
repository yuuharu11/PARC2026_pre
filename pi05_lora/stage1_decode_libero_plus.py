"""Decode `lerobot/libero_plus` (v3.0, packed video, fps=20) episodes into
per-episode .npz files that stage2 can feed into openpi's pinned (old) lerobot
to build a v2.0-style dataset. Runs in the main venv (modern lerobot 0.4.4),
which is the only one that reads this repo's v3.0 metadata/video layout.

Kept deliberately dependency-light: reads the data/meta parquet files with
pandas directly (bypassing lerobot's LeRobotDataset/HF `datasets` loader,
which fails on this repo's newer "List" feature-type metadata -- see
conversation notes), and decodes video frames with lerobot's own
video_utils.decode_video_frames helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets import video_utils

ROOT = Path("/work/PARC2026_data/lerobot/lerobot/libero_plus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, default=None)
    parser.add_argument("--end-episode", type=int, default=None, help="exclusive")
    parser.add_argument(
        "--episode-list-file",
        type=Path,
        default=None,
        help="optional .npy of explicit episode indices; overrides --start/--end-episode",
    )
    parser.add_argument("--tolerance-s", type=float, default=1e-3)
    parser.add_argument("--video-backend", default="pyav")
    args = parser.parse_args()
    if args.episode_list_file is None and (args.start_episode is None or args.end_episode is None):
        parser.error("either --episode-list-file or both --start-episode/--end-episode are required")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    episodes_meta = pd.read_parquet(args.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    episodes_meta = episodes_meta.set_index("episode_index", drop=False)

    data_cache: dict[tuple[int, int], pd.DataFrame] = {}

    def get_data_file(chunk_index: int, file_index: int) -> pd.DataFrame:
        key = (chunk_index, file_index)
        if key not in data_cache:
            path = args.root / "data" / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
            data_cache[key] = pd.read_parquet(path)
        return data_cache[key]

    if args.episode_list_file is not None:
        episode_indices = np.load(args.episode_list_file).tolist()
    else:
        episode_indices = range(args.start_episode, args.end_episode)

    for episode_index in episode_indices:
        out_path = args.out_dir / f"episode_{episode_index:06d}.npz"
        if out_path.exists():
            continue

        ep = episodes_meta.loc[episode_index]
        rows = get_data_file(int(ep["data/chunk_index"]), int(ep["data/file_index"])).iloc[
            int(ep["dataset_from_index"]) : int(ep["dataset_to_index"])
        ]
        assert (rows["episode_index"] == episode_index).all()

        state = np.stack(rows["observation.state"].to_numpy())
        action = np.stack(rows["action"].to_numpy())
        local_ts = rows["timestamp"].to_numpy(dtype=np.float64)

        frames = {}
        for cam in ("front", "wrist"):
            chunk_i = int(ep[f"videos/observation.images.{cam}/chunk_index"])
            file_i = int(ep[f"videos/observation.images.{cam}/file_index"])
            from_ts = float(ep[f"videos/observation.images.{cam}/from_timestamp"])
            video_path = (
                args.root
                / "videos"
                / f"observation.images.{cam}"
                / f"chunk-{chunk_i:03d}"
                / f"file-{file_i:03d}.mp4"
            )
            query_ts = (local_ts + from_ts).tolist()
            decoded = video_utils.decode_video_frames(
                video_path, query_ts, args.tolerance_s, backend=args.video_backend
            )
            # (T, C, H, W) float32 [0,1] -> (T, H, W, C) uint8
            frames[cam] = (decoded.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)

        task = ep["tasks"][0] if len(ep["tasks"]) else ""

        np.savez_compressed(
            out_path,
            image=frames["front"],
            wrist_image=frames["wrist"],
            state=state.astype(np.float32),
            actions=action.astype(np.float32),
            task=task,
        )
        print(f"episode {episode_index}: {len(rows)} frames -> {out_path}")


if __name__ == "__main__":
    main()
