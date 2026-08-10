"""Build a metadata-consistent LeRobot subset without duplicating parquet data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/work/PARC2026_data/lerobot/physical-intelligence/libero"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/work/PARC2026_data/lerobot/physical-intelligence/libero-smoke"),
    )
    parser.add_argument("--episodes", type=int, default=914)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")

    episode_rows = [
        json.loads(line)
        for line in (source / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ][: args.episodes]
    if len(episode_rows) != args.episodes:
        raise ValueError(f"requested {args.episodes} episodes, found {len(episode_rows)} metadata rows")
    expected = list(range(args.episodes))
    actual = [row["episode_index"] for row in episode_rows]
    if actual != expected:
        raise ValueError("smoke subset must be a contiguous prefix so frame indices remain valid")

    missing = []
    for episode_index in expected:
        chunk = episode_index // 1000
        path = source / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} parquet files; first is {missing[0]}")

    (destination / "meta").mkdir(parents=True)
    for name in ("tasks.jsonl", "stats.json"):
        shutil.copy2(source / "meta" / name, destination / "meta" / name)

    info = json.loads((source / "meta" / "info.json").read_text())
    info["total_episodes"] = args.episodes
    info["total_frames"] = sum(row["length"] for row in episode_rows)
    info["total_chunks"] = (args.episodes + info["chunks_size"] - 1) // info["chunks_size"]
    info["splits"] = {"train": f"0:{args.episodes}"}
    (destination / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n")
    (destination / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows)
    )

    for episode_index in expected:
        chunk = episode_index // info["chunks_size"]
        src = source / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        dst = destination / "data" / f"chunk-{chunk:03d}" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)

    print(
        f"created {destination}: {args.episodes} episodes, "
        f"{info['total_frames']} frames (parquet files are hard-linked)"
    )


if __name__ == "__main__":
    main()
