"""Quickly scan brightness of the first frame of every remaining (unused in
training) tomato-sauce episode, to check whether any darker texture-
perturbation variants (like the eval-time "table_27" one) exist among the
training-available demonstrations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lerobot.datasets import video_utils

ROOT_ = __import__("pathlib").Path("/work/PARC2026_data/lerobot/lerobot/libero_plus")


def main() -> None:
    episodes_meta = pd.read_parquet(ROOT_ / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    episodes_meta = episodes_meta.set_index("episode_index", drop=False)

    remaining = np.load("/work/PARC2026_data/tomato_remaining_episodes.npy")

    results = []
    for episode_index in remaining:
        ep = episodes_meta.loc[int(episode_index)]
        chunk_i = int(ep["videos/observation.images.front/chunk_index"])
        file_i = int(ep["videos/observation.images.front/file_index"])
        from_ts = float(ep["videos/observation.images.front/from_timestamp"])
        video_path = (
            ROOT_ / "videos" / "observation.images.front" / f"chunk-{chunk_i:03d}" / f"file-{file_i:03d}.mp4"
        )
        decoded = video_utils.decode_video_frames(video_path, [from_ts], 1e-3, backend="pyav")
        frame = (decoded[0].permute(1, 2, 0).numpy() * 255)
        brightness = float(frame.mean())
        results.append((int(episode_index), brightness))
        print(f"episode {episode_index}: brightness={brightness:.1f}")

    results.sort(key=lambda x: x[1])
    print("\n--- darkest 15 ---")
    for idx, b in results[:15]:
        print(idx, b)
    print("\n--- brightness distribution ---")
    vals = np.array([b for _, b in results])
    print(f"min={vals.min():.1f} max={vals.max():.1f} mean={vals.mean():.1f} median={np.median(vals):.1f}")
    print(f"count below 50: {(vals < 50).sum()}, below 40: {(vals < 40).sum()}, below 35: {(vals < 35).sum()}")

    np.save(
        "/work/PARC2026_data/tomato_remaining_brightness.npy",
        np.array(results, dtype=[("episode_index", "i8"), ("brightness", "f8")]),
    )


if __name__ == "__main__":
    main()
