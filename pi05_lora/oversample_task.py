"""Duplicate all episodes of one task N-1 extra times within an existing
openpi-format LeRobotDataset, in place, to bias the shared multi-task LoRA's
training-batch composition toward that task without touching the other
tasks' data or introducing any synthetic augmentation.

Uses the same pyarrow-preserving-schema-metadata approach as
extract_task_dataset.py (plain pandas read/write_parquet silently drops the
"huggingface" schema metadata key that tells HF `datasets` to auto-decode
the image columns -- see that script's docstring for the full story).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--task", required=True, help="exact task prompt string to duplicate")
    parser.add_argument("--extra-copies", type=int, required=True, help="N-1 extra copies to add")
    args = parser.parse_args()

    root = args.dataset_dir
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())

    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    stats = {row["episode_index"]: row for row in load_jsonl(root / "meta" / "episodes_stats.jsonl")}
    task_to_index = {row["task"]: row["task_index"] for row in load_jsonl(root / "meta" / "tasks.jsonl")}
    if args.task not in task_to_index:
        raise ValueError(f"task {args.task!r} not found in {root}/meta/tasks.jsonl")
    task_index = task_to_index[args.task]

    matched = [ep for ep in episodes if ep["tasks"][0] == args.task]
    if not matched:
        raise ValueError(f"no episodes found with task={args.task!r}")

    next_episode_index = info["total_episodes"]
    next_row_index = info["total_frames"]
    new_episode_rows = []
    new_stats_rows = []
    total_new_frames = 0
    total_new_episodes = 0

    for _ in range(args.extra_copies):
        for ep_meta in matched:
            src_idx = ep_meta["episode_index"]
            src_parquet = root / "data" / "chunk-000" / f"episode_{src_idx:06d}.parquet"
            table = pq.read_table(src_parquet)

            n = table.num_rows
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array([next_episode_index] * n, type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("task_index"),
                "task_index",
                pa.array([task_index] * n, type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                pa.array(range(next_row_index, next_row_index + n), type=pa.int64()),
            )
            dst_parquet = root / "data" / "chunk-000" / f"episode_{next_episode_index:06d}.parquet"
            pq.write_table(table, dst_parquet)

            ep_row = dict(ep_meta)
            ep_row["episode_index"] = next_episode_index
            new_episode_rows.append(ep_row)

            stats_row = dict(stats[src_idx])
            stats_row["episode_index"] = next_episode_index
            new_stats_rows.append(stats_row)

            next_row_index += n
            next_episode_index += 1
            total_new_frames += n
            total_new_episodes += 1

    append_jsonl(root / "meta" / "episodes.jsonl", new_episode_rows)
    append_jsonl(root / "meta" / "episodes_stats.jsonl", new_stats_rows)

    info["total_episodes"] += total_new_episodes
    info["total_frames"] += total_new_frames
    info["splits"]["train"] = f"0:{info['total_episodes']}"
    info_path.write_text(json.dumps(info, indent=4))

    print(
        f"added {total_new_episodes} duplicate episodes ({args.extra_copies}x{len(matched)}), "
        f"{total_new_frames} frames -> {root} now has {info['total_episodes']} episodes total"
    )


if __name__ == "__main__":
    main()
