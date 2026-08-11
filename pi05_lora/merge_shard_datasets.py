"""Merge episodes from N independently-built shard LeRobotDatasets into an
existing master dataset, on disk, without going through LeRobotDataset's
per-frame add_frame()/save_episode() API (that path is what makes stage2
slow: ~45s/episode regardless of writer thread/process config).

Key simplification: task_index numbers are shard-local and meaningless
across shards, but each shard's own meta/episodes.jsonl already records the
authoritative task *string* per episode. So we don't reconcile task_index
between shards at all -- we just look up each episode's task string in the
existing master vocabulary (meta/tasks.jsonl), which already covers every
task used here (verified: no new tasks are introduced by the shards), and
write that looked-up index into the copied parquet's task_index column.

Only episode_index and the cumulative `index` column need renumbering, both
straightforward sequential continuations from the master's current totals.
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
    parser.add_argument("--master-dir", type=Path, required=True)
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    master = args.master_dir
    info_path = master / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    chunks_size = info["chunks_size"]

    task_to_index = {row["task"]: row["task_index"] for row in load_jsonl(master / "meta" / "tasks.jsonl")}

    next_episode_index = info["total_episodes"]
    next_row_index = info["total_frames"]
    new_episode_rows = []
    new_stats_rows = []
    total_new_frames = 0
    total_new_episodes = 0

    for shard_dir in args.shard_dirs:
        shard_episodes = {row["episode_index"]: row for row in load_jsonl(shard_dir / "meta" / "episodes.jsonl")}
        shard_stats = {row["episode_index"]: row for row in load_jsonl(shard_dir / "meta" / "episodes_stats.jsonl")}

        for local_idx in sorted(shard_episodes):
            ep_meta = shard_episodes[local_idx]
            task_str = ep_meta["tasks"][0]
            if task_str not in task_to_index:
                raise ValueError(
                    f"unknown task {task_str!r} in {shard_dir} (not in master vocabulary; "
                    "this script assumes no new tasks)"
                )
            global_task_index = task_to_index[task_str]

            src_parquet = shard_dir / "data" / "chunk-000" / f"episode_{local_idx:06d}.parquet"
            table = pq.read_table(src_parquet)
            assert table.column("episode_index")[0].as_py() == local_idx

            n = table.num_rows
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array([next_episode_index] * n, type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("task_index"),
                "task_index",
                pa.array([global_task_index] * n, type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                pa.array(range(next_row_index, next_row_index + n), type=pa.int64()),
            )

            # episode_chunk follows the same `episode_index // chunks_size`
            # convention info.json's data_path template implies -- crossing a
            # chunks_size boundary (e.g. episode 1000 at the default 1000)
            # without this rolls episodes into the wrong directory, which
            # LeRobotDataset's own path resolution then can't find, throwing
            # an assertion error deep in its __init__ on next load. Verified
            # by hitting exactly this merging a 780-episode dataset past 1000.
            episode_chunk = next_episode_index // chunks_size
            chunk_dir = master / "data" / f"chunk-{episode_chunk:03d}"
            dst_parquet = chunk_dir / f"episode_{next_episode_index:06d}.parquet"
            if not args.dry_run:
                chunk_dir.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, dst_parquet)

            ep_row = dict(ep_meta)
            ep_row["episode_index"] = next_episode_index
            new_episode_rows.append(ep_row)

            stats_row = dict(shard_stats[local_idx])
            stats_row["episode_index"] = next_episode_index
            new_stats_rows.append(stats_row)

            next_row_index += n
            next_episode_index += 1
            total_new_frames += n
            total_new_episodes += 1

    print(f"merging {total_new_episodes} episodes, {total_new_frames} frames from {len(args.shard_dirs)} shards")

    if args.dry_run:
        print("dry run: no files written")
        return

    append_jsonl(master / "meta" / "episodes.jsonl", new_episode_rows)
    append_jsonl(master / "meta" / "episodes_stats.jsonl", new_stats_rows)

    info["total_episodes"] += total_new_episodes
    info["total_frames"] += total_new_frames
    info["splits"]["train"] = f"0:{info['total_episodes']}"
    info["total_chunks"] = max(1, (info["total_episodes"] - 1) // chunks_size + 1)
    info_path.write_text(json.dumps(info, indent=4))

    print(f"done: master now has {info['total_episodes']} episodes, {info['total_frames']} frames")


if __name__ == "__main__":
    main()
