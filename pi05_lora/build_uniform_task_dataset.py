"""Build a dataset with exactly N episodes per task from an existing
openpi-format dataset, filtering evenly across ALL tasks (no special
treatment for any task, including Track1's). Uses pyarrow directly (not
pandas) to preserve the parquet schema's "huggingface" image-decode
metadata -- see merge_shard_datasets.py's docstring for why this matters.
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, required=True)
    parser.add_argument("--dst-dir", type=Path, required=True)
    parser.add_argument("--n-per-task", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src, dst = args.src_dir, args.dst_dir
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        import shutil

        shutil.rmtree(dst)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir(parents=True)

    src_info = json.loads((src / "meta" / "info.json").read_text())
    chunks_size = src_info["chunks_size"]
    src_episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    src_stats = {row["episode_index"]: row for row in load_jsonl(src / "meta" / "episodes_stats.jsonl")}

    by_task: dict[str, list[dict]] = {}
    for ep in src_episodes:
        by_task.setdefault(ep["tasks"][0], []).append(ep)

    tasks_sorted = sorted(by_task.keys())
    new_episode_rows = []
    new_stats_rows = []
    next_row_index = 0
    total_frames = 0
    task_index_map = {t: i for i, t in enumerate(tasks_sorted)}

    new_idx = 0
    shortfalls = []
    for task in tasks_sorted:
        eps = sorted(by_task[task], key=lambda e: e["episode_index"])[: args.n_per_task]
        if len(eps) < args.n_per_task:
            shortfalls.append((task, len(eps)))
        for ep_meta in eps:
            local_idx = ep_meta["episode_index"]
            src_chunk = local_idx // chunks_size
            src_parquet = src / "data" / f"chunk-{src_chunk:03d}" / f"episode_{local_idx:06d}.parquet"
            table = pq.read_table(src_parquet)
            assert table.column("episode_index")[0].as_py() == local_idx

            n = table.num_rows
            table = table.set_column(
                table.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array([new_idx] * n, type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("task_index"),
                "task_index",
                pa.array([task_index_map[task]] * n, type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("index"),
                "index",
                pa.array(range(next_row_index, next_row_index + n), type=pa.int64()),
            )
            pq.write_table(table, dst / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet")

            ep_row = dict(ep_meta)
            ep_row["episode_index"] = new_idx
            new_episode_rows.append(ep_row)

            stats_row = dict(src_stats[local_idx])
            stats_row["episode_index"] = new_idx
            new_stats_rows.append(stats_row)

            next_row_index += n
            total_frames += n
            new_idx += 1

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for row in new_episode_rows:
            f.write(json.dumps(row) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in new_stats_rows:
            f.write(json.dumps(row) + "\n")
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        for t, i in task_index_map.items():
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")

    info = dict(src_info)
    info["total_episodes"] = new_idx
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks_sorted)
    info["total_chunks"] = 1
    info["splits"]["train"] = f"0:{new_idx}"
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"built {new_idx} episodes ({len(tasks_sorted)} tasks), {total_frames} frames -> {dst}")
    if shortfalls:
        print(f"NOTE: {len(shortfalls)} tasks had fewer than {args.n_per_task} episodes available:")
        for t, n in shortfalls:
            print(f"  {n}/{args.n_per_task}: {t}")


if __name__ == "__main__":
    main()
