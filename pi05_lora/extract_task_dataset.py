"""Build a single-task LeRobotDataset by filesystem-level extraction from an
existing openpi-format dataset (same v2.1 flat-key layout produced by
stage2_build_openpi_dataset.py / merge_shard_datasets.py).

Used to isolate one task (e.g. tomato-sauce) for standalone LoRA training,
so augmentation-strength experiments aren't confounded by the other three
Track1 tasks sharing the same LoRA weights.

Renumbers episode_index/index sequentially from 0 and remaps task_index to
a single-entry tasks.jsonl (index 0), since PromptFromLeRobotTask only
needs task_index -> prompt string to resolve correctly, not the original
global index.

IMPORTANT: uses pyarrow directly (not pandas' read_parquet/to_parquet,
which silently drops the parquet schema's top-level "huggingface" metadata
key -- the JSON blob that tells HF `datasets` a column is an Image feature
to auto-decode. Losing it makes every image column load as a raw
{"bytes", "path"} dict instead of a decoded tensor, and training crashes
deep inside lerobot's hf_transform_to_torch with "Could not infer dtype of
dict". Verified this by inspecting `pyarrow.parquet.read_table(...).schema.metadata`
on an original stage2-written file vs a pandas-round-tripped copy -- only
the former has the b"huggingface" key. Preserving it means copying the
column data via pyarrow.Table.set_column rather than through a DataFrame.
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
    parser.add_argument("--task", required=True, help="exact task prompt string to extract")
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

    src_episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    src_stats = {row["episode_index"]: row for row in load_jsonl(src / "meta" / "episodes_stats.jsonl")}
    chunks_size = json.loads((src / "meta" / "info.json").read_text())["chunks_size"]

    matched = [ep for ep in src_episodes if ep["tasks"][0] == args.task]
    if not matched:
        raise ValueError(f"no episodes found with task={args.task!r}")

    new_episode_rows = []
    new_stats_rows = []
    next_row_index = 0
    total_frames = 0

    for new_idx, ep_meta in enumerate(matched):
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
            pa.array([0] * n, type=pa.int64()),
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

    with (dst / "meta" / "episodes.jsonl").open("w") as f:
        for row in new_episode_rows:
            f.write(json.dumps(row) + "\n")
    with (dst / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in new_stats_rows:
            f.write(json.dumps(row) + "\n")
    with (dst / "meta" / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": args.task}) + "\n")

    info = json.loads((src / "meta" / "info.json").read_text())
    info["total_episodes"] = len(matched)
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["splits"]["train"] = f"0:{len(matched)}"
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    print(f"extracted {len(matched)} episodes, {total_frames} frames -> {dst}")


if __name__ == "__main__":
    main()
