#!/usr/bin/env python3
"""Build one action-only LeRobot dataset from all four BiCAT YAM tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASKS = ("box", "bottles", "pen", "plate")
RIGHT_ARM_SIGN = np.asarray((-1, 1, 1, 1, -1, -1, 1), dtype=np.float32)


def _load_task(root: Path, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    task_root = root / task
    info = json.loads((task_root / "meta/info.json").read_text())
    files = sorted((task_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet data under {task_root / 'data'}")
    columns = ("action_joint_angles", "episode_index", "frame_index")
    tables = [pq.read_table(path, columns=list(columns)) for path in files]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    action_column = table.column("action_joint_angles").combine_chunks()
    actions = np.asarray(action_column.values.to_numpy(), dtype=np.float32).reshape(-1, 14)
    episodes = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
    frames = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
    task_table = pq.read_table(task_root / "meta/tasks.parquet")
    instruction = str(task_table.column("__index_level_0__")[0].as_py())
    if len(actions) != int(info["total_frames"]):
        raise ValueError(f"{task}: frame count differs from meta/info.json")
    if len(np.unique(episodes)) != int(info["total_episodes"]):
        raise ValueError(f"{task}: episode count differs from meta/info.json")
    if not np.isfinite(actions).all():
        raise ValueError(f"{task}: action_joint_angles contains non-finite values")
    return actions, episodes, frames, instruction, info


def prepare(source: Path, destination: Path) -> dict:
    if destination.exists():
        manifest_path = destination / "source_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"Refusing to overwrite incomplete dataset: {destination}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("source") != str(source.resolve()):
            raise ValueError("Existing action-only dataset was built from another source")
        return manifest

    staging = destination.with_name(destination.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Remove or inspect incomplete staging directory: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset.create(
        repo_id="local/bicat_yam_joint_all",
        root=staging,
        fps=30,
        robot_type="bi_yam_follower",
        use_videos=False,
        features={
            "action_joint_angles": {"dtype": "float32", "shape": (14,), "names": None},
        },
    )

    task_counts: dict[str, dict[str, int]] = {}
    shared_min = np.full(7, np.inf, dtype=np.float32)
    shared_max = np.full(7, -np.inf, dtype=np.float32)
    for task in TASKS:
        actions, episodes, frames, instruction, info = _load_task(source, task)
        canonical = np.concatenate((actions[:, :7], actions[:, 7:] * RIGHT_ARM_SIGN), axis=0)
        shared_min = np.minimum(shared_min, canonical.min(axis=0))
        shared_max = np.maximum(shared_max, canonical.max(axis=0))
        for episode in range(int(info["total_episodes"])):
            locations = np.flatnonzero(episodes == episode)
            if not locations.size or not np.array_equal(frames[locations], np.arange(len(locations))):
                raise ValueError(f"{task} episode {episode} has missing or out-of-order frames")
            for action in actions[locations]:
                dataset.add_frame({"action_joint_angles": action, "task": instruction})
            dataset.save_episode(parallel_encoding=False)
        task_counts[task] = {
            "episodes": int(info["total_episodes"]),
            "frames": len(actions),
        }
        print(f"Prepared {task}: {len(actions)} frames, {int(info['total_episodes'])} episodes", flush=True)

    dataset.finalize()
    manifest = {
        "source": str(source.resolve()),
        "tasks": task_counts,
        "total_episodes": sum(item["episodes"] for item in task_counts.values()),
        "total_frames": sum(item["frames"] for item in task_counts.values()),
        "action_key": "action_joint_angles",
        "action_dim": 14,
        "right_arm_sign": RIGHT_ARM_SIGN.tolist(),
        "shared_action_min": shared_min.tolist(),
        "shared_action_max": shared_max.tolist(),
    }
    (staging / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/BiCAT-BimanualYAM-Dataset"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/BiCAT-BimanualYAM-JointOnly"),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
