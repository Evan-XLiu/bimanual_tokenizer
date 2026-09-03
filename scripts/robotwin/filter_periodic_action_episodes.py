#!/usr/bin/env python3
"""Build the audited RoboTwin episode exclusion list and retained-only stats.

The exclusions are explicit because a magnitude threshold cannot reliably find
2*pi-equivalent action branches.  The raw dataset is never changed.  Both codec
and policy training consume the generated exclusion list and action statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_ROOT = Path("data/lerobot/local/robotwin2_depth_merged_clean")
JOINT_NAMES = (
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
)
ARM_DIMS = tuple(range(6)) + tuple(range(7, 13))

PERIODIC_EPISODES_BY_TASK = {
    "adjust_bottle": (16,),
    "place_dual_shoes": (352, 353, 355, 359, 374, 382, 385, 392, 399),
    "place_shoe": (466, 474, 485, 487, 495),
    "rotate_qrcode": (517, 525, 538),
    "turn_switch": (1257, 1274, 1292),
    "hanging_mug": (1407,),
    "grab_roller": (1988,),
    "move_stapler_pad": (2056,),
    "place_mouse_pad": (2264, 2265, 2285, 2286, 2292, 2294, 2297),
}
TEMPORAL_JUMP_EPISODES_BY_TASK = {
    "rotate_qrcode": (510, 527, 530, 531),
}


def _flatten_episode_map(episodes_by_task: dict[str, tuple[int, ...]]) -> set[int]:
    return {
        episode
        for task_episodes in episodes_by_task.values()
        for episode in task_episodes
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/robotwin_hopper/fast_episode_filter"),
    )
    return parser.parse_args()


def _action_stats(actions: np.ndarray) -> dict[str, list[float] | list[int]]:
    quantiles = np.quantile(actions, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": actions.min(axis=0).tolist(),
        "max": actions.max(axis=0).tolist(),
        "mean": actions.mean(axis=0, dtype=np.float64).tolist(),
        "std": actions.std(axis=0, dtype=np.float64).tolist(),
        "count": [int(actions.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def main() -> None:
    args = _args()

    parquet_files = sorted((args.dataset_root / "data").glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data under {args.dataset_root / 'data'}")

    tables = [
        pq.read_table(
            path, columns=["action", "episode_index", "frame_index", "task_index"]
        )
        for path in parquet_files
    ]
    actions = np.concatenate(
        [np.asarray(table["action"].to_pylist(), dtype=np.float32) for table in tables]
    )
    episodes = np.concatenate(
        [table["episode_index"].to_numpy() for table in tables]
    ).astype(np.int64, copy=False)
    frames = np.concatenate([table["frame_index"].to_numpy() for table in tables])
    task_indices = np.concatenate(
        [table["task_index"].to_numpy() for table in tables]
    ).astype(np.int64, copy=False)
    if actions.ndim != 2 or actions.shape[1] != len(JOINT_NAMES):
        raise ValueError(f"Expected action shape (N, 14), got {actions.shape}")

    task_table = pq.read_table(args.dataset_root / "meta" / "tasks.parquet")
    task_rows = task_table.to_pylist()
    task_names = {
        int(row["task_index"]): str(row["__index_level_0__"]) for row in task_rows
    }

    periodic_episodes = _flatten_episode_map(PERIODIC_EPISODES_BY_TASK)
    jump_episodes = _flatten_episode_map(TEMPORAL_JUMP_EPISODES_BY_TASK)
    overlap = periodic_episodes.intersection(jump_episodes)
    if overlap:
        raise RuntimeError(
            f"Periodic and temporal exclusion lists must not overlap: {sorted(overlap)}"
        )
    excluded = sorted(periodic_episodes | jump_episodes)
    excluded_set = set(excluded)
    all_episode_ids = sorted(np.unique(episodes).astype(int).tolist())
    unknown_exclusions = excluded_set.difference(all_episode_ids)
    if unknown_exclusions:
        raise ValueError(
            f"Audited exclusions are outside the dataset: {sorted(unknown_exclusions)}"
        )
    retained_rows = ~np.isin(episodes, excluded)

    details = []
    for episode in excluded:
        ep_rows = episodes == episode
        order = np.argsort(frames[ep_rows])
        ep_actions = actions[ep_rows][order]
        ep_frames = frames[ep_rows][order]
        ep_tasks = task_indices[ep_rows]
        unique_tasks = np.unique(ep_tasks)
        if len(unique_tasks) != 1:
            raise ValueError(
                f"Episode {episode} must have one task_index, got {unique_tasks.tolist()}."
            )
        max_flat = int(np.argmax(np.abs(ep_actions[:, ARM_DIMS])))
        max_row, max_arm_col = np.unravel_index(
            max_flat, (len(ep_actions), len(ARM_DIMS))
        )
        max_dim = ARM_DIMS[int(max_arm_col)]
        deltas = np.abs(np.diff(ep_actions[:, ARM_DIMS], axis=0))
        delta_flat = int(np.argmax(deltas))
        delta_row, delta_arm_col = np.unravel_index(delta_flat, deltas.shape)
        delta_dim = ARM_DIMS[int(delta_arm_col)]
        source_task = next(
            task
            for episode_map in (
                PERIODIC_EPISODES_BY_TASK,
                TEMPORAL_JUMP_EPISODES_BY_TASK,
            )
            for task, task_episodes in episode_map.items()
            if episode in task_episodes
        )
        details.append(
            {
                "episode_index": episode,
                "exclusion_type": (
                    "periodic_2pi_branch"
                    if episode in periodic_episodes
                    else "temporal_jump"
                ),
                "source_task": source_task,
                "task_index": int(unique_tasks[0]),
                "instruction": task_names.get(int(unique_tasks[0]), "<unknown>"),
                "num_frames": int(ep_rows.sum()),
                "max_abs": float(abs(ep_actions[max_row, max_dim])),
                "max_abs_joint": JOINT_NAMES[max_dim],
                "max_abs_frame_index": int(ep_frames[max_row]),
                "max_adjacent_delta": float(deltas[delta_row, delta_arm_col]),
                "max_delta_joint": JOINT_NAMES[delta_dim],
                "max_delta_frame_index": int(ep_frames[delta_row + 1]),
            }
        )

    retained_actions = actions[retained_rows]
    report = {
        "criterion": {
            "description": (
                "explicitly audited 2*pi-equivalent action branches and "
                "adjacent-frame action jumps"
            ),
            "periodic_episodes_by_task": PERIODIC_EPISODES_BY_TASK,
            "temporal_jump_episodes_by_task": TEMPORAL_JUMP_EPISODES_BY_TASK,
            "checked_dimensions": [JOINT_NAMES[i] for i in ARM_DIMS],
        },
        "dataset_root": str(args.dataset_root.resolve()),
        "num_rows": int(len(actions)),
        "num_episodes": len(all_episode_ids),
        "num_excluded_episodes": len(excluded),
        "num_retained_episodes": len(all_episode_ids) - len(excluded),
        "num_excluded_rows": int((~retained_rows).sum()),
        "num_retained_rows": int(retained_rows.sum()),
        "excluded_episodes": excluded,
        "excluded_details": details,
    }
    stats = {"action": _action_stats(retained_actions)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / "excluded_episodes.json").write_text(
        json.dumps(excluded, indent=2) + "\n"
    )
    (args.output_dir / "retained_action_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )

    print(
        f"episodes: total={len(all_episode_ids)} excluded={len(excluded)} "
        f"retained={len(all_episode_ids) - len(excluded)}"
    )
    print(
        f"rows: total={len(actions)} excluded={(~retained_rows).sum()} "
        f"retained={retained_rows.sum()}"
    )
    for item in details:
        print(
            f"exclude episode={item['episode_index']} "
            f"type={item['exclusion_type']} source_task={item['source_task']} "
            f"max_abs={item['max_abs']:.6f} joint={item['max_abs_joint']} "
            f"max_delta={item['max_adjacent_delta']:.6f}"
        )
    print(f"wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
