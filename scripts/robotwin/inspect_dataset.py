# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Validate and summarize a local RoboTwin LeRobot dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def configure_local_hf_cache(
    dataset_root: Path, cache_dir: str | Path | None = None
) -> Path:
    cache = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else dataset_root.parent / ".huggingface" / "datasets"
    )
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return cache


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_num_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _feature_dim(features: dict[str, Any], key: str) -> int:
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"Dataset metadata is missing required feature {key!r}.")
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(
            f"Feature {key!r} must have one-dimensional shape, got {shape!r}."
        )
    return int(shape[0])


def _camera_shape(features: dict[str, Any], key: str) -> list[int]:
    feature = features.get(key)
    if not isinstance(feature, dict):
        raise ValueError(f"Dataset metadata is missing required camera {key!r}.")
    shape = feature.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or (shape[0] != 3 and shape[-1] != 3)
    ):
        raise ValueError(
            f"Camera {key!r} must have a positive rank-3 RGB shape in CHW or HWC "
            f"layout, got {shape!r}."
        )
    return shape


def _resolve_payload_paths(
    *,
    dataset_root: Path,
    repo_id: str,
    total_episodes: int,
    camera_keys: list[str],
) -> tuple[set[Path], dict[str, set[Path]]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    if int(metadata.total_episodes) != total_episodes:
        raise ValueError(
            "LeRobot metadata episode count disagrees with info.json: "
            f"{metadata.total_episodes} != {total_episodes}."
        )

    data_paths: set[Path] = set()
    video_paths = {key: set() for key in camera_keys}
    for episode_index in range(total_episodes):
        episode = metadata.episodes[episode_index]
        stored_index = episode.get("episode_index")
        if stored_index is not None and int(stored_index) != episode_index:
            raise ValueError(
                "Episode metadata is not contiguous: "
                f"row {episode_index} identifies episode {stored_index}."
            )
        data_paths.add((dataset_root / metadata.get_data_file_path(episode_index)).resolve())
        for key in camera_keys:
            video_paths[key].add(
                (dataset_root / metadata.get_video_file_path(episode_index, key)).resolve()
            )
    return data_paths, video_paths


def inspect_dataset(
    root: str | Path,
    *,
    hf_datasets_cache: str | Path | None = None,
    expected_action_dim: int = 14,
    expected_state_dim: int = 14,
    expected_camera_keys: tuple[str, ...] = EXPECTED_CAMERA_KEYS,
) -> dict[str, Any]:
    """Return a census and fail if the dataset violates the RoboTwin contract."""
    dataset_root = Path(root).expanduser().resolve()
    resolved_hf_cache = configure_local_hf_cache(dataset_root, hf_datasets_cache)
    meta_root = dataset_root / "meta"
    required = {
        "info.json": meta_root / "info.json",
        "stats.json": meta_root / "stats.json",
        "tasks.parquet": meta_root / "tasks.parquet",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    episode_files = sorted((meta_root / "episodes").rglob("*.parquet"))
    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not episode_files:
        missing.append(str(meta_root / "episodes" / "*.parquet"))
    if not data_files:
        missing.append(str(dataset_root / "data" / "*.parquet"))
    if missing:
        raise FileNotFoundError("Incomplete LeRobot metadata: " + ", ".join(missing))

    try:
        info = json.loads(required["info.json"].read_text())
        stats = json.loads(required["stats.json"].read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid dataset JSON metadata: {exc}") from exc
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("meta/info.json must contain a features mapping.")
    for template_key in ("data_path", "video_path"):
        if not isinstance(info.get(template_key), str) or not info[template_key]:
            raise ValueError(f"meta/info.json must contain non-empty {template_key!r}.")
    repo_id = info.get("repo_id", "local/robotwin2_depth_merged_clean")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("meta/info.json repo_id must be a non-empty string when set.")

    video_keys = sorted(
        key
        for key, value in features.items()
        if isinstance(value, dict)
        and value.get("dtype") in {"image", "video"}
    )
    camera_keys = [key for key in video_keys if key.startswith("observation.images.")]
    missing_cameras = sorted(set(expected_camera_keys) - set(camera_keys))
    if missing_cameras:
        raise ValueError(
            f"Dataset is missing required RoboTwin camera features: {missing_cameras}. "
            f"Available camera features: {camera_keys}."
        )
    camera_shapes = {
        key: _camera_shape(features, key) for key in expected_camera_keys
    }

    action_dim = _feature_dim(features, "action")
    state_dim = _feature_dim(features, "observation.state")
    if action_dim != expected_action_dim:
        raise ValueError(
            f"RoboTwin action dimension must be {expected_action_dim}, got {action_dim}."
        )
    if state_dim != expected_state_dim:
        raise ValueError(
            f"RoboTwin state dimension must be {expected_state_dim}, got {state_dim}."
        )
    for key, expected_dim in (
        ("action", expected_action_dim),
        ("observation.state", expected_state_dim),
    ):
        feature_stats = stats.get(key) if isinstance(stats, dict) else None
        if not isinstance(feature_stats, dict):
            raise ValueError(f"stats.json is missing statistics for {key!r}.")
        for statistic in ("min", "max", "mean", "std"):
            values = feature_stats.get(statistic)
            if not isinstance(values, list) or len(values) != expected_dim:
                raise ValueError(
                    f"stats.json {key}.{statistic} must contain {expected_dim} values."
                )

    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    fps = float(info.get("fps", -1))
    if total_episodes <= 0 or total_frames <= 0 or fps <= 0:
        raise ValueError(
            "Dataset totals must be positive: "
            f"episodes={total_episodes}, frames={total_frames}, fps={fps}."
        )

    episode_metadata_rows = sum(_parquet_num_rows(path) for path in episode_files)
    if episode_metadata_rows != total_episodes:
        raise ValueError(
            "Episode metadata row count does not match info.json: "
            f"{episode_metadata_rows} != {total_episodes}."
        )

    data_rows = sum(_parquet_num_rows(path) for path in data_files)
    if data_rows != total_frames:
        raise ValueError(
            "Data parquet row count does not match info.json: "
            f"{data_rows} != {total_frames}."
        )

    task_rows = _parquet_num_rows(required["tasks.parquet"])
    total_tasks = info.get("total_tasks")
    if total_tasks is not None and task_rows != int(total_tasks):
        raise ValueError(
            "Task metadata row count does not match info.json: "
            f"{task_rows} != {int(total_tasks)}."
        )

    expected_data_paths, expected_video_paths = _resolve_payload_paths(
        dataset_root=dataset_root,
        repo_id=repo_id,
        total_episodes=total_episodes,
        camera_keys=video_keys,
    )
    actual_data_paths = {path.resolve() for path in data_files}
    if actual_data_paths != expected_data_paths:
        missing_data = sorted(str(path) for path in expected_data_paths - actual_data_paths)
        extra_data = sorted(str(path) for path in actual_data_paths - expected_data_paths)
        raise FileNotFoundError(
            f"Data payload identity mismatch; missing={missing_data}, extra={extra_data}."
        )

    actual_video_paths = {
        path.resolve() for path in (dataset_root / "videos").rglob("*.mp4")
    }
    all_expected_videos = set().union(*expected_video_paths.values())
    if actual_video_paths != all_expected_videos:
        missing_videos = sorted(
            str(path) for path in all_expected_videos - actual_video_paths
        )
        extra_videos = sorted(str(path) for path in actual_video_paths - all_expected_videos)
        raise FileNotFoundError(
            "Video payload identity mismatch; "
            f"missing={missing_videos}, extra={extra_videos}."
        )
    video_counts = {
        key: len(paths) for key, paths in sorted(expected_video_paths.items())
    }

    report: dict[str, Any] = {
        "root": str(dataset_root),
        "hf_datasets_cache": str(resolved_hf_cache),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": int(fps) if fps.is_integer() else fps,
        "task_rows": task_rows,
        "episode_metadata_rows": episode_metadata_rows,
        "data_rows": data_rows,
        "camera_keys": camera_keys,
        "camera_shapes": camera_shapes,
        "video_keys": video_keys,
        "video_file_counts": video_counts,
        "unique_data_file_count": len(expected_data_paths),
        "action_dim": action_dim,
        "state_dim": state_dim,
        "metadata_sha256": {name: _sha256(path) for name, path in required.items()},
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--hf-datasets-cache", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = inspect_dataset(args.root, hf_datasets_cache=args.hf_datasets_cache)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
