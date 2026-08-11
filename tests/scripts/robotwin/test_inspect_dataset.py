# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.robotwin.inspect_dataset import inspect_dataset


def _write_metadata(root: Path, *, action_dim: int = 14) -> None:
    meta = root / "meta"
    (meta / "episodes" / "chunk-000").mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 3,
                "total_frames": 120,
                "total_tasks": 3,
                "fps": 10,
                "data_path": "data/chunk-{episode_chunk:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "observation.images.cam_high": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
                    "observation.images.cam_left_wrist": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
                    "observation.images.cam_right_wrist": {
                        "dtype": "video",
                        "shape": [480, 640, 3],
                    },
                    "observation.depth.cam_high": {
                        "dtype": "video",
                        "shape": [480, 640, 1],
                    },
                    "observation.state": {"dtype": "float32", "shape": [14]},
                    "action": {"dtype": "float32", "shape": [action_dim]},
                },
            }
        )
    )
    stats = {
        key: {name: [0.0] * 14 for name in ("min", "max", "mean", "std")}
        for key in ("action", "observation.state")
    }
    (meta / "stats.json").write_text(json.dumps(stats))
    (meta / "tasks.parquet").write_bytes(b"tasks")
    (meta / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"episodes")
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    data_path.write_bytes(b"frames")
    for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        for episode_index in range(3):
            path = (
                root
                / "videos"
                / "chunk-000"
                / f"observation.images.{camera}"
                / f"episode_{episode_index:06d}.mp4"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"video")
    for episode_index in range(3):
        path = (
            root
            / "videos/chunk-000/observation.depth.cam_high"
            / f"episode_{episode_index:06d}.mp4"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"depth-video")


def test_inspect_dataset_reports_identity_and_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_metadata(tmp_path)
    monkeypatch.setattr(
        "scripts.robotwin.inspect_dataset._parquet_num_rows",
        lambda path: 120 if "data" in path.parts else 3,
    )
    monkeypatch.setattr(
        "scripts.robotwin.inspect_dataset._resolve_payload_paths",
        lambda **_kwargs: (
            {(tmp_path / "data/chunk-000/file-000.parquet").resolve()},
            {
                key: {
                    (
                        tmp_path
                        / "videos/chunk-000"
                        / key
                        / f"episode_{episode_index:06d}.mp4"
                    ).resolve()
                    for episode_index in range(3)
                }
                for key in (
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                    "observation.depth.cam_high",
                )
            },
        ),
    )

    report = inspect_dataset(tmp_path)

    assert report["total_episodes"] == 3
    assert report["total_frames"] == 120
    assert report["fps"] == 10
    assert report["task_rows"] == 3
    assert report["episode_metadata_rows"] == 3
    assert report["data_rows"] == 120
    assert report["action_dim"] == 14
    assert report["state_dim"] == 14
    assert report["camera_keys"] == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    assert report["video_keys"] == [
        "observation.depth.cam_high",
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    assert report["metadata_sha256"]["info.json"]


def test_inspect_dataset_rejects_wrong_action_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_metadata(tmp_path, action_dim=7)
    monkeypatch.setattr(
        "scripts.robotwin.inspect_dataset._parquet_num_rows",
        lambda path: 120 if "data" in path.parts else 3,
    )

    with pytest.raises(ValueError, match="action.*14"):
        inspect_dataset(tmp_path)


def test_inspect_dataset_rejects_non_rgb_camera_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_metadata(tmp_path)
    info_path = tmp_path / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.images.cam_high"]["shape"] = [480, 640, 1]
    info_path.write_text(json.dumps(info))
    monkeypatch.setattr(
        "scripts.robotwin.inspect_dataset._parquet_num_rows",
        lambda path: 120 if "data" in path.parts else 3,
    )

    with pytest.raises(ValueError, match="rank-3 RGB shape"):
        inspect_dataset(tmp_path)
