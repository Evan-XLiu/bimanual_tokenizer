# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.robotwin.verify_eval import _probe_video, verify_task_eval


def _write_eval(output_dir: Path) -> Path:
    task_dir = output_dir / "pick_apple"
    task_dir.mkdir(parents=True)
    (task_dir / "episode_0000_success.mp4").write_bytes(b"video-0")
    (task_dir / "episode_0001_failure.mp4").write_bytes(b"video-1")
    records = output_dir / "episode_records.jsonl"
    records.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task": "pick_apple",
                        "episode_index": 0,
                        "seed": 100001,
                        "success": True,
                        "terminal_reason": "success",
                        "policy_steps": 19,
                        "video_width": 640,
                        "video_height": 480,
                        "video_path": str(task_dir / "episode_0000_success.mp4"),
                    }
                ),
                json.dumps(
                    {
                        "task": "pick_apple",
                        "episode_index": 1,
                        "seed": 100005,
                        "success": False,
                        "terminal_reason": "step_limit",
                        "policy_steps": 19,
                        "video_width": 640,
                        "video_height": 480,
                        "video_path": str(task_dir / "episode_0001_failure.mp4"),
                    }
                ),
            ]
        )
        + "\n"
    )
    return records


def test_probe_video_falls_back_to_pyav_without_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    expected = {
        "width": 320,
        "height": 240,
        "duration": 1.0,
        "decoded_frames": 10,
    }
    monkeypatch.setattr("scripts.robotwin.verify_eval.shutil.which", lambda _: None)
    monkeypatch.setattr(
        "scripts.robotwin.verify_eval._probe_video_with_av", lambda _: expected
    )

    assert _probe_video(video) == expected


@pytest.mark.parametrize(
    "probe_result",
    [
        subprocess.CompletedProcess([], 1, "", "decode failed"),
        subprocess.CompletedProcess([], 0, '{"streams": [{}]}', ""),
    ],
)
def test_probe_video_falls_back_to_pyav_for_failed_or_incomplete_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    probe_result: subprocess.CompletedProcess[str],
) -> None:
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"video")
    expected = {
        "width": 320,
        "height": 240,
        "duration": 1.0,
        "decoded_frames": 10,
    }
    monkeypatch.setattr(
        "scripts.robotwin.verify_eval.shutil.which", lambda _: "/usr/bin/ffprobe"
    )
    monkeypatch.setattr(
        "scripts.robotwin.verify_eval.subprocess.run", lambda *_args, **_kwargs: probe_result
    )
    monkeypatch.setattr(
        "scripts.robotwin.verify_eval._probe_video_with_av", lambda _: expected
    )

    assert _probe_video(video) == expected


def test_verify_eval_writes_machine_readable_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _write_eval(tmp_path)
    monkeypatch.setattr(
        "scripts.robotwin.verify_eval._probe_video",
        lambda _path: {
            "width": 640,
            "height": 480,
            "duration": 2.0,
            "decoded_frames": 20,
        },
    )

    result = verify_task_eval(
        output_dir=tmp_path,
        task_name="pick_apple",
        episodes_per_task=2,
        episode_records_path=records,
    )

    assert result["successes"] == 1
    assert result["success_rate"] == 0.5
    assert result["seeds"] == [100001, 100005]
    manifest = json.loads((tmp_path / "pick_apple" / "media_manifest.json").read_text())
    assert len(manifest["episodes"]) == 2
    assert all(
        item["video_probe"]["decoded_frames"] > 0 for item in manifest["episodes"]
    )
    assert [item["policy_steps"] for item in manifest["episodes"]] == [19, 19]
    assert (tmp_path / "pick_apple" / "comparison.json").exists()
    assert (tmp_path / "pick_apple" / "comparison.csv").exists()


def test_verify_eval_rejects_missing_episode_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _write_eval(tmp_path)
    (tmp_path / "pick_apple" / "episode_0001_failure.mp4").unlink()
    monkeypatch.setattr(
        "scripts.robotwin.verify_eval._probe_video",
        lambda _path: {
            "width": 640,
            "height": 480,
            "duration": 2.0,
            "decoded_frames": 20,
        },
    )

    with pytest.raises(ValueError, match="Expected 2 episode videos"):
        verify_task_eval(
            output_dir=tmp_path,
            task_name="pick_apple",
            episodes_per_task=2,
            episode_records_path=records,
        )


def test_verify_eval_rejects_video_shorter_than_policy_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _write_eval(tmp_path)
    monkeypatch.setattr(
        "scripts.robotwin.verify_eval._probe_video",
        lambda _path: {
            "width": 640,
            "height": 480,
            "duration": 0.1,
            "decoded_frames": 1,
        },
    )

    with pytest.raises(ValueError, match="policy_steps=19 requires exactly 20"):
        verify_task_eval(
            output_dir=tmp_path,
            task_name="pick_apple",
            episodes_per_task=2,
            episode_records_path=records,
        )
