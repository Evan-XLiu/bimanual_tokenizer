# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Verify RoboTwin videos and write auditable task-level results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import av
from av.error import FFmpegError

_VIDEO_RE = re.compile(r"^episode_(\d+)_(success|failure)\.mp4$")


def _validate_probe(
    *, path: Path, width: int, height: int, duration: float, decoded_frames: int
) -> dict[str, int | float]:
    if width <= 0 or height <= 0 or duration <= 0 or decoded_frames <= 0:
        raise ValueError(
            f"Video {path} has invalid decoded properties: {width}x{height}, "
            f"duration={duration}, frames={decoded_frames}."
        )
    return {
        "width": width,
        "height": height,
        "duration": duration,
        "decoded_frames": decoded_frames,
    }


def _probe_video_with_av(path: Path) -> dict[str, int | float]:
    probe: dict[str, int | float] | None = None
    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError(f"Video {path} has no video stream.")
            stream = container.streams.video[0]
            decoded_frames = sum(1 for _ in container.decode(stream))
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif stream.average_rate is not None:
                duration = decoded_frames / float(stream.average_rate)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                raise ValueError(f"Video {path} has no decodable duration.")
            probe = _validate_probe(
                path=path,
                width=int(stream.width),
                height=int(stream.height),
                duration=duration,
                decoded_frames=decoded_frames,
            )
    except (FFmpegError, OSError, ZeroDivisionError) as exc:
        raise ValueError(f"PyAV could not decode video {path}: {exc}") from exc
    if probe is None:
        raise ValueError(f"PyAV did not return decoded metadata for video {path}.")
    return probe


def _probe_video(path: Path) -> dict[str, int | float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return _probe_video_with_av(path)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,duration,nb_frames,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return _probe_video_with_av(path)
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(stream["duration"])
        decoded_frames = int(stream.get("nb_read_frames") or stream["nb_frames"])
        return _validate_probe(
            path=path,
            width=width,
            height=height,
            duration=duration,
            decoded_frames=decoded_frames,
        )
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError):
        return _probe_video_with_av(path)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_episode_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                episode_index = int(record["episode_index"])
                seed = record["seed"]
                task = record["task"]
                success = record["success"]
                terminal_reason = record["terminal_reason"]
                policy_steps = record["policy_steps"]
                video_width = record["video_width"]
                video_height = record["video_height"]
                video_path = record["video_path"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Malformed episode record at {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(
                    f"Episode {episode_index} seed must be an integer, got {seed!r}."
                )
            if not isinstance(task, str) or not task:
                raise ValueError(f"Episode {episode_index} task must be non-empty.")
            if not isinstance(success, bool):
                raise ValueError(f"Episode {episode_index} success must be boolean.")
            if terminal_reason not in {"success", "step_limit"}:
                raise ValueError(
                    f"Episode {episode_index} has invalid terminal_reason "
                    f"{terminal_reason!r}."
                )
            if (terminal_reason == "success") != success:
                raise ValueError(
                    f"Episode {episode_index} success and terminal_reason disagree."
                )
            if (
                isinstance(policy_steps, bool)
                or not isinstance(policy_steps, int)
                or policy_steps <= 0
            ):
                raise ValueError(
                    f"Episode {episode_index} policy_steps must be a positive integer, "
                    f"got {policy_steps!r}."
                )
            for label, value in (
                ("video_width", video_width),
                ("video_height", video_height),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"Episode {episode_index} {label} must be a positive integer, "
                        f"got {value!r}."
                    )
            if not isinstance(video_path, str) or not video_path:
                raise ValueError(
                    f"Episode {episode_index} video_path must be non-empty."
                )
            if episode_index in records:
                raise ValueError(f"Duplicate episode record for index {episode_index}.")
            records[episode_index] = record
    return records


def verify_task_eval(
    *,
    output_dir: str | Path,
    task_name: str,
    episodes_per_task: int,
    episode_records_path: str | Path,
) -> dict[str, Any]:
    """Verify one task and write comparison/media artifacts or fail closed."""
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    root = Path(output_dir).expanduser().resolve()
    task_dir = root / task_name
    records_path = Path(episode_records_path).expanduser().resolve()
    records = _read_episode_records(records_path)

    videos: dict[int, tuple[Path, bool]] = {}
    if task_dir.is_dir():
        for path in task_dir.iterdir():
            match = _VIDEO_RE.match(path.name)
            if match is None:
                continue
            index = int(match.group(1))
            if index in videos:
                raise ValueError(f"Multiple videos found for episode index {index}.")
            videos[index] = (path, match.group(2) == "success")

    expected_indices = set(range(episodes_per_task))
    errors: list[str] = []
    if set(videos) != expected_indices:
        errors.append(
            f"Expected {episodes_per_task} episode videos with indices "
            f"0..{episodes_per_task - 1}, found {sorted(videos)}."
        )
    if set(records) != expected_indices:
        errors.append(
            f"Expected {episodes_per_task} episode records with indices "
            f"0..{episodes_per_task - 1}, found {sorted(records)}."
        )

    media_episodes: list[dict[str, Any]] = []
    for index in range(episodes_per_task):
        video_info = videos.get(index)
        record = records.get(index)
        path = video_info[0] if video_info is not None else None
        if record is not None:
            if record["task"] != task_name:
                errors.append(
                    f"Episode {index} task {record['task']!r} does not match {task_name!r}."
                )
            outcome = "success" if record["success"] else "failure"
            expected_path = task_dir / f"episode_{index:04d}_{outcome}.mp4"
            if Path(record["video_path"]).resolve() != expected_path.resolve():
                errors.append(
                    f"Episode {index} recorded path {record['video_path']!r} "
                    f"does not match {expected_path}."
                )
            if video_info is not None and video_info[1] != record["success"]:
                errors.append(
                    f"Episode {index} filename outcome disagrees with record."
                )

        nonempty = bool(path is not None and path.is_file() and path.stat().st_size > 0)
        probe: dict[str, int | float] | None = None
        if video_info is not None and not nonempty:
            errors.append(f"Episode {index} video is missing or empty: {path}.")
        elif video_info is not None:
            try:
                probe = _probe_video(video_info[0])
            except ValueError as exc:
                errors.append(str(exc))
            if (
                probe is not None
                and record is not None
                and int(probe["decoded_frames"]) != int(record["policy_steps"]) + 1
            ):
                errors.append(
                    f"Episode {index} video has {probe['decoded_frames']} decoded frames, "
                    f"but policy_steps={record['policy_steps']} requires exactly "
                    f"{int(record['policy_steps']) + 1}."
                )
            if probe is not None and record is not None:
                actual_size = (int(probe["width"]), int(probe["height"]))
                expected_size = (
                    int(record["video_width"]),
                    int(record["video_height"]),
                )
                if actual_size != expected_size:
                    errors.append(
                        f"Episode {index} video size {actual_size} does not match "
                        f"configured size {expected_size}."
                    )
        media_episodes.append(
            {
                "task": task_name,
                "episode_index": index,
                "seed": record.get("seed") if record is not None else None,
                "success": record.get("success") if record is not None else None,
                "terminal_reason": (
                    record.get("terminal_reason") if record is not None else None
                ),
                "policy_steps": (
                    record.get("policy_steps") if record is not None else None
                ),
                "video_width": (
                    record.get("video_width") if record is not None else None
                ),
                "video_height": (
                    record.get("video_height") if record is not None else None
                ),
                "video_path": str(path) if path is not None else None,
                "video_nonempty": nonempty,
                "video_probe": probe,
            }
        )

    seeds = [item["seed"] for item in media_episodes if item["seed"] is not None]
    if len(seeds) != len(set(seeds)):
        errors.append(f"Episode seeds must be unique within a task, got {seeds}.")

    task_dir.mkdir(parents=True, exist_ok=True)
    media_manifest_path = task_dir / "media_manifest.json"
    media_manifest = {
        "task": task_name,
        "episodes_per_task": episodes_per_task,
        "complete": not errors,
        "errors": errors,
        "episodes": media_episodes,
    }
    _atomic_write_text(
        media_manifest_path,
        json.dumps(media_manifest, indent=2, sort_keys=True) + "\n",
    )
    if errors:
        raise ValueError("RoboTwin eval artifacts are incomplete: " + " ".join(errors))

    successes = sum(bool(item["success"]) for item in media_episodes)
    result = {
        "task": task_name,
        "episodes": episodes_per_task,
        "successes": successes,
        "success_rate": successes / episodes_per_task,
        "seeds": seeds,
        "episode_records_path": str(records_path),
        "media_manifest_path": str(media_manifest_path),
    }
    _atomic_write_text(
        task_dir / "comparison.json",
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    comparison_csv = task_dir / "comparison.csv"
    temporary_csv = comparison_csv.with_name(f".{comparison_csv.name}.tmp")
    with temporary_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task", "episodes", "successes", "success_rate"]
        )
        writer.writeheader()
        writer.writerow({key: result[key] for key in writer.fieldnames})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_csv, comparison_csv)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes-per-task", required=True, type=int)
    parser.add_argument("--episode-records", required=True, type=Path)
    args = parser.parse_args()
    result = verify_task_eval(
        output_dir=args.output_dir,
        task_name=args.task,
        episodes_per_task=args.episodes_per_task,
        episode_records_path=args.episode_records,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
