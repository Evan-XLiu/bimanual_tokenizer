# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Run and aggregate the official 50-task RoboTwin clean evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))

from scripts.robotwin.run_eval import run_task_eval

DEFAULT_TASKS_PATH = Path(__file__).resolve().with_name("all_tasks.yml")
EXPECTED_TASK_COUNT = 50


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_official_tasks(path: str | Path = DEFAULT_TASKS_PATH) -> list[str]:
    payload = yaml.safe_load(Path(path).expanduser().resolve().read_text())
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if (
        not isinstance(tasks, list)
        or len(tasks) != EXPECTED_TASK_COUNT
        or not all(isinstance(task, str) and task for task in tasks)
        or len(set(tasks)) != EXPECTED_TASK_COUNT
    ):
        raise ValueError(
            "RoboTwin clean task list must contain exactly 50 unique task names."
        )
    return tasks


def assigned_tasks(tasks: list[str], *, worker_index: int, worker_count: int) -> list[str]:
    if worker_count <= 0 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError(
            f"Invalid worker assignment index={worker_index}, count={worker_count}."
        )
    return tasks[worker_index::worker_count]


def run_worker(
    *,
    robotwin_root: Path,
    tasks_path: Path,
    worker_index: int,
    worker_count: int,
    policy_host: str,
    policy_port: int,
    output_dir: Path,
    run_id: str,
    expected_checkpoint: Path,
    episodes_per_task: int = 20,
    max_seed_attempts: int = 1000,
    resume: bool = False,
) -> list[dict[str, Any]]:
    tasks = assigned_tasks(
        load_official_tasks(tasks_path),
        worker_index=worker_index,
        worker_count=worker_count,
    )
    results = []
    for task in tasks:
        task_dir = output_dir.expanduser().resolve() / task
        completed_manifest = task_dir / "run_manifest.json"
        completed_result = task_dir / "comparison.json"
        if resume and completed_manifest.is_file() and completed_result.is_file():
            manifest = json.loads(completed_manifest.read_text())
            if manifest.get("status") == "completed":
                results.append(json.loads(completed_result.read_text()))
                print(f"ROBOTWIN_SUITE_TASK_RESUME_SKIP worker={worker_index} task={task}", flush=True)
                continue
        print(
            f"ROBOTWIN_SUITE_TASK_START worker={worker_index} task={task}",
            flush=True,
        )
        result = run_task_eval(
            robotwin_root=robotwin_root,
            task_name=task,
            task_config="demo_clean",
            episodes_per_task=episodes_per_task,
            record_episodes_per_task=episodes_per_task,
            seed=0,
            policy_host=policy_host,
            policy_port=policy_port,
            output_dir=output_dir,
            run_id=run_id,
            expected_checkpoint=expected_checkpoint,
            max_seed_attempts=max_seed_attempts,
            resume=resume,
        )
        results.append(result)
        print(
            f"ROBOTWIN_SUITE_TASK_COMPLETE worker={worker_index} task={task} "
            f"successes={result['successes']}/{result['episodes']}",
            flush=True,
        )
    return results


def aggregate_suite(
    *,
    tasks_path: Path,
    output_dir: Path,
    expected_checkpoint: Path,
    run_id: str,
    episodes_per_task: int = 20,
) -> dict[str, Any]:
    tasks = load_official_tasks(tasks_path)
    root = output_dir.expanduser().resolve()
    checkpoint = expected_checkpoint.expanduser().resolve()
    task_results: list[dict[str, Any]] = []
    media_manifest_paths: list[str] = []
    for task in tasks:
        task_dir = root / task
        result = json.loads((task_dir / "comparison.json").read_text())
        media = json.loads((task_dir / "media_manifest.json").read_text())
        manifest = json.loads((task_dir / "run_manifest.json").read_text())
        if (
            result.get("task") != task
            or result.get("episodes") != episodes_per_task
            or media.get("complete") is not True
            or len(media.get("episodes", [])) != episodes_per_task
            or manifest.get("status") != "completed"
            or manifest.get("task_config") != "demo_clean"
            or manifest.get("instruction_type") != "seen"
            or manifest.get("seed") != 0
            or manifest.get("record_episodes_per_task") != episodes_per_task
            or manifest.get("expected_checkpoint", {}).get("checkpoint", {}).get("path")
            != str(checkpoint)
        ):
            raise RuntimeError(f"Incomplete or mismatched RoboTwin task result: {task}.")
        task_results.append(result)
        media_manifest_paths.append(str(task_dir / "media_manifest.json"))

    total_successes = sum(int(result["successes"]) for result in task_results)
    total_episodes = len(tasks) * episodes_per_task
    macro_success = sum(float(result["success_rate"]) for result in task_results) / len(
        tasks
    )
    aggregate_success = total_successes / total_episodes
    summary = {
        "run_id": run_id,
        "status": "completed",
        "checkpoint_label": "final EMA step10095",
        "checkpoint_path": str(checkpoint),
        "task_config": "demo_clean",
        "instruction_type": "seen",
        "seed_group": 0,
        "start_seed": 100000,
        "expert_check": True,
        "max_seed_attempts_per_task": 1000,
        "record_all_episodes": True,
        "task_count": len(tasks),
        "episodes_per_task": episodes_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "macro_success_rate": macro_success,
        "aggregate_success_rate": aggregate_success,
        "tasks": task_results,
        "media_manifest_paths": media_manifest_paths,
    }
    _atomic_text(root / "comparison.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    csv_path = root / "comparison.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["task", "episodes", "successes", "success_rate"]
        )
        writer.writeheader()
        writer.writerows(
            {key: result[key] for key in writer.fieldnames} for result in task_results
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, csv_path)
    print(
        f"ROBOTWIN_SUITE_COMPLETE successes={total_successes}/{total_episodes} "
        f"macro={macro_success:.6f} aggregate={aggregate_success:.6f}",
        flush=True,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--robotwin-root", required=True, type=Path)
    worker.add_argument("--tasks", default=DEFAULT_TASKS_PATH, type=Path)
    worker.add_argument("--worker-index", required=True, type=int)
    worker.add_argument("--worker-count", required=True, type=int)
    worker.add_argument("--policy-host", default="127.0.0.1")
    worker.add_argument("--policy-port", required=True, type=int)
    worker.add_argument("--output-dir", required=True, type=Path)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--expected-checkpoint", required=True, type=Path)
    worker.add_argument("--episodes-per-task", type=int, default=20)
    worker.add_argument("--max-seed-attempts", type=int, default=1000)
    worker.add_argument("--resume", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--tasks", default=DEFAULT_TASKS_PATH, type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--expected-checkpoint", required=True, type=Path)
    aggregate.add_argument("--episodes-per-task", type=int, default=20)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "worker":
        result = run_worker(
            robotwin_root=args.robotwin_root,
            tasks_path=args.tasks,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            policy_host=args.policy_host,
            policy_port=args.policy_port,
            output_dir=args.output_dir,
            run_id=args.run_id,
            expected_checkpoint=args.expected_checkpoint,
            episodes_per_task=args.episodes_per_task,
            max_seed_attempts=args.max_seed_attempts,
            resume=args.resume,
        )
    else:
        result = aggregate_suite(
            tasks_path=args.tasks,
            output_dir=args.output_dir,
            expected_checkpoint=args.expected_checkpoint,
            run_id=args.run_id,
            episodes_per_task=args.episodes_per_task,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
