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
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


def select_tasks(
    path: str | Path = DEFAULT_TASKS_PATH,
    *,
    excluded_tasks: list[str] | tuple[str, ...] = (),
) -> list[str]:
    """Load the official 50-task set and optionally remove named tasks."""
    tasks = load_official_tasks(path)
    excluded = set(excluded_tasks)
    unknown = excluded.difference(tasks)
    if unknown:
        raise ValueError(f"Unknown RoboTwin task exclusions: {sorted(unknown)}")
    selected = [task for task in tasks if task not in excluded]
    if not selected:
        raise ValueError("Task exclusions removed every RoboTwin task.")
    return selected


def assigned_tasks(
    tasks: list[str], *, worker_index: int, worker_count: int
) -> list[str]:
    if worker_count <= 0 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError(
            f"Invalid worker assignment index={worker_index}, count={worker_count}."
        )
    return tasks[worker_index::worker_count]


def _task_is_completed(output_dir: Path, task: str) -> bool:
    task_dir = output_dir / task
    manifest_path = task_dir / "run_manifest.json"
    result_path = task_dir / "comparison.json"
    if not manifest_path.is_file() or not result_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest.get("status") == "completed"


def _recorded_episode_count(output_dir: Path, task: str) -> int:
    path = output_dir / task / "episode_records.jsonl"
    if not path.is_file():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


def _task_completed_from_line(line: str) -> int | None:
    if "ROBOTWIN_EPISODE_RESULT " not in line:
        return None
    for field in line.split():
        if field.startswith("task_completed="):
            completed, _, _total = field.removeprefix("task_completed=").partition("/")
            return int(completed)
    return None


def supervise_suite(
    *,
    robotwin_root: Path,
    tasks_path: Path,
    policy_host: str,
    base_policy_port: int,
    output_dir: Path,
    run_id: str,
    expected_checkpoint: Path,
    gpu_count: int = 8,
    simulators_per_gpu: int = 3,
    server_count: int = 8,
    episodes_per_task: int = 20,
    max_seed_attempts: int = 1000,
    resume: bool = False,
    excluded_tasks: list[str] | tuple[str, ...] = (),
    task_runner: Callable[..., dict[str, Any]] = run_task_eval,
) -> dict[str, Any]:
    """Dynamically assign whole tasks to fixed simulator/policy slots."""
    if gpu_count <= 0 or simulators_per_gpu <= 0 or server_count <= 0:
        raise ValueError("GPU, simulator, and server counts must be positive.")
    tasks = select_tasks(tasks_path, excluded_tasks=excluded_tasks)
    root = output_dir.expanduser().resolve()
    slot_count = gpu_count * simulators_per_gpu
    completed = {task for task in tasks if resume and _task_is_completed(root, task)}
    pending = deque(task for task in tasks if task not in completed)
    failures: dict[str, str] = {}
    active: dict[int, str] = {}
    episode_counts = {
        task: _recorded_episode_count(root, task) if resume else 0 for task in tasks
    }
    lock = threading.Lock()
    total_episodes = len(tasks) * episodes_per_task

    def emit_locked(
        event: str,
        *,
        slot: int | None = None,
        task: str | None = None,
    ) -> None:
        assignment = ""
        if slot is not None:
            gpu = slot // simulators_per_gpu
            server_index = slot % server_count
            assignment = (
                f" slot={slot} task={task} gpu={gpu} "
                f"policy={policy_host}:{base_policy_port + server_index}"
            )
        print(
            f"ROBOTWIN_SUITE_PARALLELISM event={event} "
            f"active_slots={len(active)} total_slots={slot_count} "
            f"pending_tasks={len(pending)} completed_tasks={len(completed)} "
            f"failed_tasks={len(failures)} "
            f"completed_episodes={sum(episode_counts.values())}/{total_episodes}"
            f"{assignment}",
            flush=True,
        )

    def run_slot(slot: int) -> None:
        gpu = slot // simulators_per_gpu
        server_index = slot % server_count
        policy_port = base_policy_port + server_index
        while True:
            with lock:
                if not pending:
                    return
                task = pending.popleft()
                active[slot] = task
                emit_locked("task_start", slot=slot, task=task)

            def on_line(
                line: str,
                task_name: str = task,
                slot_index: int = slot,
            ) -> None:
                task_completed = _task_completed_from_line(line)
                if task_completed is None:
                    return
                with lock:
                    episode_counts[task_name] = task_completed
                    emit_locked("episode", slot=slot_index, task=task_name)

            try:
                result = task_runner(
                    robotwin_root=robotwin_root,
                    task_name=task,
                    task_config="demo_clean",
                    episodes_per_task=episodes_per_task,
                    record_episodes_per_task=episodes_per_task,
                    seed=0,
                    policy_host=policy_host,
                    policy_port=policy_port,
                    output_dir=root,
                    run_id=run_id,
                    expected_checkpoint=expected_checkpoint,
                    max_seed_attempts=max_seed_attempts,
                    resume=resume,
                    subprocess_env={"CUDA_VISIBLE_DEVICES": str(gpu)},
                    line_callback=on_line,
                )
            except Exception as exc:
                with lock:
                    active.pop(slot, None)
                    failures[task] = (
                        f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
                    )
                    emit_locked("task_failed", slot=slot, task=task)
                continue

            with lock:
                active.pop(slot, None)
                completed.add(task)
                episode_counts[task] = int(result.get("episodes", episodes_per_task))
                emit_locked("task_complete", slot=slot, task=task)

    with lock:
        emit_locked("resume_loaded")
    with ThreadPoolExecutor(max_workers=slot_count) as executor:
        futures = [executor.submit(run_slot, slot) for slot in range(slot_count)]
        for future in futures:
            future.result()

    remaining = [task for task in tasks if task not in completed]
    summary = {
        "run_id": run_id,
        "status": "completed" if not remaining else "incomplete",
        "task_count": len(tasks),
        "completed_tasks": len(completed),
        "failed_tasks": failures,
        "remaining_tasks": remaining,
        "completed_episodes": sum(episode_counts.values()),
        "total_episodes": total_episodes,
        "slot_count": slot_count,
        "gpu_count": gpu_count,
        "simulators_per_gpu": simulators_per_gpu,
        "server_count": server_count,
        "excluded_tasks": sorted(set(excluded_tasks)),
    }
    with lock:
        emit_locked("supervisor_complete")
    return summary


def supervisor_exit_code(summary: dict[str, Any]) -> int:
    return 0 if summary.get("status") == "completed" else 1


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
    excluded_tasks: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    tasks = assigned_tasks(
        select_tasks(tasks_path, excluded_tasks=excluded_tasks),
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
                print(
                    f"ROBOTWIN_SUITE_TASK_RESUME_SKIP worker={worker_index} task={task}",
                    flush=True,
                )
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
    excluded_tasks: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    tasks = select_tasks(tasks_path, excluded_tasks=excluded_tasks)
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
            raise RuntimeError(
                f"Incomplete or mismatched RoboTwin task result: {task}."
            )
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
        "excluded_tasks": sorted(set(excluded_tasks)),
    }
    _atomic_text(
        root / "comparison.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
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
    worker.add_argument("--exclude-task", action="append", default=[])

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--tasks", default=DEFAULT_TASKS_PATH, type=Path)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--expected-checkpoint", required=True, type=Path)
    aggregate.add_argument("--episodes-per-task", type=int, default=20)
    aggregate.add_argument("--exclude-task", action="append", default=[])

    supervise = subparsers.add_parser("supervise")
    supervise.add_argument("--robotwin-root", required=True, type=Path)
    supervise.add_argument("--tasks", default=DEFAULT_TASKS_PATH, type=Path)
    supervise.add_argument("--policy-host", default="127.0.0.1")
    supervise.add_argument("--base-policy-port", required=True, type=int)
    supervise.add_argument("--output-dir", required=True, type=Path)
    supervise.add_argument("--run-id", required=True)
    supervise.add_argument("--expected-checkpoint", required=True, type=Path)
    supervise.add_argument("--gpu-count", type=int, default=8)
    supervise.add_argument("--simulators-per-gpu", type=int, default=3)
    supervise.add_argument("--server-count", type=int, default=8)
    supervise.add_argument("--episodes-per-task", type=int, default=20)
    supervise.add_argument("--max-seed-attempts", type=int, default=1000)
    supervise.add_argument("--resume", action="store_true")
    supervise.add_argument("--exclude-task", action="append", default=[])
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
            excluded_tasks=args.exclude_task,
        )
    elif args.command == "aggregate":
        result = aggregate_suite(
            tasks_path=args.tasks,
            output_dir=args.output_dir,
            expected_checkpoint=args.expected_checkpoint,
            run_id=args.run_id,
            episodes_per_task=args.episodes_per_task,
            excluded_tasks=args.exclude_task,
        )
    else:
        result = supervise_suite(
            robotwin_root=args.robotwin_root,
            tasks_path=args.tasks,
            policy_host=args.policy_host,
            base_policy_port=args.base_policy_port,
            output_dir=args.output_dir,
            run_id=args.run_id,
            expected_checkpoint=args.expected_checkpoint,
            gpu_count=args.gpu_count,
            simulators_per_gpu=args.simulators_per_gpu,
            server_count=args.server_count,
            episodes_per_task=args.episodes_per_task,
            max_seed_attempts=args.max_seed_attempts,
            resume=args.resume,
            excluded_tasks=args.exclude_task,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "supervise":
        assert isinstance(result, dict)
        raise SystemExit(supervisor_exit_code(result))


if __name__ == "__main__":
    main()
