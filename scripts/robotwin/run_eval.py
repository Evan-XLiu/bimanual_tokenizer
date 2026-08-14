# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Run and verify one official RoboTwin task against a Praxis policy server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.robotwin.verify_eval import verify_task_eval

POLICY_NAME = "praxis_ar"
DEFAULT_PLUGIN_SOURCE = Path(__file__).resolve().parent / "policy" / POLICY_NAME
DEFAULT_EVALUATOR_OVERLAY = Path(__file__).resolve().parent / "stable2_evaluator.py"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RoboTwin run manifest must be a JSON object: {path}.")
    return payload


def _terminate_process(process: subprocess.Popen[str], *, timeout: float = 10.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def build_eval_command(
    *,
    robotwin_root: Path,
    task_name: str,
    task_config: str,
    episodes_per_task: int,
    record_episodes_per_task: int,
    seed: int,
    policy_host: str,
    policy_port: int,
    output_dir: Path,
    run_id: str,
    expected_checkpoint: Path,
    plugin_config: Path,
    max_seed_attempts: int | None = None,
    python_executable: str = sys.executable,
    evaluator_overlay: Path = DEFAULT_EVALUATOR_OVERLAY,
    resume: bool = False,
) -> list[str]:
    """Build the official evaluator command with a full-recording assertion."""
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")
    if record_episodes_per_task != episodes_per_task:
        raise ValueError(
            "record_episodes_per_task must equal episodes_per_task for RoboTwin eval: "
            f"{record_episodes_per_task} != {episodes_per_task}."
        )
    seed_attempt_limit = (
        episodes_per_task * 50 if max_seed_attempts is None else max_seed_attempts
    )
    if seed_attempt_limit < episodes_per_task:
        raise ValueError(
            "max_seed_attempts must be at least episodes_per_task: "
            f"{seed_attempt_limit} < {episodes_per_task}."
        )
    command = [
        python_executable,
        "-u",
        str(evaluator_overlay.expanduser().resolve()),
        "--robotwin-root",
        str(robotwin_root.expanduser().resolve()),
        "--plugin-config",
        str(plugin_config.expanduser().resolve()),
        "--task",
        task_name,
        "--task-config",
        task_config,
        "--episodes-per-task",
        str(episodes_per_task),
        "--record-episodes-per-task",
        str(record_episodes_per_task),
        "--seed",
        str(seed),
        "--policy-host",
        policy_host,
        "--policy-port",
        str(policy_port),
        "--output-dir",
        str(output_dir.expanduser().resolve()),
        "--run-id",
        run_id,
        "--max-seed-attempts",
        str(seed_attempt_limit),
        "--expected-checkpoint",
        str(expected_checkpoint.expanduser().resolve()),
    ]
    if resume:
        command.append("--resume")
    return command


def run_task_eval(
    *,
    robotwin_root: str | Path,
    task_name: str,
    task_config: str,
    episodes_per_task: int,
    record_episodes_per_task: int,
    seed: int,
    policy_host: str,
    policy_port: int,
    output_dir: str | Path,
    run_id: str,
    expected_checkpoint: str | Path,
    max_seed_attempts: int | None = None,
    plugin_source: str | Path = DEFAULT_PLUGIN_SOURCE,
    evaluator_overlay: str | Path = DEFAULT_EVALUATOR_OVERLAY,
    python_executable: str = sys.executable,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(robotwin_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    task_output = output / task_name
    if task_output.exists() and not resume:
        raise FileExistsError(f"Use --resume for an existing task eval output directory: {task_output}")
    task_output.mkdir(parents=True, exist_ok=resume)
    run_manifest_path = task_output / "run_manifest.json"
    _atomic_json(
        run_manifest_path,
        {
            "status": "preflight",
            "failure": None,
            "task": task_name,
            "task_config": task_config,
            "run_id": run_id,
            "robotwin_root": str(root),
            "expected_checkpoint_path": str(
                Path(expected_checkpoint).expanduser().resolve()
            ),
        },
    )

    source = Path(plugin_source).expanduser().resolve()
    plugin_config = source / "deploy_policy.yml"
    process: subprocess.Popen[str] | None = None
    stage = "preflight"
    try:
        official_entrypoint = root / "script" / "eval_policy.py"
        if not official_entrypoint.is_file():
            raise FileNotFoundError(
                f"Official RoboTwin evaluator not found: {official_entrypoint}"
            )
        required_plugin_files = (
            source / "__init__.py",
            source / "deploy_policy.py",
            plugin_config,
        )
        if not all(path.is_file() for path in required_plugin_files):
            raise FileNotFoundError(
                "Incomplete Praxis RoboTwin policy plugin; expected __init__.py, "
                f"deploy_policy.py, and deploy_policy.yml under {source}."
            )
        command = build_eval_command(
            robotwin_root=root,
            task_name=task_name,
            task_config=task_config,
            episodes_per_task=episodes_per_task,
            record_episodes_per_task=record_episodes_per_task,
            seed=seed,
            policy_host=policy_host,
            policy_port=policy_port,
            output_dir=output,
            run_id=run_id,
            expected_checkpoint=Path(expected_checkpoint),
            plugin_config=plugin_config,
            max_seed_attempts=max_seed_attempts,
            python_executable=python_executable,
            evaluator_overlay=Path(evaluator_overlay),
            # The parent always creates the task directory and preflight manifest
            # before handing the run to the evaluator process.
            resume=True,
        )
        log_dir = output / "_attempt_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{task_name}.log"
        stage = "rollout"
        with log_path.open("a" if resume else "x", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_handle.write(line)
                log_handle.flush()
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Controlled RoboTwin evaluator failed with code {return_code}; "
                f"see {log_path}."
            )
        child_manifest = _load_manifest(run_manifest_path)
        if child_manifest.get("status") != "rollout_completed":
            raise RuntimeError(
                "Controlled RoboTwin evaluator exited successfully without a "
                f"rollout_completed manifest: {run_manifest_path}."
            )
        episode_records = task_output / "episode_records.jsonl"
        stage = "verification"
        verification = verify_task_eval(
            output_dir=output,
            task_name=task_name,
            episodes_per_task=episodes_per_task,
            episode_records_path=episode_records,
        )
        child_manifest.update(
            {
                "status": "completed",
                "failure": None,
                "verification": verification,
            }
        )
        _atomic_json(run_manifest_path, child_manifest)
        return verification
    except BaseException as exc:
        manifest = _load_manifest(run_manifest_path)
        parent_failure = {
            "stage": stage,
            "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
        }
        manifest["status"] = "failed"
        manifest["parent_failure"] = parent_failure
        if manifest.get("failure") is None:
            manifest["failure"] = parent_failure
        _atomic_json(run_manifest_path, manifest)
        raise
    finally:
        if process is not None:
            _terminate_process(process)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--episodes-per-task", required=True, type=int)
    parser.add_argument("--record-episodes-per-task", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=50051)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Filesystem-safe identifier for this policy checkpoint/eval run.",
    )
    parser.add_argument(
        "--expected-checkpoint",
        required=True,
        type=Path,
        help="Local/shared pretrained_model directory that the server must identify.",
    )
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        default=None,
        help="Expert-precheck seed attempt limit (default: 50x episode count).",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_task_eval(
        robotwin_root=args.robotwin_root,
        task_name=args.task,
        task_config=args.task_config,
        episodes_per_task=args.episodes_per_task,
        record_episodes_per_task=args.record_episodes_per_task,
        seed=args.seed,
        policy_host=args.policy_host,
        policy_port=args.policy_port,
        output_dir=args.output_dir,
        run_id=args.run_id,
        expected_checkpoint=args.expected_checkpoint,
        max_seed_attempts=args.max_seed_attempts,
        resume=args.resume,
    )
    print(result)


if __name__ == "__main__":
    main()
