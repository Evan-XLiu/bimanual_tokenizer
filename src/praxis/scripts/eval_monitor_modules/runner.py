"""Subprocess runner helpers for eval monitor."""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
from collections.abc import Callable
from pathlib import Path


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Send SIGTERM to process group, escalate to SIGKILL if needed."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def launch_eval(
    *,
    eval_command: list[str],
    pretrained_path: Path,
    eval_output_dir: Path,
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
    extra_eval_overrides: list[str] | None,
    build_eval_overrides_fn: Callable[..., list[str]],
    normalize_eval_overrides_fn: Callable[[list[str]], list[str]],
    logger: logging.Logger,
) -> subprocess.Popen:
    train_config_path = pretrained_path / "train_config.yaml"
    train_eval_overrides = build_eval_overrides_fn(
        train_config_path,
        num_eval_per_task=num_eval_per_task,
        num_parallel_env=num_parallel_env,
        record_episodes_per_task=record_episodes_per_task,
    )
    cmd = [
        *eval_command,
        *train_eval_overrides,
        f"policy.pretrained_path={pretrained_path}",
        f"eval.output_dir={eval_output_dir}",
    ]
    if extra_eval_overrides:
        cmd.extend(normalize_eval_overrides_fn(extra_eval_overrides))
    logger.info("Launching: %s", " ".join(str(c) for c in cmd))
    env = {**os.environ, "MUJOCO_GL": "egl", "HYDRA_FULL_ERROR": "1"}
    return subprocess.Popen(
        [str(c) for c in cmd],
        start_new_session=True,
        env=env,
    )
