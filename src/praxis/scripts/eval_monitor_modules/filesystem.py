# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Filesystem and artifact helpers for eval monitor."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from praxis_eval.evaluation.artifacts import resolve_eval_step_dir

from praxis.core.checkpoint_artifacts import is_complete_safetensors_dir
from praxis.scripts.eval_monitor_modules.candidates import read_manifest


def staging_dir_for_step(eval_base: Path, step: int) -> Path:
    return eval_base / "staging" / f"step_{step}"


def has_results(eval_base: Path, step: int, *, results_filename: str) -> bool:
    return (resolve_eval_step_dir(eval_base, step) / results_filename).exists()


def read_results(
    eval_base: Path,
    step: int,
    *,
    results_filename: str,
) -> dict | None:
    path = resolve_eval_step_dir(eval_base, step) / results_filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def cleanup_eval_dir(
    eval_base: Path,
    step: int,
    *,
    keep_results: bool,
    results_filename: str,
    logger,
) -> None:
    d = resolve_eval_step_dir(eval_base, step)
    if not d.exists():
        return
    results_path = d / results_filename
    if keep_results and results_path.exists():
        media = d / "media"
        if media.exists():
            shutil.rmtree(media, ignore_errors=True)
            logger.info("Removed media/ for eval step %d (kept results)", step)
        return
    shutil.rmtree(d, ignore_errors=True)
    logger.info("Removed eval dir for step %d", step)


def cleanup_staging_dir(eval_base: Path, step: int, *, logger) -> None:
    d = staging_dir_for_step(eval_base, step)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        logger.info("Removed staging dir for step %d", step)


def read_evaluated_steps(eval_base: Path, *, evaluated_steps_filename: str) -> set[int]:
    path = eval_base / evaluated_steps_filename
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return set()
    if not isinstance(payload, list):
        return set()
    steps: set[int] = set()
    for value in payload:
        try:
            steps.add(int(value))
        except (TypeError, ValueError):
            continue
    return steps


def write_evaluated_steps(
    eval_base: Path,
    evaluated: set[int],
    *,
    evaluated_steps_filename: str,
) -> None:
    eval_base.mkdir(parents=True, exist_ok=True)
    out_path = eval_base / evaluated_steps_filename
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(sorted(evaluated), indent=2) + "\n")
    os.replace(tmp_path, out_path)


def read_retained_eval_steps(
    eval_base: Path,
    *,
    best_manifest_filename: str,
) -> set[int]:
    manifest = read_manifest(eval_base / "checkpoints" / best_manifest_filename)
    retained: set[int] = set()
    for entry in manifest or []:
        try:
            retained.add(int(entry["step"]))
        except (KeyError, TypeError, ValueError):
            continue
    return retained


def cleanup_unneeded_eval_dirs(
    eval_base: Path,
    *,
    retained_steps: set[int],
    active_step: int | None,
    cleanup_eval_dir_fn,
) -> None:
    for path in (eval_base / "by_step").glob("step_*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError:
            continue
        if step == active_step or step in retained_steps:
            continue
        cleanup_eval_dir_fn(eval_base, step, keep_results=True)


def write_cumulative_results(
    eval_base: Path,
    results: dict[int, dict[str, float]],
    *,
    cumulative_results_filename: str,
) -> None:
    eval_base.mkdir(parents=True, exist_ok=True)
    payload = {str(step): metrics for step, metrics in sorted(results.items())}
    out_path = eval_base / cumulative_results_filename
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_path, out_path)


def snapshot_candidate(candidate: dict[str, Any], eval_base: Path, *, logger) -> Path:
    def _looks_complete_pretrained_dir(path: Path) -> bool:
        required = (path / "train_config.yaml", path / "config.json")
        return all(p.exists() for p in required) and is_complete_safetensors_dir(path)

    step = int(candidate["step"])
    src = Path(candidate["path"]) / "pretrained_model"
    dst = staging_dir_for_step(eval_base, step) / "pretrained_model"
    if dst.exists() and _looks_complete_pretrained_dir(dst):
        return dst
    if dst.exists():
        logger.warning("Staging checkpoint for step %d is incomplete; recopying.", step)
        shutil.rmtree(dst.parent, ignore_errors=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".tmp_pretrained_model_{os.getpid()}_{time.time_ns()}"
    shutil.copytree(src, tmp, dirs_exist_ok=False)
    os.replace(tmp, dst)
    return dst
