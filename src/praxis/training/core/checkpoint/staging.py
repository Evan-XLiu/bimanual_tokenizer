# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Atomic checkpoint staging helpers."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def remove_path(path: Path) -> None:
    """Remove a file or directory if it exists."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def publish_staged_directory(
    *,
    staging_dir: Path,
    checkpoint_dir: Path,
    backup_dir: Path,
) -> None:
    """Atomically replace ``checkpoint_dir`` with ``staging_dir``.

    If publishing fails after moving the old checkpoint aside, restore the old
    directory before re-raising.
    """
    replaced_old = False
    try:
        if checkpoint_dir.exists():
            checkpoint_dir.rename(backup_dir)
            replaced_old = True
        staging_dir.rename(checkpoint_dir)
    except Exception:
        remove_path(checkpoint_dir)
        if replaced_old and backup_dir.exists():
            backup_dir.rename(checkpoint_dir)
        raise
    else:
        remove_path(backup_dir)
    finally:
        remove_path(staging_dir)


def read_checkpoint_wandb_run_info(
    checkpoint_dir: str | Path,
) -> dict[str, str | None] | None:
    """Return `{id, project, entity}` stamped in a checkpoint, or None."""
    info_path = Path(checkpoint_dir) / "training_state" / "wandb_run_info.json"
    if not info_path.exists():
        return None
    try:
        data = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to read wandb run info from %s (%s); starting a new run.",
            info_path,
            exc,
        )
        return None
    run_id = data.get("id")
    if not run_id:
        return None
    return {
        "id": run_id,
        "project": data.get("project"),
        "entity": data.get("entity"),
    }


def stamp_wandb_run_info(trainer: object, staging_dir: Path) -> None:
    """Persist the active wandb run identity inside ``training_state/``."""
    trainer_logger = getattr(trainer, "logger", None)
    run = getattr(trainer_logger, "_wandb_run", None) if trainer_logger else None
    if run is None:
        return
    info = {
        "id": getattr(run, "id", None),
        "project": getattr(run, "project", None),
        "entity": getattr(run, "entity", None),
    }
    if not info["id"]:
        return
    state_dir = staging_dir / "training_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "wandb_run_info.json").write_text(json.dumps(info) + "\n")
