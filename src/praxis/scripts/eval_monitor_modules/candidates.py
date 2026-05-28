"""Candidate manifest helpers for eval monitor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_manifest(path: Path) -> list[dict] | None:
    """Read a checkpoint manifest, returning ``None`` if missing or corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else None
    except (json.JSONDecodeError, OSError):
        return None


def read_candidate_manifest(
    checkpoint_dir: Path,
    *,
    best_manifest_filename: str,
    recent_manifest_filename: str,
) -> list[dict[str, Any]] | None:
    """Read best/recent manifests and return a deduped candidate list."""
    best_manifest = read_manifest(checkpoint_dir / "best" / best_manifest_filename)
    recent_manifest = read_manifest(
        checkpoint_dir / "recent" / recent_manifest_filename
    )
    if best_manifest is None and recent_manifest is None:
        return None

    best_steps: set[int] = set()
    best_candidates: list[dict[str, Any]] = []
    for entry in best_manifest or []:
        try:
            step = int(entry["step"])
            file_name = str(entry["file"])
        except (KeyError, TypeError, ValueError):
            continue
        best_steps.add(step)
        best_candidates.append(
            {"step": step, "path": checkpoint_dir / "best" / file_name, "tier": "best"}
        )

    recent_candidates: list[dict[str, Any]] = []
    for entry in recent_manifest or []:
        try:
            step = int(entry["step"])
            file_name = str(entry["file"])
        except (KeyError, TypeError, ValueError):
            continue
        if step in best_steps:
            continue
        recent_candidates.append(
            {
                "step": step,
                "path": checkpoint_dir / "recent" / file_name,
                "tier": "recent",
            }
        )

    recent_candidates.sort(key=lambda e: int(e["step"]), reverse=True)
    return best_candidates + recent_candidates
