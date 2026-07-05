# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Metric-driven checkpoint retention managers."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _TopkEntry:
    step: int
    value: float
    path: Path


@dataclass
class _RecentEntry:
    step: int
    path: Path


class TopKCheckpointManager:
    """Keep the best *k* checkpoints ranked by a single metric.

    Parameters
    ----------
    k : int
        Number of top checkpoints to retain.
    mode : str
        ``"min"`` (lower is better) or ``"max"`` (higher is better).
    metric_key : str
        Key to look up in the metrics dict passed to :meth:`consider`.
    directory : str | Path
        Directory where top-K checkpoint files and the manifest are written.
    """

    def __init__(
        self,
        k: int,
        mode: str,
        metric_key: str,
        directory: str | Path,
        *,
        manage_files: bool = True,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

        self.k = k
        self.mode = mode
        self.metric_key = metric_key
        self.directory = Path(directory)
        self.manage_files = manage_files
        self._entries: list[_TopkEntry] = []
        self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consider(
        self,
        step: int,
        metrics: dict[str, float],
        save_fn: Callable[[Path], None],
    ) -> bool:
        """Evaluate whether *step* qualifies for the top-K leaderboard.

        If it does, ``save_fn(path)`` is called to persist the checkpoint,
        and the displaced worst checkpoint (if any) is deleted.

        Returns ``True`` when a checkpoint was saved, ``False`` otherwise.
        """
        value = metrics.get(self.metric_key)
        if value is None:
            return False
        if not isinstance(value, (int, float)):
            return False
        if math.isnan(value) or math.isinf(value):
            return False

        if not self._qualifies(value):
            return False

        path = self.directory / f"step_{step}"
        save_fn(path)

        self._entries = [e for e in self._entries if e.step != step]
        self._entries.append(_TopkEntry(step=step, value=value, path=path))
        self._sort_entries()

        # Evict worst if over capacity
        if len(self._entries) > self.k:
            evicted = self._entries.pop()  # worst is last after sort
            if self.manage_files:
                try:
                    if evicted.path.is_dir():
                        shutil.rmtree(evicted.path, ignore_errors=True)
                    else:
                        evicted.path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Failed to delete displaced checkpoint %s", evicted.path
                    )

        if self.manage_files:
            self._write_manifest()
        return True

    def candidate_path_if_qualified(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> Path | None:
        """Return the checkpoint path if this candidate should be saved."""
        value = metrics.get(self.metric_key)
        if value is None:
            return None
        if not isinstance(value, (int, float)):
            return None
        if math.isnan(value) or math.isinf(value):
            return None
        if not self._qualifies(value):
            return None
        return self.directory / f"step_{step}"

    def record_saved_checkpoint(self, step: int, value: float) -> None:
        """Record a checkpoint after all distributed ranks have saved it."""
        path = self.directory / f"step_{step}"
        self._entries = [e for e in self._entries if e.step != step]
        self._entries.append(_TopkEntry(step=step, value=value, path=path))
        self._sort_entries()

        if len(self._entries) > self.k:
            evicted = self._entries.pop()
            if self.manage_files:
                try:
                    if evicted.path.is_dir():
                        shutil.rmtree(evicted.path, ignore_errors=True)
                    else:
                        evicted.path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Failed to delete displaced checkpoint %s", evicted.path
                    )

        if self.manage_files:
            self._write_manifest()

    @property
    def best(self) -> _TopkEntry | None:
        """Return the current best entry, or ``None``."""
        return self._entries[0] if self._entries else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _qualifies(self, value: float) -> bool:
        """Return True if *value* should enter the leaderboard."""
        if len(self._entries) < self.k:
            return True
        worst = self._entries[-1].value
        if self.mode == "min":
            return value < worst
        return value > worst

    def _sort_entries(self) -> None:
        """Sort entries best-first.  Tie-break: earlier step wins."""
        reverse = self.mode == "max"
        self._entries.sort(key=lambda e: (e.value if not reverse else -e.value, e.step))

    def _load_manifest(self) -> None:
        """Restore top-K entries from a previously written manifest."""
        manifest_path = self.directory / "topk_manifest.json"
        if not manifest_path.exists():
            return
        skipped_incompatible = 0
        try:
            entries = json.loads(manifest_path.read_text())
            if not isinstance(entries, list):
                raise TypeError("top-K manifest must contain a list of entries")
            for e in entries:
                if not isinstance(e, dict):
                    raise TypeError("top-K manifest entries must be objects")
                if e.get("metric_key") != self.metric_key or e.get("mode") != self.mode:
                    skipped_incompatible += 1
                    continue
                path = self.directory / e["file"]
                if path.exists():
                    self._entries.append(
                        _TopkEntry(
                            step=int(e["step"]),
                            value=float(e["value"]),
                            path=path,
                        )
                    )
            self._sort_entries()
        except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
            logger.warning(
                "Failed to load top-K manifest from %s; starting fresh.", manifest_path
            )
            self._entries.clear()
            return
        if skipped_incompatible:
            logger.warning(
                "Discarded %d top-K manifest entries from %s because they do not "
                "match metric_key=%r and mode=%r.",
                skipped_incompatible,
                manifest_path,
                self.metric_key,
                self.mode,
            )

    def _write_manifest(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "step": e.step,
                "metric_key": self.metric_key,
                "mode": self.mode,
                "value": e.value,
                "file": e.path.name,
            }
            for e in self._entries
        ]
        manifest_path = self.directory / "topk_manifest.json"
        tmp_path = manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp_path, manifest_path)


class RecentCheckpointManager:
    """Keep the most recent *k* checkpoints by step number."""

    def __init__(
        self, k: int, directory: str | Path, *, manage_files: bool = True
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.directory = Path(directory)
        self.manage_files = manage_files
        self._entries: list[_RecentEntry] = []
        self._load_manifest()

    def consider(self, step: int, save_fn: Callable[[Path], None]) -> bool:
        path = self.directory / f"step_{step}"
        save_fn(path)

        self.record_saved_checkpoint(step)
        return True

    def checkpoint_path(self, step: int) -> Path:
        return self.directory / f"step_{step}"

    def record_saved_checkpoint(self, step: int) -> None:
        path = self.checkpoint_path(step)
        self._entries = [e for e in self._entries if e.step != step]
        self._entries.append(_RecentEntry(step=step, path=path))
        self._sort_entries()

        if len(self._entries) > self.k:
            evicted = self._entries.pop()
            if self.manage_files:
                try:
                    if evicted.path.is_dir():
                        shutil.rmtree(evicted.path, ignore_errors=True)
                    else:
                        evicted.path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Failed to delete displaced recent checkpoint %s",
                        evicted.path,
                    )

        if self.manage_files:
            self._write_manifest()

    def _sort_entries(self) -> None:
        self._entries.sort(key=lambda e: e.step, reverse=True)

    def _load_manifest(self) -> None:
        manifest_path = self.directory / "recent_manifest.json"
        if not manifest_path.exists():
            return
        try:
            entries = json.loads(manifest_path.read_text())
            for e in entries:
                path = self.directory / e["file"]
                if path.exists():
                    self._entries.append(_RecentEntry(step=e["step"], path=path))
            self._sort_entries()
        except (json.JSONDecodeError, KeyError, OSError):
            logger.warning(
                "Failed to load recent manifest from %s; starting fresh.", manifest_path
            )
            self._entries.clear()

    def _write_manifest(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = [{"step": e.step, "file": e.path.name} for e in self._entries]
        manifest_path = self.directory / "recent_manifest.json"
        tmp_path = manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp_path, manifest_path)
