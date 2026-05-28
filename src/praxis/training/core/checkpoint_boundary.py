"""Checkpoint-boundary completion tracking for resumable train loops."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from praxis.training.core.rank_errors import (
    raise_if_rank0_error,
    rank0_error_message,
)


class CheckpointBoundary:
    """Tracks whether checkpoint-step post-save work completed."""

    _BOUNDARY_STATE_DIRNAME = "training_state"
    _MARKER_SUFFIX = "checkpoint_boundary.json"
    _PENDING_STEP_LOG_SUFFIX = "checkpoint_boundary_step_log.json"

    def __init__(self, trainer: Any, *, checkpoint_every: int | None) -> None:
        self.trainer = trainer
        self.checkpoint_every = checkpoint_every

    @staticmethod
    def _marker_path(checkpoint_dir: Path) -> Path:
        return (
            CheckpointBoundary._state_dir(checkpoint_dir)
            / f"{checkpoint_dir.name}_{CheckpointBoundary._MARKER_SUFFIX}"
        )

    @staticmethod
    def _pending_step_log_path(checkpoint_dir: Path) -> Path:
        return (
            CheckpointBoundary._state_dir(checkpoint_dir)
            / f"{checkpoint_dir.name}_{CheckpointBoundary._PENDING_STEP_LOG_SUFFIX}"
        )

    @staticmethod
    def _state_dir(checkpoint_dir: Path) -> Path:
        return checkpoint_dir.parent / CheckpointBoundary._BOUNDARY_STATE_DIRNAME

    @staticmethod
    def _require_checkpoint_dir(checkpoint_dir: Path) -> None:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(
                "Checkpoint boundary state requires an existing checkpoint dir: "
                f"{checkpoint_dir}"
            )

    def _resume_checkpoint_dir(self) -> Path | None:
        resume_checkpoint = getattr(self.trainer.run_context, "resume_checkpoint", None)
        return Path(resume_checkpoint) if resume_checkpoint else None

    def _is_marked_complete(self, checkpoint_dir: Path) -> bool:
        marker_path = self._marker_path(checkpoint_dir)
        if not marker_path.exists():
            return False
        try:
            payload = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return int(payload.get("step", -1)) == int(self.trainer.global_step)

    def should_replay_on_resume(self) -> bool:
        checkpoint_dir = self._resume_checkpoint_dir()
        if checkpoint_dir is None:
            return False
        if self.trainer.global_step <= 0:
            return False
        if (
            not self.checkpoint_every
            or self.trainer.global_step % self.checkpoint_every != 0
        ):
            return False

        rank0_error = None
        should_replay = False
        if self.trainer.runtime.is_main_process:
            try:
                should_replay = not self._is_marked_complete(checkpoint_dir)
            except BaseException as exc:
                rank0_error = rank0_error_message(
                    "Checkpoint boundary marker read",
                    exc,
                )
        payload = self.trainer.runtime.broadcast_object(
            (rank0_error, should_replay),
            src=0,
        )
        rank0_error, should_replay = payload
        if rank0_error is not None:
            raise RuntimeError(rank0_error)
        return bool(should_replay)

    def restore_step_log_payload(self, checkpoint_dir: Path) -> dict[str, float]:
        rank0_error = None
        payload: dict[str, float] = {}
        if self.trainer.runtime.is_main_process:
            try:
                self._require_checkpoint_dir(checkpoint_dir)
                payload = self._read_step_log_payload(checkpoint_dir)
            except BaseException as exc:
                rank0_error = rank0_error_message(
                    "Checkpoint boundary step-log read",
                    exc,
                )
        rank0_error, payload = self.trainer.runtime.broadcast_object(
            (rank0_error, payload),
            src=0,
        )
        if rank0_error is not None:
            raise RuntimeError(rank0_error)
        return payload

    def write_step_log_payload(
        self,
        checkpoint_dir: Path,
        payload: dict[str, float],
    ) -> None:
        rank0_error = None
        if self.trainer.runtime.is_main_process:
            try:
                self._require_checkpoint_dir(checkpoint_dir)
                marker_path = self._pending_step_log_path(checkpoint_dir)
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = marker_path.with_name(
                    f"{marker_path.name}.tmp.{os.getpid()}"
                )
                tmp_path.write_text(
                    json.dumps(
                        {
                            "metrics": payload,
                            "step": int(self.trainer.global_step),
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                tmp_path.replace(marker_path)
            except BaseException as exc:
                rank0_error = rank0_error_message(
                    "Checkpoint boundary step-log write",
                    exc,
                )
        raise_if_rank0_error(self.trainer.runtime, rank0_error)

    def mark_complete(self, checkpoint_dir: Path) -> None:
        rank0_error = None
        if self.trainer.runtime.is_main_process:
            try:
                self._require_checkpoint_dir(checkpoint_dir)
                marker_path = self._marker_path(checkpoint_dir)
                marker_path.parent.mkdir(parents=True, exist_ok=True)
                marker_path.write_text(
                    json.dumps(
                        {
                            "complete": True,
                            "step": int(self.trainer.global_step),
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                self._pending_step_log_path(checkpoint_dir).unlink(
                    missing_ok=True,
                )
            except BaseException as exc:
                rank0_error = rank0_error_message(
                    "Checkpoint boundary marker write",
                    exc,
                )
        raise_if_rank0_error(self.trainer.runtime, rank0_error)

    def _read_step_log_payload(self, checkpoint_dir: Path) -> dict[str, float]:
        payload_path = self._pending_step_log_path(checkpoint_dir)
        if not payload_path.exists():
            return {}
        try:
            raw = json.loads(payload_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return {}
            if int(raw.get("step", -1)) != int(self.trainer.global_step):
                return {}
            metrics = raw.get("metrics", {})
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        if not isinstance(metrics, dict):
            return {}
        return {
            str(key): float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }
