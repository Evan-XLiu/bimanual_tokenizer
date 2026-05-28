"""Structural protocols used by checkpoint services."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

_T = TypeVar("_T")


class CheckpointRuntimeProtocol(Protocol):
    """Runtime collectives and model helpers required by checkpoint I/O."""

    global_rank: int

    @property
    def is_main_process(self) -> bool: ...

    def unwrap_model(self, model: nn.Module) -> nn.Module: ...

    def wait_for_everyone(self) -> None: ...

    def broadcast_object(self, obj: _T, *, src: int = 0) -> _T: ...

    def gather_object(self, obj: _T) -> list[_T]: ...


class CheckpointEmaProtocol(Protocol):
    """EMA surface needed for checkpoint save/load and exports."""

    @property
    def restore_needed(self) -> bool: ...

    def refresh(self, model: nn.Module) -> None: ...

    def apply(
        self,
        model: nn.Module,
        *,
        backup_device: torch.device | str | None = None,
    ) -> None: ...

    def restore(self, model: nn.Module) -> None: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None: ...


class CheckpointTrainerProtocol(Protocol):
    """Trainer fields consumed by strategy-specific checkpoint I/O."""

    runtime: CheckpointRuntimeProtocol
    model: nn.Module
    optimizer: torch.optim.Optimizer | None
    scheduler: LRScheduler | None
    ema: CheckpointEmaProtocol | None
    global_step: int
    cfg: Any

    def _serialize_model_config(self, model: nn.Module) -> dict[str, Any] | None: ...


class CheckpointTrainerConfigProtocol(Protocol):
    """Trainer config fields consumed by checkpoint orchestration."""

    distributed_strategy: object
    checkpoint_dir: str | Path


class TopKCheckpointManagerProtocol(Protocol):
    """Top-K retention manager surface used by checkpoint orchestration."""

    metric_key: str

    def candidate_path_if_qualified(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> Path | None: ...

    def record_saved_checkpoint(self, step: int, value: float) -> None: ...

    def consider(
        self,
        step: int,
        metrics: dict[str, float],
        save_fn: Callable[[Path], None],
    ) -> bool: ...


class RecentCheckpointManagerProtocol(Protocol):
    """Recent-checkpoint retention manager surface used by orchestration."""

    def checkpoint_path(self, step: int) -> Path: ...

    def record_saved_checkpoint(self, step: int) -> None: ...

    def consider(self, step: int, save_fn: Callable[[Path], None]) -> bool: ...


class CheckpointServiceTrainerProtocol(CheckpointTrainerProtocol, Protocol):
    """Trainer facade required by the checkpoint orchestration service."""

    tc: CheckpointTrainerConfigProtocol
    topk_manager: TopKCheckpointManagerProtocol | None
    recent_manager: RecentCheckpointManagerProtocol | None

    def _save_checkpoint_dir(
        self,
        checkpoint_dir: Path,
        *,
        apply_ema_weights: bool = False,
        include_training_state: bool = True,
    ) -> None: ...
