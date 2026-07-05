# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Run-level training context and output directory lifecycle."""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class TrainingBarrier(Protocol):
    @property
    def is_main_process(self) -> bool: ...

    def wait_for_everyone(self) -> None: ...


@dataclass(frozen=True)
class TrainerRunContext:
    """Resolved run lifecycle for one trainer instance.

    ``overwrite`` runs own ``output_dir`` preparation and may remove stale
    contents. ``resume`` runs preserve the output directory and load from the
    explicit checkpoint path instead.
    """

    output_dir: Path | None
    resume_checkpoint: str | None
    run_mode: str = "overwrite"

    @property
    def has_output_dir(self) -> bool:
        return self.output_dir is not None

    @property
    def owns_output_dir(self) -> bool:
        return (
            self.output_dir is not None
            and self.run_mode == "overwrite"
            and self.resume_checkpoint is None
        )

    def require_output_dir(self, *, purpose: str) -> Path:
        if self.output_dir is None:
            raise RuntimeError(f"Trainer output_dir is required for {purpose}.")
        return self.output_dir

    def completion_sentinel_path(self) -> Path | None:
        if self.output_dir is None:
            return None
        return self.output_dir / "TRAINING_COMPLETE"


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def resolve_trainer_run_context(cfg: DictConfig) -> TrainerRunContext:
    """Resolve output/resume settings and enforce their lifecycle contract."""
    output_dir_raw = OmegaConf.select(cfg, "output_dir", default=None)
    run_mode = str(OmegaConf.select(cfg, "run_mode", default="overwrite")).strip()
    resume_checkpoint = _optional_str(
        OmegaConf.select(cfg, "resume_checkpoint", default=None)
    )
    if run_mode not in {"overwrite", "resume"}:
        raise ValueError(
            f"Unsupported run_mode={run_mode!r}. Available modes: overwrite, resume."
        )
    if run_mode == "resume" and resume_checkpoint is None:
        raise ValueError("run_mode=resume requires resume_checkpoint to be set.")
    if run_mode == "overwrite" and resume_checkpoint is not None:
        raise ValueError(
            "resume_checkpoint is only valid with run_mode=resume. "
            "Use run_mode=resume to continue a run, or clear resume_checkpoint "
            "to overwrite output_dir."
        )
    output_dir = None if output_dir_raw is None else Path(str(output_dir_raw))
    return TrainerRunContext(
        output_dir=output_dir,
        run_mode=run_mode,
        resume_checkpoint=resume_checkpoint,
    )


def _rmtree_with_retries(path: Path, *, attempts: int = 3) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.2 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _trash_path_for(path: Path) -> Path:
    trash_dir = path.parent / ".trainer_trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    return trash_dir / f"{path.name}.{os.getpid()}.{time.time_ns()}"


def _remove_existing_path(path: Path) -> None:
    """Remove an existing file, symlink, or directory.

    Shared filesystems can transiently report non-empty directories during
    recursive deletion. If direct removal still fails after retries, move the
    stale run aside atomically so the fresh run gets a clean output path.
    """
    if path.is_dir() and not path.is_symlink():
        try:
            _rmtree_with_retries(path)
            return
        except OSError as exc:
            trash_path = _trash_path_for(path)
            logger.warning(
                "Could not fully remove existing output_dir %s (%s); "
                "moving it aside to %s.",
                path,
                exc,
                trash_path,
            )
            path.rename(trash_path)
            try:
                _rmtree_with_retries(trash_path)
            except OSError as trash_exc:
                logger.warning(
                    "Stale output_dir was moved aside but could not be fully "
                    "deleted: %s (%s).",
                    trash_path,
                    trash_exc,
                )
    else:
        path.unlink()


class RunDirectoryManager:
    """Prepare ``output_dir`` only for fresh overwrite-mode training runs."""

    def prepare(
        self,
        *,
        context: TrainerRunContext,
        cfg: DictConfig,
        runtime: TrainingBarrier,
    ) -> None:
        if context.output_dir is None or not context.owns_output_dir:
            return

        self._prepare_fresh_output_dir(context.output_dir, runtime=runtime)
        self._write_fresh_config_snapshot(
            context=context,
            cfg=cfg,
            runtime=runtime,
        )

    def _prepare_fresh_output_dir(
        self,
        output_dir: Path,
        *,
        runtime: TrainingBarrier,
    ) -> None:
        resolved = output_dir.expanduser().resolve()
        if resolved == resolved.parent:
            raise ValueError(f"Refusing to overwrite filesystem root: {output_dir}")

        main_process_first = getattr(runtime, "main_process_first", None)
        if callable(main_process_first):
            scoped_barrier = cast(
                Callable[[], AbstractContextManager[None]], main_process_first
            )
            with scoped_barrier():
                if runtime.is_main_process:
                    if output_dir.exists() or output_dir.is_symlink():
                        logger.warning(
                            "Fresh training run is overwriting existing output_dir: %s",
                            output_dir,
                        )
                        _remove_existing_path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
            return

        if runtime.is_main_process:
            if output_dir.exists() or output_dir.is_symlink():
                logger.warning(
                    "Fresh training run is overwriting existing output_dir: %s",
                    output_dir,
                )
                _remove_existing_path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        runtime.wait_for_everyone()

    def _write_fresh_config_snapshot(
        self,
        *,
        context: TrainerRunContext,
        cfg: DictConfig,
        runtime: TrainingBarrier,
    ) -> None:
        if context.output_dir is None:
            return
        output_dir = context.output_dir

        def _write() -> None:
            if runtime.is_main_process:
                config_path = output_dir / "train_config.yaml"
                config_path.write_text(OmegaConf.to_yaml(cfg))

        main_process_first = getattr(runtime, "main_process_first", None)
        if callable(main_process_first):
            scoped_barrier = cast(
                Callable[[], AbstractContextManager[None]], main_process_first
            )
            with scoped_barrier():
                _write()
            return

        _write()
        runtime.wait_for_everyone()
