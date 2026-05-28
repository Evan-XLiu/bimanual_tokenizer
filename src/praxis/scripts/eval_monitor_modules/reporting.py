"""Reporting/logging helpers for eval monitor."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any


class WandBState:
    """Lazy wandb connection that attaches to training run on first success."""

    def __init__(
        self,
        train_dir: Path,
        *,
        disabled: bool = False,
        run_info_filename: str = "wandb_run_info.json",
        logger: logging.Logger | None = None,
    ):
        self._train_dir = train_dir
        self._disabled = disabled
        self._run_info_filename = run_info_filename
        self._logger = logger or logging.getLogger("eval-monitor")
        self._run: Any = None
        self._metrics_defined = False
        self._run_dir: Path | None = None

    def _ensure(self) -> Any:
        if self._disabled or self._run is not None:
            return self._run
        info_path = self._train_dir / self._run_info_filename
        if not info_path.exists():
            return None
        try:
            info = json.loads(info_path.read_text())
            import wandb

            self._run_dir = Path(tempfile.mkdtemp(prefix="praxis-eval-monitor-wandb-"))
            self._run = wandb.init(
                id=info["id"],
                project=info.get("project", "praxis"),
                entity=info.get("entity"),
                resume="allow",
                dir=str(self._run_dir),
            )
            self._define_metrics()
            self._logger.info("Attached to wandb run %s", info["id"])
        except Exception as exc:
            self._disable(reason=f"wandb init failed: {exc}")
        return self._run

    def _define_metrics(self) -> None:
        if self._metrics_defined or self._run is None:
            return
        try:
            import wandb

            wandb.define_metric("sim_eval/eval_step")
            wandb.define_metric("sim_eval/*", step_metric="sim_eval/eval_step")
            self._metrics_defined = True
        except Exception as exc:
            self._logger.warning("wandb define_metric failed: %s", exc)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        run = self._ensure()
        if run is None:
            return
        try:
            import wandb

            payload = {"sim_eval/eval_step": int(step), **metrics}
            wandb.log(payload)
        except Exception as exc:
            self._disable(reason=f"wandb log failed: {exc}")

    def finish(self, *, exit_code: int = 0) -> None:
        if self._run is not None:
            try:
                import wandb

                wandb.finish(exit_code=exit_code, quiet=True)
            except Exception:
                pass
        self._run = None
        self._cleanup_run_dir()

    def _disable(self, *, reason: str) -> None:
        if self._disabled:
            return
        self._logger.warning("%s. Disabling.", reason)
        self._disabled = True
        self.finish(exit_code=0)

    def _cleanup_run_dir(self) -> None:
        run_dir = self._run_dir
        self._run_dir = None
        if run_dir is None:
            return
        with contextlib.suppress(Exception):
            shutil.rmtree(run_dir, ignore_errors=True)


def log_monitor_startup(
    *,
    logger: logging.Logger,
    train_dir: Path,
    checkpoint_dir: Path,
    poll_interval: int,
    num_eval_per_task: int,
    num_parallel_env: int,
    stale_timeout_sec: int,
    eval_timeout_sec: int,
    eval_overrides: list[str],
) -> None:
    logger.info("Train dir:      %s", train_dir)
    logger.info("Checkpoint dir: %s", checkpoint_dir)
    logger.info("Poll interval:  %ds", poll_interval)
    logger.info(
        "num_eval_per_task=%d  num_parallel_env=%d",
        num_eval_per_task,
        num_parallel_env,
    )
    logger.info(
        "stale_timeout_sec=%d  eval_timeout_sec=%d",
        stale_timeout_sec,
        eval_timeout_sec,
    )
    if eval_overrides:
        logger.info("eval_overrides: %s", eval_overrides)
