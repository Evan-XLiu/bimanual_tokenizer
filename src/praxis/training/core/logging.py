# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Logging utilities for training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Logger:
    """Thin logging wrapper supporting console and optional WandB."""

    def __init__(
        self,
        use_wandb: bool = False,
        project: str = "praxis",
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_dir: str | None = None,
        local_log_path: str | None = None,
        resume_id: str | None = None,
        wandb_init_timeout: float | None = None,
    ) -> None:
        self.use_wandb = use_wandb
        self._wandb_run = None
        self.local_log_path = Path(local_log_path) if local_log_path else None
        if self.local_log_path is not None:
            self.local_log_path.parent.mkdir(parents=True, exist_ok=True)

        if use_wandb:
            try:
                import wandb

                init_kwargs: dict[str, Any] = {
                    "project": project,
                    "entity": entity,
                    "config": config,
                }
                if run_name:
                    init_kwargs["name"] = run_name
                if run_dir:
                    run_dir_path = Path(run_dir)
                    run_dir_path.mkdir(parents=True, exist_ok=True)
                    init_kwargs["dir"] = str(run_dir_path)
                if resume_id:
                    init_kwargs["id"] = resume_id
                    init_kwargs["resume"] = "allow"
                if wandb_init_timeout is not None:
                    init_kwargs["settings"] = wandb.Settings(
                        init_timeout=float(wandb_init_timeout)
                    )

                self._wandb_run = wandb.init(**init_kwargs)
                # Persist run info so external eval monitors can attach.
                if run_dir:
                    _info = {
                        "id": self._wandb_run.id,
                        "project": project,
                        "entity": entity,
                    }
                    (Path(run_dir) / "wandb_run_info.json").write_text(
                        json.dumps(_info) + "\n"
                    )
            except (
                Exception
            ) as exc:  # pragma: no cover - depends on external package/runtime
                self.use_wandb = False
                logger.warning(
                    "W&B disabled due to initialization failure (%s). Falling back to local logging only.",
                    exc,
                )

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics."""
        if self.local_log_path is not None:
            record = {"step": int(step), **metrics}
            with self.local_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def log_videos(
        self, video_paths: list[str], step: int, key: str = "media/videos"
    ) -> None:
        """Log video artifacts to W&B when available."""
        if not (self.use_wandb and self._wandb_run is not None):
            return

        existing_paths = [p for p in video_paths if Path(p).exists()]
        if not existing_paths:
            return

        import wandb

        videos = [
            wandb.Video(path, caption=Path(path).name, format="mp4")
            for path in existing_paths
        ]
        wandb.log({key: videos}, step=step)

    def finish(self) -> None:
        """Finalize logging."""
        if self.use_wandb and self._wandb_run is not None:
            import wandb

            wandb.finish()
