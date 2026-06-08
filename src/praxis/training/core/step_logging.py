# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Step-level training metric aggregation and logging."""

from __future__ import annotations

from numbers import Real
from typing import Any

import torch

from praxis.training.core.rank_errors import raise_if_rank0_error, rank0_error_message


def _coerce_train_info_value(value: Any, *, key: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise TypeError(
                f"Training info metric {key!r} must be scalar, got tensor shape {tuple(value.shape)}."
            )
        return float(value.detach().item())
    if isinstance(value, Real):
        return float(value)
    raise TypeError(
        f"Training info metric {key!r} must be a scalar float/int or 0D tensor, got {type(value).__name__}."
    )


def accumulate_train_info_metrics(
    metric_sums: dict[str, float],
    metric_weights: dict[str, float],
    info: dict[str, Any],
    *,
    weight: int,
) -> None:
    batch_weight = float(weight)
    if batch_weight <= 0:
        return
    for key, value in info.items():
        scalar = _coerce_train_info_value(value, key=key)
        metric_sums[key] = metric_sums.get(key, 0.0) + (scalar * batch_weight)
        metric_weights[key] = metric_weights.get(key, 0.0) + batch_weight


def reduce_train_info_metrics(
    runtime: Any,
    metric_sums: dict[str, float],
    metric_weights: dict[str, float],
) -> dict[str, float]:
    gathered = runtime.gather_for_metrics(
        [{"metric_sums": metric_sums, "metric_weights": metric_weights}],
        use_gather_object=True,
    )
    if not runtime.is_main_process:
        return {}

    entries = gathered if isinstance(gathered, list) else [gathered]
    reduced_sums: dict[str, float] = {}
    reduced_weights: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        local_sums = entry.get("metric_sums", {})
        local_weights = entry.get("metric_weights", {})
        if not isinstance(local_sums, dict) or not isinstance(local_weights, dict):
            continue
        for key, value in local_sums.items():
            if key not in local_weights:
                continue
            reduced_sums[key] = reduced_sums.get(key, 0.0) + float(value)
            reduced_weights[key] = reduced_weights.get(key, 0.0) + float(
                local_weights[key]
            )

    return {
        key: reduced_sums[key] / reduced_weights[key]
        for key in sorted(reduced_sums)
        if reduced_weights.get(key, 0.0) > 0.0
    }


class StepLogPayload:
    """Builds and logs one W&B/local metric payload for a global step."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self._metrics: dict[str, float] = {}

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def restore_metrics(self, metrics: dict[str, float]) -> None:
        collisions = set(self._metrics) & set(metrics)
        if collisions:
            raise RuntimeError(
                f"Metric key collision within one logging step: {sorted(collisions)}."
            )
        self._metrics.update(metrics)

    def add_metrics(
        self,
        metrics: dict[str, float],
        *,
        default_group: str,
    ) -> None:
        formatted = self.trainer._format_logged_metrics(
            metrics,
            default_group=default_group,
        )
        collisions = set(self._metrics) & set(formatted)
        if collisions:
            raise RuntimeError(
                f"Metric key collision within one logging step: {sorted(collisions)}."
            )
        self._metrics.update(formatted)

    def add_train_info_metrics(self, metrics: dict[str, float]) -> None:
        """Add model diagnostics without colliding with trainer-owned train keys."""
        if not metrics:
            return
        trainer_owned_keys = set(self._metrics)
        safe_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            namespaced = self.trainer._namespace_logged_metric_key(
                key,
                default_group="train",
            )
            if namespaced in trainer_owned_keys:
                safe_metrics[f"train/model/{key.replace('/', '_')}"] = value
            else:
                safe_metrics[key] = value
        self.add_metrics(safe_metrics, default_group="train")

    def log_if_due(self, *, did_eval: bool) -> None:
        trainer = self.trainer
        if trainer.global_step % trainer.tc.log_every != 0 and not did_eval:
            return

        rank0_error = None
        if trainer.runtime.is_main_process and self._metrics:
            try:
                trainer.logger.log(self._metrics, step=trainer.global_step)
            except BaseException as exc:
                rank0_error = rank0_error_message(
                    "Metric logging",
                    exc,
                )
        raise_if_rank0_error(trainer.runtime, rank0_error)
