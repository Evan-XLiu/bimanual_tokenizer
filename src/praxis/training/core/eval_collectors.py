# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Collector pipeline for policy reconstruction metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, TypeVar, cast

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ReconstructionBatchContext:
    """Per-batch reconstruction context passed to collectors."""

    pred_action: Tensor
    gt_action: Tensor
    batch: dict[str, Any]
    reconst_info: dict[str, Any]
    policy: Any
    action_key: str


class ReconstructionMetricCollector(Protocol):
    """Interface for DDP-safe reconstruction metric collectors."""

    def update(self, ctx: ReconstructionBatchContext) -> None: ...

    def reduce(self, runtime: Any) -> None: ...

    def finalize(self) -> dict[str, float]: ...


class ReconstructionMetricCollectorFactory(Protocol):
    """Build optional policy/scheme-specific reconstruction metric collectors."""

    def __call__(
        self,
        *,
        policy: Any,
        cfg: Any,
        device: torch.device,
    ) -> list[ReconstructionMetricCollector]: ...


class ReconstructionInfoAnnotator(Protocol):
    """Attach derived per-batch reconstruction info before metric collection."""

    def __call__(self, ctx: ReconstructionBatchContext) -> dict[str, Any]: ...


class ReconstructionInfoAnnotatorFactory(Protocol):
    """Build optional policy/scheme-specific reconstruction info annotators."""

    def __call__(
        self,
        *,
        policy: Any,
        cfg: Any,
        device: torch.device,
    ) -> list[ReconstructionInfoAnnotator]: ...


_CollectorFactoryT = TypeVar(
    "_CollectorFactoryT", bound=ReconstructionMetricCollectorFactory
)
_RECONSTRUCTION_METRIC_COLLECTOR_FACTORIES: list[
    ReconstructionMetricCollectorFactory
] = []
_AnnotatorFactoryT = TypeVar(
    "_AnnotatorFactoryT", bound=ReconstructionInfoAnnotatorFactory
)
_RECONSTRUCTION_INFO_ANNOTATOR_FACTORIES: list[ReconstructionInfoAnnotatorFactory] = []


def register_reconstruction_metric_collector_factory(
    factory: _CollectorFactoryT,
) -> _CollectorFactoryT:
    """Register a scheme-owned reconstruction metric collector factory."""
    if factory not in _RECONSTRUCTION_METRIC_COLLECTOR_FACTORIES:
        _RECONSTRUCTION_METRIC_COLLECTOR_FACTORIES.append(factory)
    return factory


def unregister_reconstruction_metric_collector_factory(
    factory: ReconstructionMetricCollectorFactory,
) -> None:
    """Remove a registered collector factory, primarily for isolated tests."""
    if factory in _RECONSTRUCTION_METRIC_COLLECTOR_FACTORIES:
        _RECONSTRUCTION_METRIC_COLLECTOR_FACTORIES.remove(factory)


def register_reconstruction_info_annotator_factory(
    factory: _AnnotatorFactoryT,
) -> _AnnotatorFactoryT:
    """Register a scheme-owned reconstruction-info annotator factory."""
    if factory not in _RECONSTRUCTION_INFO_ANNOTATOR_FACTORIES:
        _RECONSTRUCTION_INFO_ANNOTATOR_FACTORIES.append(factory)
    return factory


def unregister_reconstruction_info_annotator_factory(
    factory: ReconstructionInfoAnnotatorFactory,
) -> None:
    """Remove a registered annotator factory, primarily for isolated tests."""
    if factory in _RECONSTRUCTION_INFO_ANNOTATOR_FACTORIES:
        _RECONSTRUCTION_INFO_ANNOTATOR_FACTORIES.remove(factory)


def build_reconstruction_metric_collectors(
    *,
    policy: Any,
    cfg: Any,
    device: torch.device,
) -> list[ReconstructionMetricCollector]:
    """Build optional reconstruction collectors contributed by registered hooks."""
    collectors: list[ReconstructionMetricCollector] = []
    for factory in _RECONSTRUCTION_METRIC_COLLECTOR_FACTORIES:
        collectors.extend(factory(policy=policy, cfg=cfg, device=device))
    return collectors


def build_reconstruction_info_annotators(
    *,
    policy: Any,
    cfg: Any,
    device: torch.device,
) -> list[ReconstructionInfoAnnotator]:
    """Build optional reconstruction info annotators contributed by hooks."""
    annotators: list[ReconstructionInfoAnnotator] = []
    for factory in _RECONSTRUCTION_INFO_ANNOTATOR_FACTORIES:
        annotators.extend(factory(policy=policy, cfg=cfg, device=device))
    return annotators


class ReconstructionMetricPipeline:
    """Composable orchestrator for reconstruction metric collectors."""

    def __init__(
        self,
        collectors: list[ReconstructionMetricCollector],
        *,
        annotators: list[ReconstructionInfoAnnotator] | None = None,
    ) -> None:
        self.collectors = collectors
        self.annotators = [] if annotators is None else annotators

    def update(self, ctx: ReconstructionBatchContext) -> None:
        if self.annotators:
            info = dict(ctx.reconst_info)
            for annotator in self.annotators:
                extra = annotator(replace(ctx, reconst_info=info))
                if not extra:
                    continue
                overlap = sorted(set(info) & set(extra))
                if overlap:
                    raise RuntimeError(
                        f"Duplicate reconstruction info annotation key(s): {overlap}."
                    )
                info.update(extra)
            ctx = replace(ctx, reconst_info=info)

        for collector in self.collectors:
            collector.update(ctx)

    def reduce(self, runtime: Any) -> None:
        for collector in self.collectors:
            collector.reduce(runtime)

    def finalize(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for collector in self.collectors:
            metrics = collector.finalize()
            for key, value in metrics.items():
                if key in out:
                    raise RuntimeError(
                        f"Duplicate reconstruction metric key emitted: {key!r}."
                    )
                out[key] = value
        return out


class ReconstLossCollector:
    """Weighted action-space reconstruction loss collector."""

    def __init__(self, *, device: torch.device) -> None:
        self.loss_info = torch.zeros(2, device=device)

    def update(self, ctx: ReconstructionBatchContext) -> None:
        loss_value = F.mse_loss(ctx.pred_action, ctx.gt_action)
        batch_size = int(ctx.gt_action.shape[0])
        self.loss_info[0] += loss_value.item() * batch_size
        self.loss_info[1] += batch_size

    def reduce(self, runtime: Any) -> None:
        self.loss_info = cast(Tensor, runtime.reduce(self.loss_info, reduction="sum"))

    def finalize(self) -> dict[str, float]:
        if self.loss_info[1] == 0:
            return {}
        return {"val_reconst_loss": (self.loss_info[0] / self.loss_info[1]).item()}
