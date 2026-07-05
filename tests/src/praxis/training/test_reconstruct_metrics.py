# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reconstruction metric collectors."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.training.core.eval_collectors import (
    ReconstLossCollector,
    ReconstructionBatchContext,
    ReconstructionInfoAnnotator,
    ReconstructionMetricCollector,
    ReconstructionMetricPipeline,
    register_reconstruction_info_annotator_factory,
    register_reconstruction_metric_collector_factory,
    unregister_reconstruction_info_annotator_factory,
    unregister_reconstruction_metric_collector_factory,
)
from praxis.training.policy_trainer import PolicyTrainer


def _policy_io(
    *,
    obs_key: str = "observation.state",
    language_key: str | None = None,
) -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key=obs_key,
                type=FeatureType.STATE,
                shape=(4,),
                norm_mode=NormMode.MIN_MAX,
            )
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(2,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=1,
        n_action_steps=2,
        language_key=language_key,
    )


class _IdentityRuntime:
    def reduce(self, tensor: torch.Tensor, *, reduction: str) -> torch.Tensor:
        assert reduction == "sum"
        return tensor


class _NonTokenPolicy:
    pass


def _make_ctx(
    *,
    pred_action: torch.Tensor,
    gt_action: torch.Tensor,
    reconst_info: dict[str, Any],
    policy: Any,
) -> ReconstructionBatchContext:
    batch: dict[str, Any] = {"action": gt_action}
    return ReconstructionBatchContext(
        pred_action=pred_action,
        gt_action=gt_action,
        batch=batch,
        reconst_info=reconst_info,
        policy=policy,
        action_key="action",
    )


def test_pipeline_reports_reconstruction_loss() -> None:
    pred_action = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    gt_action = pred_action.clone()

    pipeline = ReconstructionMetricPipeline(
        [ReconstLossCollector(device=torch.device("cpu"))]
    )
    pipeline.update(
        _make_ctx(
            pred_action=pred_action,
            gt_action=gt_action,
            reconst_info={},
            policy=_NonTokenPolicy(),
        )
    )
    pipeline.reduce(_IdentityRuntime())
    out = pipeline.finalize()

    assert out == {"val_reconst_loss": pytest.approx(0.0)}


def test_metric_pipeline_applies_reconstruction_info_annotator() -> None:
    pred_action = torch.zeros((2, 2, 2), dtype=torch.float32)
    gt_action = torch.zeros((2, 2, 2), dtype=torch.float32)

    class _AnnotatedValueCollector:
        def __init__(self) -> None:
            self.total = torch.zeros(1)

        def update(self, ctx: ReconstructionBatchContext) -> None:
            self.total += float(ctx.reconst_info.get("annotated_value", 0.0))

        def reduce(self, runtime: Any) -> None:
            self.total = runtime.reduce(self.total, reduction="sum")

        def finalize(self) -> dict[str, float]:
            return {"val_annotated_value": float(self.total.item())}

    def _annotator(ctx: ReconstructionBatchContext) -> dict[str, Any]:
        del ctx
        return {"annotated_value": 3.0}

    pipeline = ReconstructionMetricPipeline(
        collectors=[_AnnotatedValueCollector()],
        annotators=[_annotator],
    )
    pipeline.update(
        _make_ctx(
            pred_action=pred_action,
            gt_action=gt_action,
            reconst_info={},
            policy=_NonTokenPolicy(),
        )
    )
    pipeline.reduce(_IdentityRuntime())
    out = pipeline.finalize()

    assert out["val_annotated_value"] == pytest.approx(3.0)


def test_policy_trainer_reconstruct_includes_registered_metric_hooks() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    gt_action = torch.zeros((1, 2, 2), dtype=torch.float32)

    class _RegisteredCollector:
        def __init__(self) -> None:
            self.total = torch.zeros(1)

        def update(self, ctx: ReconstructionBatchContext) -> None:
            self.total += float(ctx.reconst_info.get("registered_metric", 0.0))

        def reduce(self, runtime: Any) -> None:
            self.total = runtime.reduce(self.total, reduction="sum")

        def finalize(self) -> dict[str, float]:
            return {"debug/eval/registered_metric": float(self.total.item())}

    def _collector_hook(
        *,
        policy: Any,
        cfg: Any,
        device: torch.device,
    ) -> list[ReconstructionMetricCollector]:
        del policy, cfg, device
        return [_RegisteredCollector()]

    def _annotator_hook(
        *,
        policy: Any,
        cfg: Any,
        device: torch.device,
    ) -> list[ReconstructionInfoAnnotator]:
        del policy, cfg, device

        def _annotator(ctx: ReconstructionBatchContext) -> dict[str, Any]:
            del ctx
            return {"registered_metric": 2.0}

        return [_annotator]

    class _Accel:
        is_main_process = True

        def autocast(self):
            return nullcontext()

        def reduce(self, tensor, *, reduction: str):
            assert reduction == "sum"
            return tensor

    class _Model:
        def __init__(self) -> None:
            self.train_called = 0

        def train(self) -> None:
            self.train_called += 1

    class _Policy:
        def __init__(self) -> None:
            self.io = _policy_io()
            self.eval_called = 0

        def eval(self) -> None:
            self.eval_called += 1

        def predict_action_chunk(self, obs, **kwargs):
            del obs, kwargs
            return gt_action.clone(), {}

    policy = _Policy()
    model = _Model()
    stub = cast(Any, trainer)
    stub.val_dataloader = [
        {
            "observation.state": torch.zeros((1, 1, 4), dtype=torch.float32),
            "action": gt_action,
        }
    ]
    stub.action_key = "action"
    stub.device = torch.device("cpu")
    stub.runtime = _Accel()
    stub.max_reconst_batches = None
    stub.preprocess_batch = lambda batch: batch
    stub._move_batch_tensors_to_device = lambda batch, phase: batch
    stub._eval_progress = lambda iterable, desc, total: iterable
    stub._ema_eval_scope = lambda: nullcontext(policy)
    stub.model = model
    stub.cfg = SimpleNamespace()

    register_reconstruction_metric_collector_factory(_collector_hook)
    register_reconstruction_info_annotator_factory(_annotator_hook)
    try:
        out = trainer.reconstruct()
    finally:
        unregister_reconstruction_info_annotator_factory(_annotator_hook)
        unregister_reconstruction_metric_collector_factory(_collector_hook)

    assert out["val_reconst_loss"] == pytest.approx(0.0)
    assert out["debug/eval/registered_metric"] == pytest.approx(2.0)
    assert policy.eval_called == 1
    assert model.train_called == 1


def test_policy_trainer_reconstruct_uses_exact_policy_io_obs_keys() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    gt_action = torch.zeros((1, 2, 2), dtype=torch.float32)

    class _Accel:
        is_main_process = True

        def autocast(self):
            return nullcontext()

        def reduce(self, tensor, *, reduction: str):
            assert reduction == "sum"
            return tensor

    class _Model:
        def train(self) -> None:
            pass

    class _Policy:
        def __init__(self) -> None:
            self.io = _policy_io()
            self.seen_obs: dict[str, Any] | None = None

        def eval(self) -> None:
            pass

        def predict_action_chunk(self, obs, **kwargs):
            del kwargs
            self.seen_obs = dict(obs)
            return gt_action.clone(), {}

    policy = _Policy()
    stub = cast(Any, trainer)
    stub.val_dataloader = [
        {
            "observation.state": torch.zeros((1, 1, 4), dtype=torch.float32),
            "observation.images.extra": torch.ones(
                (1, 1, 3, 8, 8), dtype=torch.float32
            ),
            "action": gt_action,
        }
    ]
    stub.io = policy.io
    stub.action_key = "action"
    stub.device = torch.device("cpu")
    stub.runtime = _Accel()
    stub.max_reconst_batches = None
    stub.preprocess_batch = lambda batch: batch
    stub._move_batch_tensors_to_device = lambda batch, phase: batch
    stub._eval_progress = lambda iterable, desc, total: iterable
    stub._ema_eval_scope = lambda: nullcontext(policy)
    stub.model = _Model()
    stub.cfg = SimpleNamespace()

    out = trainer.reconstruct()

    assert out["val_reconst_loss"] == pytest.approx(0.0)
    assert policy.seen_obs is not None
    assert set(policy.seen_obs) == {"observation.state"}


def test_policy_trainer_reconstruct_fails_clearly_on_missing_policy_io_key() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    gt_action = torch.zeros((1, 2, 2), dtype=torch.float32)

    class _Accel:
        is_main_process = True

        def autocast(self):
            return nullcontext()

    class _Policy:
        io = _policy_io(obs_key="observation.images.front")

        def eval(self) -> None:
            pass

        def predict_action_chunk(self, obs, **kwargs):
            raise AssertionError("predict_action_chunk should not run")

    policy = _Policy()
    stub = cast(Any, trainer)
    stub.val_dataloader = [
        {
            "observation.images.wrist": torch.zeros(
                (1, 1, 3, 8, 8), dtype=torch.float32
            ),
            "action": gt_action,
        }
    ]
    stub.io = policy.io
    stub.action_key = "action"
    stub.device = torch.device("cpu")
    stub.runtime = _Accel()
    stub.max_reconst_batches = None
    stub.preprocess_batch = lambda batch: batch
    stub._move_batch_tensors_to_device = lambda batch, phase: batch
    stub._eval_progress = lambda iterable, desc, total: iterable
    stub._ema_eval_scope = lambda: nullcontext(policy)
    stub.model = SimpleNamespace(train=lambda: None)
    stub.cfg = SimpleNamespace()

    with pytest.raises(KeyError, match="Missing PolicyIO observation key"):
        trainer.reconstruct()
