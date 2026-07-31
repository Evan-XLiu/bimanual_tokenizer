# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Reconstruction metric integration tests for X-Tokenizer."""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from praxis.codecs.latents.xtokenizer import XTokenizerCodec, XTokenizerModelConfig
from praxis.codecs.latents.xtokenizer.eval_metrics import (
    XTokenizerReconstructionMetricCollector,
)
from praxis.training.codecs_trainer import CodecsTrainer
from praxis.training.core.eval_collectors import (
    ReconstructionBatchContext,
    build_reconstruction_metric_collectors,
)


def _tiny_codec() -> XTokenizerCodec:
    return XTokenizerCodec(
        config=XTokenizerModelConfig.from_dict(
            {
                "action_dim": 10,
                "action_horizon": 2,
                "compression_ratio": 2,
                "hidden_dim": 8,
                "latent_dim": 8,
                "encoder_num_heads": 2,
                "encoder_depth": 1,
                "decoder_num_heads": 2,
                "decoder_depth": 1,
                "mam_num_heads": 2,
                "mam_depth": 1,
                "prediction_num_heads": 2,
                "prediction_depth": 1,
                "dropout": 0.0,
                "codebook_size": 4,
                "num_quantizers": 4,
                "dead_code_threshold": 0,
                "embodiment_vocab_size": 2,
                "vl_feature_dim": 4,
                "mam_weight": 0.0,
                "alignment_weight": 0.0,
                "prediction_weight": 0.0,
            }
        )
    )


def test_xtokenizer_registers_only_its_own_metric_collector() -> None:
    codec = _tiny_codec()

    collectors = build_reconstruction_metric_collectors(
        policy=codec,
        cfg=SimpleNamespace(),
        device=torch.device("cpu"),
    )
    ordinary_collectors = build_reconstruction_metric_collectors(
        policy=torch.nn.Identity(),
        cfg=SimpleNamespace(),
        device=torch.device("cpu"),
    )

    assert len(collectors) == 1
    assert type(collectors[0]).__module__ == (
        "praxis.codecs.latents.xtokenizer.eval_metrics"
    )
    assert getattr(collectors[0], "owns_default_reconstruction_loss", False) is True
    assert ordinary_collectors == []


def test_xtokenizer_owns_masked_reconstruction_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _tiny_codec()
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    target = torch.zeros(1, 2, 10)
    target[..., 3:9] = identity
    prediction = target.clone()
    prediction[0, 0, :3] = torch.tensor([1.0, 2.0, 3.0])
    prediction[0, 1] = float("nan")

    def _encode(
        self: XTokenizerCodec,
        actions: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, actions, kwargs
        return torch.zeros(1, 1, 4, dtype=torch.long)

    def _decode(
        self: XTokenizerCodec,
        indices: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, indices, kwargs
        return prediction

    class _UnexpectedDefaultCollector:
        def __init__(self, *, device: torch.device) -> None:
            del device
            raise AssertionError(
                "X-Tokenizer must replace the default reconstruction loss collector."
            )

    monkeypatch.setattr(XTokenizerCodec, "encode", _encode)
    monkeypatch.setattr(XTokenizerCodec, "decode", _decode)
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.ReconstLossCollector",
        _UnexpectedDefaultCollector,
    )

    class _Runtime:
        @staticmethod
        def reduce(tensor: torch.Tensor, *, reduction: str) -> torch.Tensor:
            assert reduction == "sum"
            return tensor

        @staticmethod
        def autocast():
            return contextlib.nullcontext()

    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.run_train_loop = True
    trainer.val_dataloader = [
        {
            "action": target,
            "action_is_pad": torch.tensor([[False, True]]),
            "observation.state": torch.zeros(1, 10),
            "robot_type": torch.zeros(1, dtype=torch.long),
        }
    ]
    trainer.max_reconst_batches = None
    trainer.action_key = "action"
    trainer.device = torch.device("cpu")
    trainer.runtime = _Runtime()
    trainer.model = codec
    trainer.cfg = SimpleNamespace()
    trainer._eval_progress = lambda iterable, **kwargs: iterable
    trainer._move_batch_tensors_to_device = lambda batch, *, phase: batch

    @contextlib.contextmanager
    def _ema_scope():
        yield codec

    trainer._ema_eval_scope = _ema_scope

    metrics = trainer.reconstruct()

    assert metrics["val_reconst_loss"] == pytest.approx(1.4)
    assert all(math.isfinite(value) for value in metrics.values())


def test_xtokenizer_rotation_metric_reports_radians_and_excludes_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = _tiny_codec()
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    invalid_rotation = torch.tensor([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    target = torch.zeros(1, 2, 10)
    target[..., 3:9] = identity
    prediction = target.clone()
    prediction[0, 0, 3:9] = z_90
    prediction[0, 1, 3:9] = invalid_rotation

    def _encode(
        self: XTokenizerCodec,
        actions: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, actions, kwargs
        return torch.zeros(1, 1, 4, dtype=torch.long)

    def _decode(
        self: XTokenizerCodec,
        indices: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, indices, kwargs
        return prediction

    monkeypatch.setattr(XTokenizerCodec, "encode", _encode)
    monkeypatch.setattr(XTokenizerCodec, "decode", _decode)

    class _Runtime:
        @staticmethod
        def reduce(tensor: torch.Tensor, *, reduction: str) -> torch.Tensor:
            assert reduction == "sum"
            return tensor

        @staticmethod
        def autocast():
            return contextlib.nullcontext()

    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.run_train_loop = True
    trainer.val_dataloader = [
        {
            "action": target,
            "action_is_pad": torch.tensor([[False, True]]),
            "observation.state": torch.zeros(1, 10),
            "robot_type": torch.zeros(1, dtype=torch.long),
        }
    ]
    trainer.max_reconst_batches = None
    trainer.action_key = "action"
    trainer.device = torch.device("cpu")
    trainer.runtime = _Runtime()
    trainer.model = codec
    trainer.cfg = SimpleNamespace()
    trainer._eval_progress = lambda iterable, **kwargs: iterable
    trainer._move_batch_tensors_to_device = lambda batch, *, phase: batch

    @contextlib.contextmanager
    def _ema_scope():
        yield codec

    trainer._ema_eval_scope = _ema_scope

    metrics = trainer.reconstruct()

    assert metrics["val_rotation_geodesic"] == pytest.approx(math.pi / 2)
    for level in range(4):
        assert metrics[f"val_codebook_perplexity_q{level}"] == pytest.approx(1.0)


def test_xtokenizer_metrics_use_reordered_named_position_and_rotation() -> None:
    layout = {
        "convention": "absolute",
        "dof_dims": {
            "follow_left_gripper": 1,
            "follow_left_ee_cartesian_pos": 3,
            "follow_left_ee_rotation_6D": 6,
        },
        "dof_order": [
            "follow_left_gripper",
            "follow_left_ee_cartesian_pos",
            "follow_left_ee_rotation_6D",
        ],
    }
    codec = XTokenizerCodec(
        config=XTokenizerModelConfig.from_dict(
            {
                **_tiny_codec().config.to_dict(),
                "action_layout": layout,
            }
        )
    )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    target = torch.zeros(1, 2, 10)
    target[..., 4:10] = identity
    prediction = target.clone()
    prediction[..., 0] = 100.0
    prediction[..., 1:4] = torch.tensor([1.0, 2.0, 3.0])
    prediction[..., 4:10] = z_90
    collector = XTokenizerReconstructionMetricCollector(
        codec,
        device=torch.device("cpu"),
    )

    collector.update(
        ReconstructionBatchContext(
            pred_action=prediction,
            gt_action=target,
            batch={},
            reconst_info={"xtokenizer_indices": torch.zeros(1, 1, 4, dtype=torch.long)},
            policy=codec,
            action_key="action",
        )
    )
    metrics = collector.finalize()

    assert metrics["val_translation_l1"] == pytest.approx(2.0)
    assert metrics["val_rotation_geodesic"] == pytest.approx(math.pi / 2)
