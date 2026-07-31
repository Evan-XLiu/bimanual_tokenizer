# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Hydra and trainer integration coverage for X-Tokenizer pretraining."""

from __future__ import annotations

import contextlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from praxis.codecs.latents.xtokenizer import XTokenizerCodec, XTokenizerModelConfig
from praxis.training.codecs_trainer import CodecsTrainer
from praxis.training.core.trainer import DefaultTrainer

_ARTIFACT_ROOT = "/mnt/robot-research-exp-0/user/hxs"
_XTOKENIZER_STATS_ROOT = f"{_ARTIFACT_ROOT}/xtokenizer/bridge_single_view"


def _compose(overrides: list[str]):
    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="train_codecs", overrides=overrides)


def test_xtokenizer_default_stats_path_is_horizon_qualified() -> None:
    cfg = _compose(
        [
            "codec=latents/xtokenizer",
            "dataset=xtokenizer_bridge",
            "train_recipe/codecs=xtokenizer",
        ]
    )

    expected_stats = f"{_XTOKENIZER_STATS_ROOT}/stats_v3_h32.json"
    assert cfg.codec.action_horizon == 32
    assert cfg.dataset.action_horizon == 32
    assert cfg.dataset.xtokenizer_quantile_stats == expected_stats
    assert cfg.dataset.canonical_stats == expected_stats


def test_xtokenizer_train_config_composes_with_documented_cli() -> None:
    cfg = _compose(
        [
            "codec=latents/xtokenizer",
            "dataset=xtokenizer_bridge",
            "train_recipe/codecs=xtokenizer",
        ]
    )

    assert cfg.codec.kind == "latents"
    assert cfg.codec.name == "xtokenizer"
    assert cfg.codec.action_horizon == 32
    assert cfg.codec.compression_ratio == 4
    assert cfg.codec.codebook_size == 2048
    assert cfg.codec.num_quantizers == 4
    assert "quantizer_levels" not in cfg.codec
    assert "num_heads" not in cfg.codec
    assert "depth" not in cfg.codec
    assert cfg.codec.encoder_num_heads == 8
    assert cfg.codec.encoder_depth == 12
    assert cfg.codec.decoder_num_heads == 8
    assert cfg.codec.decoder_depth == 4
    assert cfg.codec.mam_num_heads == 4
    assert cfg.codec.mam_depth == 2
    assert cfg.codec.prediction_num_heads == 4
    assert cfg.codec.prediction_depth == 2
    assert cfg.trainer._trainer_type == "codecs"
    assert {
        "reconstruction_weight": cfg.codec.reconstruction_weight,
        "mam_weight": cfg.codec.mam_weight,
        "alignment_weight": cfg.codec.alignment_weight,
        "prediction_weight": cfg.codec.prediction_weight,
    } == pytest.approx(
        {
            "reconstruction_weight": 1.0,
            "mam_weight": 0.1,
            "alignment_weight": 0.5,
            "prediction_weight": 0.2,
        }
    )
    assert cfg.trainer.lr == pytest.approx(5.0e-5)
    assert cfg.trainer.weight_decay == pytest.approx(0.01)
    assert cfg.trainer.lr_scheduler == "cosine"
    assert cfg.trainer.warmup_steps == 200
    assert cfg.trainer.batch_size == 256
    assert cfg.trainer.global_batch_size == 256
    assert cfg.dataset.action_horizon == 32
    assert dict(cfg.dataset.action_layout.dof_dims) == {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
        "follow_left_gripper": 1,
    }
    assert list(cfg.dataset.action_layout.dof_order) == list(
        cfg.dataset.action_layout.dof_dims
    )
    for path_key in (
        "root",
        "cache_dir",
        "vl_feature_sidecar",
        "xtokenizer_quantile_stats",
        "canonical_stats",
    ):
        assert str(cfg.dataset[path_key]).startswith(f"{_ARTIFACT_ROOT}/")


@pytest.mark.parametrize("horizon", [8, 16, 32, 64])
def test_xtokenizer_train_config_accepts_horizon_matched_overrides(
    horizon: int,
) -> None:
    stats_path = f"{_XTOKENIZER_STATS_ROOT}/stats_v3_h{horizon}.json"
    cfg = _compose(
        [
            "codec=latents/xtokenizer",
            "dataset=xtokenizer_bridge",
            "train_recipe/codecs=xtokenizer",
            f"codec.action_horizon={horizon}",
            f"dataset.action_horizon={horizon}",
            f"dataset.xtokenizer_quantile_stats={stats_path}",
            "codec.codebook_size=16",
            "codec.latent_dim=64",
            "trainer.max_steps=2",
            "trainer.batch_size=2",
        ]
    )

    assert cfg.codec.action_horizon == horizon
    assert cfg.dataset.action_horizon == horizon
    assert cfg.dataset.xtokenizer_quantile_stats == stats_path
    assert cfg.dataset.canonical_stats == stats_path
    assert cfg.codec.codebook_size == 16
    assert cfg.codec.latent_dim == 64
    assert cfg.trainer.max_steps == 2
    assert cfg.trainer.batch_size == 2
    assert cfg.trainer.global_batch_size == 2


def test_legacy_codec_group_override_remains_compatible() -> None:
    cfg = _compose(["codecs@codec=tokens/bin"])

    assert cfg.codec.kind == "tokens"
    assert cfg.codec.name == "bin"


def test_codecs_trainer_advances_epoch_only_after_training_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Codec:
        def __init__(self) -> None:
            self.context_steps: list[int] = []
            self.epochs: list[int] = []
            self.current_epoch = -1

        def set_training_context(self, context: Any) -> None:
            self.context_steps.append(int(context.global_step))

        def set_training_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)
            self.current_epoch = epoch

    codec = _Codec()
    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.model = codec
    trainer.runtime = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.global_step = 0
    trainer.dataloader = [object(), object()]
    trainer.tc = SimpleNamespace(gradient_accumulation_steps=1)

    forward_epochs: list[int] = []

    def _training_forward(
        self: DefaultTrainer,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float], bool]:
        del batch
        self._publish_training_context_to_model()
        forward_epochs.append(codec.current_epoch)
        return torch.tensor(0.0), {}, True

    monkeypatch.setattr(DefaultTrainer, "train_step", _training_forward)

    # Eval-boundary publication must not consume a training batch.
    trainer._publish_training_context_to_model()
    trainer._publish_training_context_to_model()

    trainer.train_step({})
    trainer._publish_training_context_to_model()
    trainer.train_step({})
    trainer.train_step({})

    assert forward_epochs == [0, 0, 1]
    assert trainer._training_batches_seen == 3


def test_codecs_trainer_initializes_epoch_from_resumed_optimizer_step() -> None:
    class _Codec:
        def __init__(self) -> None:
            self.epochs: list[int] = []

        def set_training_epoch(self, epoch: int) -> None:
            self.epochs.append(epoch)

    codec = _Codec()
    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.model = codec
    trainer.runtime = SimpleNamespace(unwrap_model=lambda model: model)
    trainer.global_step = 3
    trainer.dataloader = [object(), object(), object(), object()]
    trainer.tc = SimpleNamespace(gradient_accumulation_steps=2)

    trainer._publish_training_context_to_model()

    assert codec.epochs == [1]
    assert not hasattr(trainer, "_training_batches_seen")


def test_xtokenizer_reconstruction_reports_physical_and_codebook_metrics() -> None:
    torch.manual_seed(3)
    codec = XTokenizerCodec(
        config=XTokenizerModelConfig.from_dict(
            {
                "action_dim": 10,
                "action_horizon": 4,
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
    actions = torch.zeros(2, 4, 10)
    actions[..., :3] = torch.tensor([0.1, -0.2, 0.3])
    actions[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    action_is_pad = torch.zeros(2, 4, dtype=torch.bool)
    action_is_pad[1, -1] = True
    actions[1, -1] = float("nan")

    class _Runtime:
        is_main_process = True

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
            "action": actions,
            "action_is_pad": action_is_pad,
            "observation.state": torch.zeros(2, 10),
            "robot_type": torch.tensor([0, 1]),
        }
    ]
    trainer.max_reconst_batches = None
    trainer.action_key = "action"
    trainer.device = torch.device("cpu")
    trainer.runtime = _Runtime()
    trainer.model = codec
    trainer.cfg = SimpleNamespace(
        codec=SimpleNamespace(kind="latents", name="xtokenizer")
    )
    trainer._eval_progress = lambda iterable, **kwargs: iterable
    trainer._move_batch_tensors_to_device = lambda batch, *, phase: batch

    @contextlib.contextmanager
    def _ema_scope():
        yield codec

    trainer._ema_eval_scope = _ema_scope

    metrics = trainer.reconstruct()

    expected = {
        "val_reconst_loss",
        "val_translation_l1",
        "val_rotation_geodesic",
        *(f"val_codebook_perplexity_q{level}" for level in range(4)),
    }
    assert expected.issubset(metrics)
    assert all(math.isfinite(metrics[key]) for key in expected)
