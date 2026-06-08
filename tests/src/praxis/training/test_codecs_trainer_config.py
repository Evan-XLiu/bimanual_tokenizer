# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for codec trainer config resolution helpers."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from praxis.training.codecs_trainer import (
    CodecsTrainer,
    _extract_codec_action_dim,
    _resolve_codec_build_cfg,
    _should_skip_codec_fit,
    _supports_train_loop,
)


class _DummyCodecWithAttr(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim


class _DummyConfig:
    def __init__(self, action_dim: int) -> None:
        self.action_dim = action_dim


class _DummyCodecWithConfig(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.config = _DummyConfig(action_dim)


class _FitCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fit_calls: list[torch.Tensor] = []

    def fit(self, actions: torch.Tensor) -> None:
        self.fit_calls.append(actions.clone())


class _ReconstructCodec(nn.Module):
    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens


class _BootstrapRuntime:
    def __init__(self, *, is_main_process: bool) -> None:
        self.is_main_process = is_main_process
        self.wait_calls = 0

    @staticmethod
    def unwrap_model(model):
        return model

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def broadcast_object(self, obj, *, src: int = 0):
        del src
        return obj

    @contextlib.contextmanager
    def autocast(self):
        yield


class _BootstrapLogger:
    def __init__(self) -> None:
        self.records: list[tuple[dict[str, float], int]] = []

    def log(self, metrics: dict[str, float], step: int) -> None:
        self.records.append((metrics, step))


def _make_bootstrap_trainer(*, is_main_process: bool, skip_fit: bool = False) -> Any:
    from praxis.training.codecs_trainer import CodecsTrainer

    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(skip_codec_fit=skip_fit),
        codec=SimpleNamespace(max_fit_chunks=3),
    )
    trainer.action_key = "action"
    trainer.dataloader = [
        {"action": torch.ones(2, 4, dtype=torch.float32)},
        {"action": torch.full((2, 4), 2.0, dtype=torch.float32)},
    ]
    trainer.fit_dataloader = trainer.dataloader
    trainer.model = _FitCodec()
    trainer.runtime = _BootstrapRuntime(is_main_process=is_main_process)
    trainer.logger = _BootstrapLogger()
    trainer.saved = 0
    trainer.ended = 0
    trainer.save_latest_checkpoint = lambda: setattr(  # type: ignore[method-assign]
        trainer, "saved", trainer.saved + 1
    )
    trainer._on_train_end = lambda: setattr(  # type: ignore[method-assign]
        trainer, "ended", trainer.ended + 1
    )
    return trainer


def test_resolve_codec_build_cfg_overrides_action_dim_for_fresh_run() -> None:
    cfg = {
        "name": "oat",
        "kind": "tokens",
        "action_dim": 8,
        "pretrained_path": None,
    }
    resolved = _resolve_codec_build_cfg(
        cfg, dataset_action_dim=7, is_main_process=False
    )
    assert resolved["action_dim"] == 7


def test_resolve_codec_build_cfg_keeps_action_dim_for_pretrained() -> None:
    cfg = {
        "name": "oat",
        "kind": "tokens",
        "action_dim": 8,
        "pretrained_path": "/tmp/some_pretrained",
    }
    resolved = _resolve_codec_build_cfg(
        cfg, dataset_action_dim=7, is_main_process=False
    )
    assert resolved["action_dim"] == 8


def test_resolve_codec_build_cfg_sets_action_dim_when_null_for_fresh_run() -> None:
    cfg = {
        "name": "fast",
        "kind": "tokens",
        "action_dim": None,
        "pretrained_path": None,
    }
    resolved = _resolve_codec_build_cfg(
        cfg, dataset_action_dim=7, is_main_process=False
    )
    assert resolved["action_dim"] == 7


def test_resolve_codec_build_cfg_keeps_null_action_dim_for_pretrained() -> None:
    cfg = {
        "name": "fast",
        "kind": "tokens",
        "action_dim": None,
        "pretrained_path": "/tmp/some_pretrained",
    }
    resolved = _resolve_codec_build_cfg(
        cfg, dataset_action_dim=7, is_main_process=False
    )
    assert resolved["action_dim"] is None


def test_extract_codec_action_dim_prefers_top_level_attr() -> None:
    codec = _DummyCodecWithAttr(action_dim=7)
    assert _extract_codec_action_dim(codec) == 7


def test_extract_codec_action_dim_falls_back_to_config() -> None:
    codec = _DummyCodecWithConfig(action_dim=7)
    assert _extract_codec_action_dim(codec) == 7


def test_supports_train_loop_true_for_trainable_with_params() -> None:
    assert _supports_train_loop(nn.Module(), [{"params": [1]}]) is True


def test_supports_train_loop_false_for_non_trainable() -> None:
    assert _supports_train_loop(nn.Module(), []) is False


def test_should_skip_codec_fit_defaults_false() -> None:
    class _TrainerCfg:
        skip_codec_fit = False

    class _Cfg:
        trainer = _TrainerCfg()

    assert _should_skip_codec_fit(_Cfg()) is False


def test_should_skip_codec_fit_true_when_requested() -> None:
    class _TrainerCfg:
        skip_codec_fit = True

    class _Cfg:
        trainer = _TrainerCfg()

    assert _should_skip_codec_fit(_Cfg()) is True


def test_codecs_trainer_rejects_fsdp_strategy() -> None:
    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.cfg = OmegaConf.create({"trainer": {"distributed_strategy": "fsdp"}})
    trainer.tc = trainer.cfg.trainer

    with pytest.raises(ValueError, match="Codec training does not support"):
        trainer._validate_distributed_config()


def test_codec_reconstruct_skips_distributed_padding_batches(monkeypatch) -> None:
    class _PaddingBatchSampler:
        @staticmethod
        def is_padding_batch(batch_index: int) -> bool:
            return batch_index == 1

    class _ValDataloader:
        batch_sampler = _PaddingBatchSampler()

        @staticmethod
        def __len__() -> int:
            return 2

        @staticmethod
        def __iter__():
            yield {"action": torch.zeros(1, 2)}
            yield {"action": torch.ones(1, 2)}

    class _Collector:
        instances: list[_Collector] = []

        def __init__(self, *, device) -> None:
            del device
            self.contexts = []
            self.instances.append(self)

        def update(self, context) -> None:
            self.contexts.append(context)

        def reduce(self, runtime) -> None:
            del runtime

        def finalize(self) -> dict[str, float]:
            return {"val_reconst_loss": float(len(self.contexts))}

    monkeypatch.setattr(
        "praxis.training.codecs_trainer.ReconstLossCollector",
        _Collector,
    )
    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    codec = _ReconstructCodec()
    trainer.run_train_loop = True
    trainer.val_dataloader = _ValDataloader()
    trainer.max_reconst_batches = None
    trainer.action_key = "action"
    trainer.device = torch.device("cpu")
    trainer.runtime = _BootstrapRuntime(is_main_process=True)
    trainer.model = codec
    trainer._eval_progress = lambda iterable, **kwargs: iterable  # type: ignore[method-assign]
    trainer._move_batch_tensors_to_device = (  # type: ignore[method-assign]
        lambda batch, *, phase: batch
    )

    @contextlib.contextmanager
    def _ema_scope():
        yield codec

    trainer._ema_eval_scope = _ema_scope  # type: ignore[method-assign]

    assert trainer.reconstruct() == {"val_reconst_loss": 1.0}
    assert len(_Collector.instances[0].contexts) == 1


def test_codecs_trainer_get_optim_params_uses_unwrapped_model() -> None:
    class _Wrapped:
        pass

    class _Runtime:
        def __init__(self, unwrapped) -> None:
            self.unwrapped = unwrapped
            self.seen = None

        def unwrap_model(self, model):
            self.seen = model
            return self.unwrapped

    class _Model:
        def __init__(self) -> None:
            self.groups = [{"params": [torch.nn.Parameter(torch.ones(1))]}]

        def get_optim_params(self):
            return self.groups

    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    wrapped = _Wrapped()
    unwrapped = _Model()
    trainer.model = wrapped
    trainer.runtime = _Runtime(unwrapped)

    assert trainer._get_optim_params() is unwrapped.groups
    assert trainer.runtime.seen is wrapped


def test_codec_bootstrap_runs_fit_only_on_main_rank() -> None:
    main_trainer = _make_bootstrap_trainer(is_main_process=True)
    worker_trainer = _make_bootstrap_trainer(is_main_process=False)

    main_trainer._run_without_training_loop()
    worker_trainer._run_without_training_loop()

    assert len(main_trainer.model.fit_calls) == 1
    torch.testing.assert_close(
        main_trainer.model.fit_calls[0],
        torch.cat(
            [
                torch.ones(2, 4, dtype=torch.float32),
                torch.full((1, 4), 2.0, dtype=torch.float32),
            ],
            dim=0,
        ),
    )
    assert worker_trainer.model.fit_calls == []
    assert main_trainer.saved == 1
    assert worker_trainer.saved == 1
    assert main_trainer.ended == 1
    assert worker_trainer.ended == 1
    assert main_trainer.logger.records == [({"codec/bootstrap_complete": 1.0}, 0)]
    assert worker_trainer.logger.records == []


def test_codec_bootstrap_fit_uses_unprepared_fit_loader() -> None:
    trainer = _make_bootstrap_trainer(is_main_process=True)
    trainer.dataloader = [{"action": torch.full((1, 4), 9.0, dtype=torch.float32)}]
    trainer.fit_dataloader = [
        {"action": torch.ones(2, 4, dtype=torch.float32)},
        {"action": torch.full((2, 4), 2.0, dtype=torch.float32)},
    ]

    trainer._run_without_training_loop()

    assert len(trainer.model.fit_calls) == 1
    torch.testing.assert_close(
        trainer.model.fit_calls[0],
        torch.cat(
            [
                torch.ones(2, 4, dtype=torch.float32),
                torch.full((1, 4), 2.0, dtype=torch.float32),
            ],
            dim=0,
        ),
    )


def test_codec_bootstrap_skip_fit_still_saves_and_ends() -> None:
    trainer = _make_bootstrap_trainer(is_main_process=True, skip_fit=True)

    trainer._run_without_training_loop()

    assert trainer.model.fit_calls == []
    assert trainer.saved == 1
    assert trainer.ended == 1
    assert trainer.logger.records == [({"codec/bootstrap_complete": 1.0}, 0)]
