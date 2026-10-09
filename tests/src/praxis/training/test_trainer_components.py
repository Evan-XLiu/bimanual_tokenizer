# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from huggingface_hub import save_torch_model
from omegaconf import OmegaConf

from praxis.training.core.checkpoint import (
    RecentCheckpointManager,
    TopKCheckpointManager,
)
from praxis.training.core.checkpoint.pretrained import (
    filter_checkpoint_incompatible_keys,
    load_pretrained_model_weights,
)
from praxis.training.core.checkpoint_boundary import CheckpointBoundary
from praxis.training.core.ema import EMAModel
from praxis.training.core.metrics import MetricsTracker
from praxis.training.core.rank_errors import rank0_error_scope
from praxis.training.core.run_context import TrainerRunContext
from praxis.training.core.step_logging import (
    accumulate_train_info_metrics,
    reduce_train_info_metrics,
)
from praxis.training.core.trainer import DefaultTrainer
from praxis.training.core.trainer_components import (
    CheckpointService,
    EvalHooksService,
    TrainLoopEngine,
)


class _MetricPayloadError(RuntimeError):
    pass


def _no_eval_hooks() -> list[tuple[int, Callable[[], dict[str, float]], bool]]:
    return []


def _empty_eval_metrics() -> dict[str, float]:
    return {}


class _StubRuntime:
    def __init__(self, gathered, *, is_main_process: bool = True) -> None:
        self._gathered = gathered
        self.is_main_process = is_main_process
        self.num_processes = 1
        self.end_training_calls = 0
        self.wait_calls = 0
        self.gather_inputs: list[Any] = []
        self.broadcast_inputs: list[tuple[Any, int]] = []
        self.broadcast_result: Any = None
        self.broadcast_results: list[Any] = []

    def gather_for_metrics(self, input_data, use_gather_object=False):
        assert use_gather_object is True
        self.gather_inputs.append(input_data)
        return self._gathered if self._gathered is not None else input_data

    def reduce(self, tensor, reduction="sum"):
        assert reduction == "sum"
        return tensor

    def gather_object(self, obj: Any) -> list[Any]:
        return [obj]

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def end_training(self) -> None:
        self.end_training_calls += 1

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        self.broadcast_inputs.append((obj, src))
        if self.broadcast_results:
            return self.broadcast_results.pop(0)
        if self.broadcast_result is not None and obj is not None:
            return self.broadcast_result
        return obj

    def unwrap_model(self, model):
        return model


class _StubLogger:
    def __init__(self) -> None:
        self.records: list[tuple[dict[str, float], int]] = []
        self.finished = False

    def log(self, payload: dict[str, float], *, step: int) -> None:
        self.records.append((payload, step))

    def finish(self) -> None:
        self.finished = True


class _StubTrainMetrics:
    def __init__(self) -> None:
        self.samples = 0
        self.steps = 0
        self.values: dict[str, float] = {}

    def step(self, samples: int | None = None) -> None:
        self.steps += 1
        if samples is not None:
            self.samples += int(samples)

    def update(self, key: str, value: float) -> None:
        self.values[key] = value


class _StubModel:
    def train(self) -> None:
        return None


class _ContextAwareModel:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    def set_training_context(self, context: Any) -> None:
        self.contexts.append(context)


def test_default_trainer_publishes_training_context_to_opt_in_model() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    trainer.runtime = _StubRuntime(None)
    trainer.model = _ContextAwareModel()
    trainer.global_step = 42

    trainer._publish_training_context_to_model()

    assert trainer.model.contexts[-1].global_step == 42


def _state_dict_for_save_pretrained(
    model: torch.nn.Module,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    state_dict = kwargs.pop("state_dict", None)
    kwargs.pop("config", None)
    assert kwargs == {}
    return dict(state_dict) if state_dict is not None else dict(model.state_dict())


def _attach_run_context(stub: Any, tmp_path: Path) -> None:
    stub.run_context = TrainerRunContext(
        output_dir=tmp_path / "run",
        resume_checkpoint=None,
    )


def _stub_latest_checkpoint_save(
    stub: Any,
    *,
    on_save: Callable[[], None] | None = None,
) -> Callable[[], None]:
    def _save_latest_checkpoint() -> None:
        (Path(str(stub.tc.checkpoint_dir)) / "latest").mkdir(
            parents=True,
            exist_ok=True,
        )
        if on_save is not None:
            on_save()

    return _save_latest_checkpoint


class _CheckpointFilterModel:
    def filter_checkpoint_incompatible_keys(
        self,
        *,
        missing: list[str] | set[str] | tuple[str, ...],
        unexpected: list[str] | set[str] | tuple[str, ...],
    ) -> tuple[list[str], list[str]]:
        trainable = {"vlm.adapter.weight"}
        filtered_missing = [
            key
            for key in missing
            if not key.startswith("action_tokenizer.")
            and not (key.startswith("vlm.") and key not in trainable)
        ]
        filtered_unexpected = [
            key
            for key in unexpected
            if not key.startswith("action_tokenizer.")
            and not (key.startswith("vlm.") and key not in trainable)
        ]
        return filtered_missing, filtered_unexpected


def test_fsdp_distributed_config_allows_policy_lora() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create(
        {"trainer": {"distributed_strategy": "fsdp"}, "policy": {"use_lora": True}}
    )
    stub.tc = stub.cfg.trainer

    trainer._validate_distributed_config()


def test_fsdp_cpu_offload_disables_fused_adamw() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.device = torch.device("cuda", 0)
    stub.tc = OmegaConf.create(
        {
            "distributed_strategy": "fsdp",
            "fsdp_cpu_offload": True,
            "lr": 1e-3,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }
    )
    param = torch.nn.Parameter(torch.ones(1))

    optimizer = trainer._build_optimizer([{"params": [param]}])

    assert optimizer.defaults["fused"] is False


def test_build_optimizer_applies_trainer_family_lrs() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.device = torch.device("cpu")
    stub.tc = OmegaConf.create(
        {
            "distributed_strategy": "ddp",
            "lrs": {"vlm": 1e-5, "action_expert": 5e-5},
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }
    )
    vlm = torch.nn.Parameter(torch.ones(1))
    expert = torch.nn.Parameter(torch.ones(1))

    optimizer = trainer._build_optimizer(
        [
            {"params": [vlm], "family": "vlm"},
            {"params": [expert], "family": "action_expert", "weight_decay": 0.0},
        ]
    )

    lrs_by_param = {
        id(param): group["lr"]
        for group in optimizer.param_groups
        for param in group["params"]
    }
    weight_decay_by_param = {
        id(param): group["weight_decay"]
        for group in optimizer.param_groups
        for param in group["params"]
    }
    assert lrs_by_param[id(vlm)] == pytest.approx(1e-5)
    assert lrs_by_param[id(expert)] == pytest.approx(5e-5)
    assert weight_decay_by_param[id(vlm)] == pytest.approx(0.01)
    assert weight_decay_by_param[id(expert)] == pytest.approx(0.0)
    assert all("family" not in group for group in optimizer.param_groups)


def test_build_optimizer_rejects_unconfigured_family_lr() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.device = torch.device("cpu")
    stub.tc = OmegaConf.create(
        {
            "distributed_strategy": "ddp",
            "lrs": {"vlm": 1e-5},
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }
    )
    param = torch.nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="trainer.lrs.action_expert"):
        trainer._build_optimizer([{"params": [param], "family": "action_expert"}])


def test_build_optimizer_rejects_policy_family_without_trainer_lrs() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.device = torch.device("cpu")
    stub.tc = OmegaConf.create(
        {
            "distributed_strategy": "ddp",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }
    )
    param = torch.nn.Parameter(torch.ones(1))

    with pytest.raises(ValueError, match="trainer.lrs is not configured"):
        trainer._build_optimizer([{"params": [param], "family": "vlm"}])


def test_ddp_non_main_enables_checkpoint_managers_for_status_broadcasts() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None, is_main_process=False)
    stub.tc = OmegaConf.create({"distributed_strategy": "ddp"})

    assert trainer._enable_checkpoint_managers()


def test_filter_checkpoint_incompatible_keys_ignores_expected_frozen_submodules() -> (
    None
):
    model = _CheckpointFilterModel()
    missing, unexpected = filter_checkpoint_incompatible_keys(
        model,
        missing=[
            "action_tokenizer.decoder.weight",
            "vlm.backbone.weight",
            "vlm.adapter.weight",
            "head.weight",
        ],
        unexpected=[
            "action_tokenizer.extra",
            "vlm.backbone.extra",
            "head.extra",
        ],
    )

    assert missing == ["vlm.adapter.weight", "head.weight"]
    assert unexpected == ["head.extra"]


def test_filter_checkpoint_incompatible_keys_rejects_non_string_keys() -> None:
    class _BadFilterModel(torch.nn.Module):
        @staticmethod
        def filter_checkpoint_incompatible_keys(*, missing, unexpected):
            del missing, unexpected
            return [123], []

    with pytest.raises(TypeError, match="filtered missing keys"):
        filter_checkpoint_incompatible_keys(
            _BadFilterModel(),
            missing=["weight"],
            unexpected=[],
        )


def test_accumulate_train_info_metrics_tracks_weighted_sums() -> None:
    metric_sums: dict[str, float] = {}
    metric_weights: dict[str, float] = {}

    accumulate_train_info_metrics(
        metric_sums,
        metric_weights,
        {"boat_primary_loss": 2.0, "boat_refine_aux_loss": 1.0},
        weight=4,
    )
    accumulate_train_info_metrics(
        metric_sums,
        metric_weights,
        {"boat_primary_loss": 1.0},
        weight=2,
    )

    assert metric_sums == {
        "boat_primary_loss": 10.0,
        "boat_refine_aux_loss": 4.0,
    }
    assert metric_weights == {
        "boat_primary_loss": 6.0,
        "boat_refine_aux_loss": 4.0,
    }


def test_reduce_train_info_metrics_combines_rank_payloads() -> None:
    runtime = _StubRuntime(
        [
            {
                "metric_sums": {"boat_primary_loss": 8.0, "boat_refine_aux_loss": 2.0},
                "metric_weights": {
                    "boat_primary_loss": 4.0,
                    "boat_refine_aux_loss": 4.0,
                },
            },
            {
                "metric_sums": {"boat_primary_loss": 6.0},
                "metric_weights": {"boat_primary_loss": 3.0},
            },
        ]
    )

    reduced = reduce_train_info_metrics(
        runtime,
        {"boat_primary_loss": 8.0, "boat_refine_aux_loss": 2.0},
        {"boat_primary_loss": 4.0, "boat_refine_aux_loss": 4.0},
    )

    assert reduced == {
        "boat_primary_loss": 2.0,
        "boat_refine_aux_loss": 0.5,
    }
    assert runtime.gather_inputs == [
        [
            {
                "metric_sums": {
                    "boat_primary_loss": 8.0,
                    "boat_refine_aux_loss": 2.0,
                },
                "metric_weights": {
                    "boat_primary_loss": 4.0,
                    "boat_refine_aux_loss": 4.0,
                },
            }
        ]
    ]


def test_reduce_train_info_metrics_handles_runtime_object_gather_shape() -> None:
    """Regression: distributed object gather flattens one list nesting level."""
    runtime = _StubRuntime(None)

    reduced = reduce_train_info_metrics(
        runtime,
        {"L_stop": 3.0, "pred_k_mean": 12.0},
        {"L_stop": 2.0, "pred_k_mean": 4.0},
    )

    assert reduced == {
        "L_stop": pytest.approx(1.5),
        "pred_k_mean": pytest.approx(3.0),
    }


def test_reduce_train_info_metrics_gathers_even_when_local_metrics_empty() -> None:
    runtime = _StubRuntime(
        [
            {"metric_sums": {}, "metric_weights": {}},
            {
                "metric_sums": {"rank_local_debug": 2.0},
                "metric_weights": {"rank_local_debug": 1.0},
            },
        ]
    )

    reduced = reduce_train_info_metrics(runtime, {}, {})

    assert reduced == {"rank_local_debug": pytest.approx(2.0)}
    assert runtime.gather_inputs == [[{"metric_sums": {}, "metric_weights": {}}]]


def test_load_pretrained_model_weights_rejects_stale_incomplete_index(
    tmp_path: Path,
) -> None:
    (tmp_path / "model.safetensors").write_text("single-file\n")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "model-00001-of-00002.safetensors",
                    "bias": "model-00002-of-00002.safetensors",
                }
            }
        )
        + "\n"
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_text("weight\n")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_pretrained_model_weights(torch.nn.Linear(1, 1), tmp_path)


def test_load_pretrained_model_weights_accepts_sharded_tied_weight_alias(
    tmp_path: Path,
) -> None:
    class _TiedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(512, 16)
            self.lm_head = torch.nn.Linear(16, 512, bias=False)
            self.lm_head.weight = self.embed.weight
            self.other = torch.nn.Parameter(torch.zeros(2048))

    source = _TiedModel()
    with torch.no_grad():
        source.embed.weight.copy_(
            torch.arange(source.embed.weight.numel(), dtype=torch.float32).reshape_as(
                source.embed.weight
            )
        )
        source.other.fill_(3.0)
    save_torch_model(source, tmp_path, max_shard_size="1KB")
    assert (tmp_path / "model.safetensors.index.json").exists()

    target = _TiedModel()
    missing, unexpected = load_pretrained_model_weights(
        target,
        tmp_path,
        strict=False,
    )

    assert missing == []
    assert unexpected == []
    torch.testing.assert_close(target.embed.weight, source.embed.weight)
    torch.testing.assert_close(target.lm_head.weight, source.lm_head.weight)
    torch.testing.assert_close(target.other, source.other)


def test_ema_eval_scope_restores_when_apply_fails() -> None:
    class _FailingEMA:
        def __init__(self, model: torch.nn.Linear) -> None:
            self.backup: dict[str, torch.Tensor] = {}
            self.model = model

        @property
        def restore_needed(self) -> bool:
            return bool(self.backup)

        def apply(self, model):
            assert model is self.model
            self.backup = {"weight": model.weight.data.clone()}
            model.weight.data.fill_(123.0)
            raise RuntimeError("ema eval apply failed")

        def restore(self, model):
            model.weight.data.copy_(self.backup["weight"])
            self.backup.clear()

    model = torch.nn.Linear(1, 1)
    model.weight.data.fill_(1.0)
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.model = model
    stub.ema = _FailingEMA(model)

    with (
        pytest.raises(RuntimeError, match="EMA eval apply failed on rank 0"),
        EvalHooksService(stub).ema_eval_scope(),
    ):
        raise AssertionError("scope body should not run")

    assert model.weight.item() == pytest.approx(1.0)


def test_train_loop_broadcasts_rank0_metric_payload_errors() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.runtime.broadcast_result = "Train metric payload failed on rank 0"

    with (
        pytest.raises(RuntimeError, match="Train metric payload failed on rank 0"),
        rank0_error_scope(stub.runtime, "Train metric payload"),
    ):
        raise _MetricPayloadError("collision")

    assert stub.runtime.broadcast_inputs == [
        (
            "Train metric payload failed on rank 0: _MetricPayloadError: collision",
            0,
        )
    ]


def test_fsdp_checkpoint_save_uses_broadcast_staging_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = _StubRuntime(None)
    runtime.broadcast_result = "shared-token"
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = runtime
    stub.model = model
    stub.optimizer = optimizer
    stub.scheduler = None
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    saved_checkpoint_ids: list[Path] = []

    def _fake_get_state_dict(*args, **kwargs):
        del args, kwargs
        return {"weight": torch.ones(1)}, {"state": {}}

    def _fake_dcp_save(state, *, checkpoint_id, no_dist):
        del state, no_dist
        path = Path(checkpoint_id)
        saved_checkpoint_ids.append(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "rank.marker").write_text("ok\n")

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_state_dict",
        _fake_get_state_dict,
    )
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict_saver.save",
        _fake_dcp_save,
    )

    checkpoint_dir = tmp_path / "latest"
    CheckpointService(stub).save_checkpoint_dir(
        checkpoint_dir,
        include_training_state=True,
    )

    token, src = next(
        (payload, src)
        for payload, src in runtime.broadcast_inputs
        if isinstance(payload, str)
    )
    assert isinstance(token, str)
    assert src == 0
    assert saved_checkpoint_ids == [tmp_path / ".latest.tmp-shared-token" / "dcp"]
    assert (checkpoint_dir / "dcp" / "rank.marker").exists()
    assert (checkpoint_dir / "training_state" / "trainer.pt").exists()


def test_fsdp_checkpoint_save_includes_ema_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = _StubRuntime(None)
    runtime.broadcast_result = "shared-token"
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = runtime
    stub.model = model
    stub.optimizer = optimizer
    stub.scheduler = None
    stub.ema = EMAModel(model, decay=0.5)
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-ema-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    saved_state_keys: list[set[str]] = []

    def _fake_get_state_dict(*args, **kwargs):
        del args, kwargs
        return {"weight": torch.ones(1)}, {"state": {}}

    def _fake_dcp_save(state, *, checkpoint_id, no_dist):
        del no_dist
        saved_state_keys.append(set(state))
        path = Path(checkpoint_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "rank.marker").write_text("ok\n")

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_state_dict",
        _fake_get_state_dict,
    )
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict_saver.save",
        _fake_dcp_save,
    )

    checkpoint_dir = tmp_path / "latest"
    CheckpointService(stub).save_checkpoint_dir(
        checkpoint_dir,
        include_training_state=True,
    )

    assert saved_state_keys == [{"model", "optimizer", "ema"}]


def test_fsdp_checkpoint_save_collects_rank_local_state_dict_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime = _StubRuntime(None)
    runtime.broadcast_result = "shared-token"
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = runtime
    stub.model = model
    stub.optimizer = optimizer
    stub.scheduler = None
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-state-error-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    def _fail_get_state_dict(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("state failed")

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_state_dict",
        _fail_get_state_dict,
    )

    with pytest.raises(RuntimeError, match="FSDP DCP state dict failed on rank 0"):
        CheckpointService(stub).save_checkpoint_dir(
            tmp_path / "latest",
            include_training_state=True,
        )

    assert not (tmp_path / "latest").exists()


def test_fsdp_checkpoint_load_collects_rank_local_state_dict_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_dir = tmp_path / "latest"
    (checkpoint_dir / "dcp").mkdir(parents=True)
    (checkpoint_dir / "training_state").mkdir()
    torch.save(
        {"train_step": 7, "scheduler": None},
        checkpoint_dir / "training_state" / "trainer.pt",
    )

    runtime = _StubRuntime(None)
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = runtime
    stub.model = model
    stub.optimizer = optimizer
    stub.scheduler = None
    stub.global_step = 0
    stub.cfg = OmegaConf.create({"name": "fsdp-load-state-error-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    def _fail_get_state_dict(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("load state failed")

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_state_dict",
        _fail_get_state_dict,
    )

    with pytest.raises(
        RuntimeError,
        match="FSDP DCP load state dict setup failed on rank 0",
    ):
        CheckpointService(stub).load_checkpoint(checkpoint_dir)


def test_ddp_checkpoint_load_requires_ema_state_when_training_state_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_dir = tmp_path / "latest"
    (checkpoint_dir / "pretrained_model").mkdir(parents=True)
    (checkpoint_dir / "training_state").mkdir()

    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.model = model
    stub.optimizer = optimizer
    stub.scheduler = None
    stub.ema = EMAModel(model, decay=0.5)
    stub.global_step = 0
    stub.cfg = OmegaConf.create({"name": "ddp-missing-ema-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "ddp"})

    monkeypatch.setattr(
        "praxis.training.core.checkpoint.io.load_pretrained_model_weights",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(
        "praxis.training.core.checkpoint.io.load_training_state",
        lambda *, checkpoint_dir, optimizer, scheduler: (5, optimizer, scheduler),
    )

    with pytest.raises(RuntimeError, match="missing EMA state"):
        CheckpointService(stub).load_checkpoint(checkpoint_dir)


def test_fsdp_candidate_checkpoint_exports_pretrained_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExportModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.saved_state: dict[str, torch.Tensor] | None = None

        def save_pretrained(self, save_directory, **kwargs):
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self.saved_state = {
                key: value.clone()
                for key, value in _state_dict_for_save_pretrained(
                    self,
                    kwargs,
                ).items()
            }
            torch.save(self.saved_state, save_directory / "state.pt")

    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.model = _ExportModel()
    stub.optimizer = None
    stub.scheduler = None
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-candidate-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    def _fake_get_model_state_dict(*args, **kwargs):
        del args, kwargs
        return {"weight": torch.ones(1)}

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        _fake_get_model_state_dict,
    )

    checkpoint_dir = tmp_path / "best" / "step_7"
    CheckpointService(stub).save_checkpoint_dir(
        checkpoint_dir,
        include_training_state=False,
    )

    saved_state = torch.load(
        checkpoint_dir / "pretrained_model" / "state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved_state["weight"].item() == pytest.approx(1.0)
    assert not (checkpoint_dir / "dcp").exists()
    assert not (checkpoint_dir / "training_state").exists()
    assert (checkpoint_dir / "pretrained_model" / "train_config.yaml").exists()


def test_fsdp_candidate_checkpoint_applies_model_export_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FilteredExportModel(torch.nn.Module):
        @staticmethod
        def filter_checkpoint_export_state_dict(state_dict: dict) -> dict:
            trainable = {"vlm.adapter.weight"}
            return {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("action_tokenizer.")
                and not (key.startswith("vlm.") and key not in trainable)
            }

        def save_pretrained(self, save_directory, **kwargs):
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                _state_dict_for_save_pretrained(self, kwargs),
                save_directory / "state.pt",
            )

    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.model = _FilteredExportModel()
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-filtered-export-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})
    captured_ignore_frozen: list[bool] = []

    def _fake_get_model_state_dict(*args, **kwargs):
        del args
        captured_ignore_frozen.append(bool(kwargs["options"].ignore_frozen_params))
        return {
            "vlm.frozen.weight": torch.ones(1),
            "vlm.adapter.weight": torch.full((1,), 2.0),
            "action_tokenizer.weight": torch.full((1,), 3.0),
            "head.weight": torch.full((1,), 4.0),
        }

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        _fake_get_model_state_dict,
    )

    checkpoint_dir = tmp_path / "best" / "step_7"
    CheckpointService(stub).save_checkpoint_dir(
        checkpoint_dir,
        include_training_state=False,
    )

    saved_state = torch.load(
        checkpoint_dir / "pretrained_model" / "state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(saved_state) == {"vlm.adapter.weight", "head.weight"}
    assert captured_ignore_frozen == [False]


def test_fsdp_export_tokenizer_only_filtering_keeps_other_frozen_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TokenizerFilteredExportModel(torch.nn.Module):
        @staticmethod
        def filter_checkpoint_export_state_dict(state_dict: dict) -> dict:
            return {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("action_tokenizer.")
            }

        def save_pretrained(self, save_directory, **kwargs):
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                _state_dict_for_save_pretrained(self, kwargs),
                save_directory / "state.pt",
            )

    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.model = _TokenizerFilteredExportModel()
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-tokenizer-filtered-export-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})
    captured_ignore_frozen: list[bool] = []

    def _fake_get_model_state_dict(*args, **kwargs):
        del args
        captured_ignore_frozen.append(bool(kwargs["options"].ignore_frozen_params))
        return {
            "action_tokenizer.weight": torch.full((1,), 3.0),
            "vlm.vision_tower.frozen.weight": torch.full((1,), 5.0),
            "head.weight": torch.full((1,), 4.0),
        }

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        _fake_get_model_state_dict,
    )

    checkpoint_dir = tmp_path / "best" / "step_7"
    CheckpointService(stub).save_checkpoint_dir(
        checkpoint_dir,
        include_training_state=False,
    )

    saved_state = torch.load(
        checkpoint_dir / "pretrained_model" / "state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(saved_state) == {"vlm.vision_tower.frozen.weight", "head.weight"}
    assert captured_ignore_frozen == [False]


def test_fsdp_candidate_checkpoint_broadcasts_rank0_export_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingExportModel(torch.nn.Module):
        def save_pretrained(self, save_directory, **kwargs):
            del save_directory, kwargs
            raise OSError("disk full")

    runtime = _StubRuntime(None)
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = runtime
    stub.model = _FailingExportModel()
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-export-error-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    def _fake_get_model_state_dict(*args, **kwargs):
        del args, kwargs
        return {}

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        _fake_get_model_state_dict,
    )

    with pytest.raises(RuntimeError, match="pretrained_model export failed on rank 0"):
        CheckpointService(stub).save_checkpoint_dir(
            tmp_path / "best" / "step_7",
            include_training_state=False,
        )

    assert any(
        isinstance(payload, str) and "disk full" in payload
        for payload, _src in runtime.broadcast_inputs
    )


def test_fsdp_candidate_checkpoint_can_export_ema_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExportModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        def save_pretrained(self, save_directory, **kwargs):
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            torch.save(
                _state_dict_for_save_pretrained(self, kwargs),
                save_directory / "state.pt",
            )

    model = _ExportModel()
    ema = EMAModel(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)

    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.model = model
    stub.optimizer = None
    stub.scheduler = None
    stub.ema = ema
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-candidate-ema-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    def _fake_get_model_state_dict(model_arg, *args, **kwargs):
        del args, kwargs
        return {"weight": model_arg.weight.detach().clone()}

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        _fake_get_model_state_dict,
    )

    checkpoint_dir = tmp_path / "best" / "step_7"
    CheckpointService(stub).save_checkpoint_dir(
        checkpoint_dir,
        apply_ema_weights=True,
        include_training_state=False,
    )

    saved_state = torch.load(
        checkpoint_dir / "pretrained_model" / "state.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved_state["weight"].item() == pytest.approx(2.0)
    assert model.weight.item() == pytest.approx(3.0)


def test_fsdp_candidate_checkpoint_restores_ema_when_apply_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExportModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def save_pretrained(self, save_directory, **kwargs):
            del save_directory, kwargs
            raise AssertionError("export should stop before save_pretrained")

    class _FailingEMA:
        def __init__(self, model: _ExportModel) -> None:
            self.backup: dict[str, torch.Tensor] = {}
            self.model = model

        @property
        def restore_needed(self) -> bool:
            return bool(self.backup)

        def refresh(self, model):
            assert model is self.model

        def apply(self, model):
            self.backup = {"weight": model.weight.data.clone()}
            model.weight.data.fill_(123.0)
            raise RuntimeError("ema apply failed")

        def restore(self, model):
            model.weight.data.copy_(self.backup["weight"])
            self.backup.clear()

    model = _ExportModel()
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _StubRuntime(None)
    stub.model = model
    stub.ema = _FailingEMA(model)
    stub.global_step = 7
    stub.cfg = OmegaConf.create({"name": "fsdp-ema-apply-error-test"})
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})

    def _unexpected_get_model_state_dict(*args, **kwargs):
        del args, kwargs
        raise AssertionError("state dict should not be gathered after EMA failure")

    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        _unexpected_get_model_state_dict,
    )

    with pytest.raises(RuntimeError, match="FSDP pretrained_model EMA apply"):
        CheckpointService(stub).save_checkpoint_dir(
            tmp_path / "best" / "step_7",
            apply_ema_weights=True,
            include_training_state=False,
        )

    assert model.weight.item() == pytest.approx(0.0)


def test_fsdp_candidate_pools_save_without_ema_or_training_state(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)
    stub.runtime = runtime
    stub.global_step = 5
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})
    stub.topk_manager = TopKCheckpointManager(
        k=1,
        mode="min",
        metric_key="val_loss",
        directory=tmp_path / "best",
        manage_files=False,
    )
    stub.recent_manager = RecentCheckpointManager(
        k=1,
        directory=tmp_path / "recent",
        manage_files=False,
    )
    calls: list[tuple[Path, dict[str, Any]]] = []
    trainer._save_checkpoint_dir = (  # type: ignore[method-assign]
        lambda path, **kwargs: calls.append((Path(path), dict(kwargs)))
    )
    runtime.broadcast_results = [
        {"path": str(tmp_path / "best" / "step_5"), "value": 0.25},
        None,
    ]

    service = CheckpointService(stub)
    service.update_best_checkpoint_pool({"val_loss": 0.25})
    service.update_recent_checkpoint_pool()

    assert calls == [
        (
            tmp_path / "best" / "step_5",
            {"apply_ema_weights": False, "include_training_state": False},
        ),
        (
            tmp_path / "recent" / "step_5",
            {"apply_ema_weights": False, "include_training_state": False},
        ),
    ]


def test_fsdp_candidate_pools_save_with_ema_applied(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)
    stub.runtime = runtime
    stub.global_step = 5
    stub.ema = object()
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})
    stub.topk_manager = TopKCheckpointManager(
        k=1,
        mode="min",
        metric_key="val_loss",
        directory=tmp_path / "best",
        manage_files=False,
    )
    stub.recent_manager = RecentCheckpointManager(
        k=1,
        directory=tmp_path / "recent",
        manage_files=False,
    )
    calls: list[tuple[Path, dict[str, Any]]] = []
    trainer._save_checkpoint_dir = (  # type: ignore[method-assign]
        lambda path, **kwargs: calls.append((Path(path), dict(kwargs)))
    )
    runtime.broadcast_results = [
        {"path": str(tmp_path / "best" / "step_5"), "value": 0.25},
        None,
    ]

    service = CheckpointService(stub)
    service.update_best_checkpoint_pool({"val_loss": 0.25})
    service.update_recent_checkpoint_pool()

    assert calls == [
        (
            tmp_path / "best" / "step_5",
            {"apply_ema_weights": True, "include_training_state": False},
        ),
        (
            tmp_path / "recent" / "step_5",
            {"apply_ema_weights": True, "include_training_state": False},
        ),
    ]


def test_fsdp_best_checkpoint_pool_broadcasts_rank0_decision(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=True)
    stub.runtime = runtime
    stub.global_step = 5
    stub.ema = None
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})
    stub.topk_manager = TopKCheckpointManager(
        k=1,
        mode="min",
        metric_key="val_loss",
        directory=tmp_path / "best",
        manage_files=True,
    )
    calls: list[tuple[Path, dict[str, Any]]] = []
    trainer._save_checkpoint_dir = (  # type: ignore[method-assign]
        lambda path, **kwargs: calls.append((Path(path), dict(kwargs)))
    )

    CheckpointService(stub).update_best_checkpoint_pool({"val_loss": 0.25})

    assert calls == [
        (
            tmp_path / "best" / "step_5",
            {"apply_ema_weights": False, "include_training_state": False},
        )
    ]
    assert runtime.broadcast_inputs[0] == (
        {"path": str(tmp_path / "best" / "step_5"), "value": 0.25},
        0,
    )
    assert (tmp_path / "best" / "topk_manifest.json").exists()


def test_fsdp_best_checkpoint_pool_non_main_uses_broadcast_path(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)
    runtime.broadcast_results = [
        {"path": str(tmp_path / "best" / "step_5"), "value": 0.25},
        None,
    ]
    stub.runtime = runtime
    stub.global_step = 5
    stub.ema = None
    stub.tc = OmegaConf.create({"distributed_strategy": "fsdp"})
    stub.topk_manager = TopKCheckpointManager(
        k=1,
        mode="min",
        metric_key="val_loss",
        directory=tmp_path / "best",
        manage_files=False,
    )
    calls: list[tuple[Path, dict[str, Any]]] = []
    trainer._save_checkpoint_dir = (  # type: ignore[method-assign]
        lambda path, **kwargs: calls.append((Path(path), dict(kwargs)))
    )

    CheckpointService(stub).update_best_checkpoint_pool({"val_loss": 0.25})

    assert calls == [
        (
            tmp_path / "best" / "step_5",
            {"apply_ema_weights": False, "include_training_state": False},
        )
    ]
    assert runtime.broadcast_inputs == [(None, 0), (None, 0)]


def test_ddp_save_latest_checkpoint_broadcasts_rank0_error(tmp_path: Path) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=True)
    stub.runtime = runtime
    stub.tc = OmegaConf.create(
        {
            "distributed_strategy": "ddp",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }
    )

    def fail_save(_path):
        raise OSError("disk full")

    trainer._save_checkpoint_dir = fail_save  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="DDP latest checkpoint save"):
        CheckpointService(stub).save_latest_checkpoint()

    assert any(
        isinstance(payload, str) and "disk full" in payload
        for payload, _src in runtime.broadcast_inputs
    )


def test_ddp_save_latest_checkpoint_non_main_waits_for_rank0_status(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)
    runtime.broadcast_results = ["DDP latest checkpoint save failed on rank 0"]
    stub.runtime = runtime
    stub.tc = OmegaConf.create(
        {
            "distributed_strategy": "ddp",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }
    )
    trainer._save_checkpoint_dir = (  # type: ignore[method-assign]
        lambda _path: (_ for _ in ()).throw(AssertionError("non-main must not save"))
    )

    with pytest.raises(RuntimeError, match="DDP latest checkpoint save"):
        CheckpointService(stub).save_latest_checkpoint()

    assert runtime.broadcast_inputs == [(None, 0)]


def test_ddp_candidate_checkpoint_broadcasts_rank0_error(tmp_path: Path) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=True)
    stub.runtime = runtime
    stub.global_step = 5
    stub.ema = None
    stub.tc = OmegaConf.create({"distributed_strategy": "ddp"})
    stub.topk_manager = TopKCheckpointManager(
        k=1,
        mode="min",
        metric_key="val_loss",
        directory=tmp_path / "best",
        manage_files=True,
    )

    def fail_save(_path, **_kwargs):
        raise OSError("disk full")

    trainer._save_checkpoint_dir = fail_save  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="DDP best checkpoint update"):
        CheckpointService(stub).update_best_checkpoint_pool({"val_loss": 0.25})

    assert any(
        isinstance(payload, str) and "disk full" in payload
        for payload, _src in runtime.broadcast_inputs
    )


def test_ddp_candidate_pools_non_main_wait_for_rank0_status(tmp_path: Path) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)
    stub.runtime = runtime
    stub.global_step = 5
    stub.ema = None
    stub.tc = OmegaConf.create({"distributed_strategy": "ddp"})
    stub.topk_manager = TopKCheckpointManager(
        k=1,
        mode="min",
        metric_key="val_loss",
        directory=tmp_path / "best",
        manage_files=False,
    )
    stub.recent_manager = RecentCheckpointManager(
        k=1,
        directory=tmp_path / "recent",
        manage_files=False,
    )
    trainer._save_checkpoint_dir = (  # type: ignore[method-assign]
        lambda _path, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-main DDP candidate ranks must not save")
        )
    )

    service = CheckpointService(stub)
    service.update_best_checkpoint_pool({})
    service.update_recent_checkpoint_pool()

    assert runtime.broadcast_inputs == [(None, 0), (None, 0)]


def test_train_loop_engine_logs_weighted_model_metrics(tmp_path: Path) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    logger = _StubLogger()

    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 2,
            "log_every": 2,
            "checkpoint_every": None,
            "val_every": None,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [
        {"action": torch.zeros(2, 1)},
        {"action": torch.zeros(4, 1)},
    ]
    stub.global_step = 0
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = _no_eval_hooks  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._get_current_lr = lambda: 1e-3  # type: ignore[method-assign]

    outputs = iter(
        [
            (
                torch.tensor(2.0),
                {"boat_primary_loss": 1.0, "boat_refine_aux_loss": 0.5},
                True,
            ),
            (
                torch.tensor(4.0),
                {"boat_primary_loss": 3.0, "boat_refine_aux_loss": 1.5},
                True,
            ),
        ]
    )
    trainer.train_step = lambda batch: next(outputs)  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    assert logger.records
    payload, step = logger.records[0]
    assert step == 2
    assert payload["train/loss"] == pytest.approx(10.0 / 3.0)
    assert payload["train/boat_primary_loss"] == pytest.approx(7.0 / 3.0)
    assert payload["train/boat_refine_aux_loss"] == pytest.approx(7.0 / 6.0)


def test_train_loop_engine_counts_accumulated_samples_per_optimizer_step(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    logger = _StubLogger()

    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_every": None,
            "val_every": None,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [
        {"action": torch.zeros(8, 1)},
        {"action": torch.zeros(8, 1)},
    ]
    stub.global_step = 0
    stub.logger = logger
    # Use a deliberately wrong configured batch size. The loop should count
    # actual microbatch samples consumed before the optimizer step.
    stub.train_metrics = MetricsTracker(batch_size=999)
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = _no_eval_hooks  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._get_current_lr = lambda: 1e-3  # type: ignore[method-assign]

    outputs = iter(
        [
            (torch.tensor(1.0), {}, False),
            (torch.tensor(1.0), {}, True),
        ]
    )
    trainer.train_step = lambda batch: next(outputs)  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    payload, step = logger.records[0]
    assert step == 1
    assert payload["train/samples"] == pytest.approx(16.0)
    assert payload["train/steps"] == pytest.approx(1.0)


def test_train_loop_engine_namespaces_model_loss_metric_collision(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    logger = _StubLogger()

    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_every": None,
            "val_every": None,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = _no_eval_hooks  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._get_current_lr = lambda: 1e-3  # type: ignore[method-assign]
    trainer.train_step = (  # type: ignore[method-assign]
        lambda batch: (torch.tensor(2.0), {"loss": 0.25}, True)
    )

    TrainLoopEngine(trainer).run()

    payload, step = logger.records[0]
    assert step == 1
    assert payload["train/loss"] == pytest.approx(2.0)
    assert payload["train/model/loss"] == pytest.approx(0.25)


def test_train_loop_engine_logs_one_payload_for_same_step_train_and_eval(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    logger = _StubLogger()

    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 1,
            "checkpoint_every": None,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._get_current_lr = lambda: 1e-3  # type: ignore[method-assign]
    trainer.train_step = (  # type: ignore[method-assign]
        lambda batch: (torch.tensor(2.0), {"boat_primary_loss": 5.0}, True)
    )

    best_updates: list[dict[str, float]] = []
    trainer._periodic_eval_hooks = lambda: [  # type: ignore[method-assign]
        (1, lambda: {"val_loss": 3.0}, False),
        (
            1,
            lambda: {
                "val_reconst_loss": 0.25,
                "debug/eval/pred_k_mean": 8.0,
            },
            True,
        ),
    ]
    trainer._update_best_checkpoint_pool = (  # type: ignore[method-assign]
        lambda metrics: best_updates.append(dict(metrics))
    )

    TrainLoopEngine(trainer).run()

    assert len(logger.records) == 1
    payload, step = logger.records[0]
    assert step == 1
    assert payload == {
        "train/loss": pytest.approx(2.0),
        "train/lr": pytest.approx(1e-3),
        "train/samples": pytest.approx(2.0),
        "train/steps": pytest.approx(1.0),
        "train/boat_primary_loss": pytest.approx(5.0),
        "eval/loss": pytest.approx(3.0),
        "eval/reconst_loss": pytest.approx(0.25),
        "debug/eval/pred_k_mean": pytest.approx(8.0),
    }
    assert best_updates == [{"val_reconst_loss": 0.25, "debug/eval/pred_k_mean": 8.0}]


def test_train_loop_engine_syncs_after_main_only_eval_and_checkpoint_work(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None)
    logger = _StubLogger()

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": 1,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]

    best_updates: list[dict[str, float]] = []
    recent_updates: list[int] = []
    latest_saves: list[int] = []
    events: list[str] = []

    def _eval_metrics() -> dict[str, float]:
        events.append("eval")
        return {"val_loss": 2.0}

    trainer._periodic_eval_hooks = lambda: [  # type: ignore[method-assign]
        (1, _eval_metrics, True)
    ]
    trainer._update_best_checkpoint_pool = (  # type: ignore[method-assign]
        lambda metrics: (
            events.append("best"),
            best_updates.append(dict(metrics)),
        )
    )
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(  # type: ignore[method-assign]
        stub,
        on_save=lambda: (
            events.append("latest"),
            latest_saves.append(int(stub.global_step)),
        ),
    )
    trainer._update_recent_checkpoint_pool = (  # type: ignore[method-assign]
        lambda: (
            events.append("recent"),
            recent_updates.append(int(stub.global_step)),
        )
    )

    TrainLoopEngine(trainer).run()

    assert best_updates == [{"val_loss": 2.0}]
    assert latest_saves == [1, 1]
    assert recent_updates == [1, 1]
    assert events[:4] == ["latest", "eval", "best", "recent"]
    assert events[-2:] == ["latest", "recent"]
    assert runtime.wait_calls == 3


def test_train_loop_engine_saves_latest_before_checkpoint_step_eval_failure(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None)

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": 1,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = _StubLogger()
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]

    events: list[str] = []

    def _raise_eval() -> dict[str, float]:
        events.append("eval")
        raise RuntimeError("sim eval exploded")

    trainer._periodic_eval_hooks = lambda: [(1, _raise_eval, True)]  # type: ignore[method-assign]
    trainer._update_best_checkpoint_pool = (  # type: ignore[method-assign]
        lambda metrics: events.append("best")
    )
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(  # type: ignore[method-assign]
        stub,
        on_save=lambda: events.append(f"latest:{int(stub.global_step)}"),
    )
    trainer._update_recent_checkpoint_pool = (  # type: ignore[method-assign]
        lambda: events.append("recent")
    )

    with pytest.raises(RuntimeError, match="sim eval exploded"):
        TrainLoopEngine(trainer).run()

    assert events == ["latest:1", "eval"]
    assert runtime.wait_calls == 0


def test_train_loop_engine_marks_checkpoint_phase_after_eval(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None)
    logger = _StubLogger()
    latest_dir = tmp_path / "run" / "checkpoints" / "latest"
    latest_dir.mkdir(parents=True)

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": 1,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = lambda: [(1, lambda: {"val_loss": 2.0}, True)]  # type: ignore[method-assign]
    trainer._update_best_checkpoint_pool = lambda metrics: None  # type: ignore[method-assign]
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    stub.run_context = TrainerRunContext(
        output_dir=tmp_path / "run",
        resume_checkpoint=str(latest_dir),
    )
    assert not CheckpointBoundary(stub, checkpoint_every=1).should_replay_on_resume()


def test_train_loop_engine_replays_unfinished_resume_checkpoint_phase(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None)
    logger = _StubLogger()
    latest_dir = tmp_path / "run" / "checkpoints" / "latest"
    latest_dir.mkdir(parents=True)

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    stub.run_context = TrainerRunContext(
        output_dir=tmp_path / "run",
        resume_checkpoint=str(latest_dir),
    )
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": 1,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = []
    stub.global_step = 1
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    CheckpointBoundary(stub, checkpoint_every=1).write_step_log_payload(
        latest_dir,
        {
            "train/loss": 1.0,
            "train/lr": 1e-3,
        },
    )
    trainer.train_step = lambda batch: pytest.fail("train step should not run")  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]

    events: list[str] = []

    def _eval_metrics() -> dict[str, float]:
        events.append(f"eval:{int(stub.global_step)}")
        return {"success_rate": 0.75}

    trainer._periodic_eval_hooks = lambda: [(1, _eval_metrics, True)]  # type: ignore[method-assign]
    trainer._update_best_checkpoint_pool = (  # type: ignore[method-assign]
        lambda metrics: events.append(f"best:{int(stub.global_step)}")
    )
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(  # type: ignore[method-assign]
        stub,
        on_save=lambda: events.append(f"latest:{int(stub.global_step)}"),
    )
    trainer._update_recent_checkpoint_pool = (  # type: ignore[method-assign]
        lambda: events.append(f"recent:{int(stub.global_step)}")
    )

    TrainLoopEngine(trainer).run()

    assert events == ["eval:1", "best:1", "recent:1", "latest:1", "recent:1"]
    assert logger.records == [
        (
            {
                "train/loss": 1.0,
                "train/lr": 1e-3,
                "sim_eval/success_rate": 0.75,
            },
            1,
        )
    ]
    assert not any((latest_dir.parent / "training_state").glob("*step_log*"))
    assert not CheckpointBoundary(stub, checkpoint_every=1).should_replay_on_resume()


def test_train_loop_engine_replays_checkpoint_only_resume_boundary(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    latest_dir = tmp_path / "run" / "checkpoints" / "latest"
    latest_dir.mkdir(parents=True)

    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    stub.run_context = TrainerRunContext(
        output_dir=tmp_path / "run",
        resume_checkpoint=str(latest_dir),
    )
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": 1,
            "val_every": None,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = []
    stub.global_step = 1
    stub.logger = _StubLogger()
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: pytest.fail("train step should not run")  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = _no_eval_hooks  # type: ignore[method-assign]

    events: list[str] = []
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(  # type: ignore[method-assign]
        stub,
        on_save=lambda: events.append(f"latest:{int(stub.global_step)}"),
    )
    trainer._update_recent_checkpoint_pool = (  # type: ignore[method-assign]
        lambda: events.append(f"recent:{int(stub.global_step)}")
    )

    TrainLoopEngine(trainer).run()

    assert events == ["recent:1", "latest:1", "recent:1"]
    assert not CheckpointBoundary(stub, checkpoint_every=1).should_replay_on_resume()


@pytest.mark.parametrize("sidecar_payload", ["{", "[]"])
def test_train_loop_engine_ignores_corrupt_replay_step_log_payload(
    tmp_path: Path,
    sidecar_payload: str,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    latest_dir = tmp_path / "run" / "checkpoints" / "latest"
    latest_dir.mkdir(parents=True)

    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    stub.run_context = TrainerRunContext(
        output_dir=tmp_path / "run",
        resume_checkpoint=str(latest_dir),
    )
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": 1,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = []
    stub.global_step = 1
    stub.logger = _StubLogger()
    stub.train_metrics = _StubTrainMetrics()

    boundary = CheckpointBoundary(stub, checkpoint_every=1)
    pending_path = boundary._pending_step_log_path(latest_dir)
    pending_path.parent.mkdir(parents=True)
    pending_path.write_text(sidecar_payload, encoding="utf-8")

    trainer.train_step = lambda batch: pytest.fail("train step should not run")  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = lambda: [(1, lambda: {"success_rate": 0.5}, True)]  # type: ignore[method-assign]
    trainer._update_best_checkpoint_pool = lambda metrics: None  # type: ignore[method-assign]
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    assert stub.logger.records == [({"sim_eval/success_rate": 0.5}, 1)]
    assert not boundary.should_replay_on_resume()


def test_train_loop_engine_skips_replay_for_completed_resume_checkpoint_phase(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    latest_dir = tmp_path / "run" / "checkpoints" / "latest"
    latest_dir.mkdir(parents=True)

    stub.run_train_loop = True
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    stub.run_context = TrainerRunContext(
        output_dir=tmp_path / "run",
        resume_checkpoint=str(latest_dir),
    )
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": 1,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = []
    stub.global_step = 1
    stub.logger = _StubLogger()
    stub.train_metrics = _StubTrainMetrics()
    CheckpointBoundary(stub, checkpoint_every=1).mark_complete(latest_dir)
    trainer.train_step = lambda batch: pytest.fail("train step should not run")  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = lambda: [  # type: ignore[method-assign]
        (1, lambda: pytest.fail("completed eval should not replay"), True)
    ]

    events: list[str] = []
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(  # type: ignore[method-assign]
        stub,
        on_save=lambda: events.append(f"latest:{int(stub.global_step)}"),
    )
    trainer._update_recent_checkpoint_pool = (  # type: ignore[method-assign]
        lambda: events.append(f"recent:{int(stub.global_step)}")
    )

    TrainLoopEngine(trainer).run()

    assert events == ["latest:1", "recent:1"]


@pytest.mark.parametrize("checkpoint_every,checkpoint_start_step", [(0, 0), (1, 2)])
def test_train_loop_engine_respects_zero_or_delayed_checkpoint(
    tmp_path: Path, checkpoint_every: int, checkpoint_start_step: int
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None)
    logger = _StubLogger()

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": checkpoint_every,
            "checkpoint_start_step": checkpoint_start_step,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = lambda: [(1, _empty_eval_metrics, False)]  # type: ignore[method-assign]

    latest_saves: list[int] = []
    recent_updates: list[int] = []
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(  # type: ignore[method-assign]
        stub,
        on_save=lambda: latest_saves.append(int(stub.global_step)),
    )
    trainer._update_recent_checkpoint_pool = (  # type: ignore[method-assign]
        lambda: recent_updates.append(int(stub.global_step))
    )

    TrainLoopEngine(trainer).run()

    assert latest_saves == [1]
    assert recent_updates == [1]
    assert runtime.wait_calls == 2


def test_train_loop_engine_updates_fsdp_best_pool_on_non_main_rank(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": None,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "fsdp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = _StubLogger()
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]

    best_updates: list[dict[str, float]] = []
    trainer._periodic_eval_hooks = lambda: [  # type: ignore[method-assign]
        (1, lambda: {"val_loss": 2.0}, True)
    ]
    trainer._update_best_checkpoint_pool = (  # type: ignore[method-assign]
        lambda metrics: best_updates.append(dict(metrics))
    )
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    assert best_updates == [{"val_loss": 2.0}]


def test_train_loop_engine_enters_ddp_best_pool_on_non_main_rank(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": None,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = _StubLogger()
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]

    best_updates: list[dict[str, float]] = []
    trainer._periodic_eval_hooks = lambda: [  # type: ignore[method-assign]
        (1, lambda: {"val_loss": 2.0}, True)
    ]
    trainer._update_best_checkpoint_pool = (  # type: ignore[method-assign]
        lambda metrics: best_updates.append(dict(metrics))
    )
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    assert best_updates == [{"val_loss": 2.0}]


def test_train_loop_engine_enters_topk_update_with_empty_rank_metrics(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None, is_main_process=False)

    stub.run_train_loop = True
    stub.runtime = runtime
    stub.cfg = OmegaConf.create({"output_dir": str(tmp_path / "run")})
    _attach_run_context(stub, tmp_path)
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": None,
            "val_every": 1,
            "checkpoint_dir": str(tmp_path / "run" / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(2, 1)}]
    stub.global_step = 0
    stub.logger = _StubLogger()
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]

    best_updates: list[dict[str, float]] = []
    trainer._periodic_eval_hooks = lambda: [  # type: ignore[method-assign]
        (1, _empty_eval_metrics, True)
    ]
    trainer._update_best_checkpoint_pool = (  # type: ignore[method-assign]
        lambda metrics: best_updates.append(dict(metrics))
    )
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    assert best_updates == [{}]


def test_train_loop_engine_skips_start_sentinel_without_output_dir(
    tmp_path: Path,
) -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    logger = _StubLogger()

    stub.run_train_loop = True
    stub.run_context = TrainerRunContext(output_dir=None, resume_checkpoint=None)
    stub.runtime = _StubRuntime(None)
    stub.cfg = OmegaConf.create({})
    stub.tc = OmegaConf.create(
        {
            "max_steps": 1,
            "log_every": 999,
            "checkpoint_every": None,
            "val_every": None,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "distributed_strategy": "ddp",
        }
    )
    stub.model = _StubModel()
    stub.device = torch.device("cpu")
    stub.action_key = "action"
    stub.dataloader = [{"action": torch.zeros(1, 1)}]
    stub.global_step = 0
    stub.logger = logger
    stub.train_metrics = _StubTrainMetrics()
    trainer.train_step = lambda batch: (torch.tensor(1.0), {}, True)  # type: ignore[method-assign]
    trainer.save_latest_checkpoint = _stub_latest_checkpoint_save(stub)  # type: ignore[method-assign]
    trainer._update_recent_checkpoint_pool = lambda: None  # type: ignore[method-assign]
    trainer._periodic_eval_hooks = _no_eval_hooks  # type: ignore[method-assign]
    trainer._on_train_end = lambda: None  # type: ignore[method-assign]

    TrainLoopEngine(trainer).run()

    assert stub.global_step == 1


def test_eval_hooks_service_skips_completion_sentinel_without_output_dir() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    stub = cast(Any, trainer)
    runtime = _StubRuntime(None)
    logger = _StubLogger()

    stub.run_context = TrainerRunContext(output_dir=None, resume_checkpoint=None)
    stub.runtime = runtime
    stub.logger = logger
    stub.global_step = 7

    EvalHooksService(trainer).on_train_end()

    assert logger.finished
    assert runtime.end_training_calls == 1
