# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for trainer checkpoint behavior with EMA-applied top-k export."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from safetensors.torch import load_file, save_file, save_model as save_model_safetensor

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.core import BasePolicy
from praxis.training.core.checkpoint.io import DdpCheckpointIO
from praxis.training.core.ema import EMAModel
from praxis.training.core.trainer import DefaultTrainer
from praxis.training.core.trainer_components import CheckpointService


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.config = {"name": "dummy"}

    def save_pretrained(self, save_directory: str | Path, config=None) -> None:
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file({"weight": self.weight.detach().cpu()}, out_dir / "model.safetensors")
        (out_dir / "config.json").write_text("{}\n")


class _DummyPolicyModel(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.config = {"name": "dummy_policy"}

    def forward(self, batch: dict[str, torch.Tensor]):
        loss = batch["action"].float().mean() * 0
        return loss, {"loss": float(loss.item())}

    def predict_action(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        del obs
        return torch.zeros((1, 2), dtype=torch.float32)


class _DummyTiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(4, 3)
        self.lm_head = nn.Linear(3, 4, bias=False)
        self.lm_head.weight = self.embed.weight
        self.config = {"name": "dummy_tied"}

    def save_pretrained(self, save_directory: str | Path, config=None) -> None:
        del config
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_model_safetensor(self, out_dir / "model.safetensors")
        (out_dir / "config.json").write_text("{}\n")


class _DummyRuntime:
    is_main_process = True

    @staticmethod
    def unwrap_model(model):
        return model


def _make_min_trainer(model: nn.Module, ema: EMAModel) -> DefaultTrainer:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    trainer.runtime = _DummyRuntime()
    trainer.model = model
    trainer.optimizer = None
    trainer.scheduler = None
    trainer.ema = ema
    trainer.global_step = 123
    trainer.cfg = OmegaConf.create({"name": "test"})
    trainer.tc = OmegaConf.create({"distributed_strategy": "ddp"})
    return trainer


class _DummyLoRAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(
            2, 2, bias=False
        )  # frozen backbone-like param
        self.decoder_adapter = nn.Linear(
            2, 2, bias=False
        )  # trainable adapter-like param
        self.normalize = nn.Module()
        self.normalize.register_buffer("stat", torch.tensor([1.0], dtype=torch.float32))
        self.config = {"name": "dummy_lora"}

        self.encoder.backbone.weight.requires_grad_(False)
        self.decoder_adapter.weight.requires_grad_(True)

    def save_pretrained(self, save_directory: str | Path, config=None) -> None:
        del config
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu() for k, v in self.state_dict().items()},
            out_dir / "model.safetensors",
        )
        (out_dir / "config.json").write_text("{}\n")


class _DummyFilteredModel(_DummyLoRAModel):
    def state_dict(self, *args, **kwargs):
        out = super().state_dict(*args, **kwargs)
        out.pop("encoder.backbone.weight", None)
        return out


def test_latest_stamps_wandb_run_info_when_logger_active(tmp_path: Path) -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)
    trainer = _make_min_trainer(model, ema)
    trainer.logger = types.SimpleNamespace(
        _wandb_run=types.SimpleNamespace(
            id="run_abc",
            project="proj",
            entity="team",
        )
    )

    out_dir = tmp_path / "latest"
    trainer._save_checkpoint_dir(
        out_dir, apply_ema_weights=False, include_training_state=True
    )

    info_path = out_dir / "training_state" / "wandb_run_info.json"
    assert info_path.exists()
    info = json.loads(info_path.read_text())
    assert info == {"id": "run_abc", "project": "proj", "entity": "team"}


def test_latest_skips_wandb_stamp_when_no_logger(tmp_path: Path) -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)
    trainer = _make_min_trainer(model, ema)

    out_dir = tmp_path / "latest_no_wandb"
    trainer._save_checkpoint_dir(
        out_dir, apply_ema_weights=False, include_training_state=True
    )

    assert not (out_dir / "training_state" / "wandb_run_info.json").exists()


def test_read_checkpoint_wandb_run_info_round_trip(tmp_path: Path) -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)
    trainer = _make_min_trainer(model, ema)
    trainer.logger = types.SimpleNamespace(
        _wandb_run=types.SimpleNamespace(
            id="run_roundtrip", project="proj", entity="team"
        )
    )

    ckpt_dir = tmp_path / "latest"
    trainer._save_checkpoint_dir(
        ckpt_dir, apply_ema_weights=False, include_training_state=True
    )

    info = CheckpointService.read_checkpoint_wandb_run_info(ckpt_dir)
    assert info == {"id": "run_roundtrip", "project": "proj", "entity": "team"}

    assert (
        CheckpointService.read_checkpoint_wandb_run_info(tmp_path / "nonexistent")
        is None
    )


def test_topk_does_not_stamp_wandb_run_info(tmp_path: Path) -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)
    trainer = _make_min_trainer(model, ema)
    trainer.logger = types.SimpleNamespace(
        _wandb_run=types.SimpleNamespace(id="run_xyz", project="proj", entity="team")
    )

    out_dir = tmp_path / "topk_no_stamp"
    trainer._save_checkpoint_dir(
        out_dir, apply_ema_weights=True, include_training_state=False
    )

    assert not (out_dir / "training_state" / "wandb_run_info.json").exists()


def test_latest_saves_raw_and_ema(tmp_path: Path) -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)

    # Build an EMA shadow that differs from raw model.
    model.weight.data.fill_(3.0)
    ema.update(model)  # shadow = 0.5 * 1 + 0.5 * 3 = 2
    trainer = _make_min_trainer(model, ema)

    out_dir = tmp_path / "latest"
    trainer._save_checkpoint_dir(
        out_dir, apply_ema_weights=False, include_training_state=True
    )

    saved = load_file(out_dir / "pretrained_model" / "model.safetensors")
    assert torch.allclose(saved["weight"], torch.tensor([3.0], dtype=torch.float32))
    assert torch.allclose(
        model.weight.detach(), torch.tensor([3.0], dtype=torch.float32)
    )
    assert (out_dir / "training_state" / "ema.pt").exists()


def test_topk_saves_ema_applied_without_ema_state(tmp_path: Path) -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)

    # Build an EMA shadow that differs from raw model.
    model.weight.data.fill_(3.0)
    ema.update(model)  # shadow = 2
    trainer = _make_min_trainer(model, ema)

    out_dir = tmp_path / "topk" / "step_100"
    trainer._save_checkpoint_dir(
        out_dir, apply_ema_weights=True, include_training_state=False
    )

    saved = load_file(out_dir / "pretrained_model" / "model.safetensors")
    # Exported model is EMA-applied.
    assert torch.allclose(saved["weight"], torch.tensor([2.0], dtype=torch.float32))
    # In-memory training model is restored to raw.
    assert torch.allclose(
        model.weight.detach(), torch.tensor([3.0], dtype=torch.float32)
    )
    assert not (out_dir / "training_state" / "ema.pt").exists()


def test_checkpoint_respects_model_state_dict_filtering(tmp_path: Path) -> None:
    model = _DummyFilteredModel()
    ema = EMAModel(model, decay=0.5)
    trainer = _make_min_trainer(model, ema)

    out_dir = tmp_path / "filtered"
    trainer._save_checkpoint_dir(
        out_dir, apply_ema_weights=False, include_training_state=True
    )

    saved = load_file(out_dir / "pretrained_model" / "model.safetensors")
    assert "encoder.backbone.weight" not in saved
    assert "decoder_adapter.weight" in saved
    assert "normalize.stat" in saved


def test_ema_eval_scope_applies_then_restores() -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)  # shadow = 2
    trainer = _make_min_trainer(model, ema)

    assert torch.allclose(
        model.weight.detach(), torch.tensor([3.0], dtype=torch.float32)
    )
    with trainer._ema_eval_scope() as eval_model:
        assert eval_model is model
        assert torch.allclose(
            model.weight.detach(), torch.tensor([2.0], dtype=torch.float32)
        )

    assert torch.allclose(
        model.weight.detach(), torch.tensor([3.0], dtype=torch.float32)
    )


def test_ema_eval_scope_restores_after_exception() -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)  # shadow = 2
    trainer = _make_min_trainer(model, ema)

    with pytest.raises(RuntimeError, match="boom"), trainer._ema_eval_scope():
        assert torch.allclose(
            model.weight.detach(), torch.tensor([2.0], dtype=torch.float32)
        )
        raise RuntimeError("boom")

    assert torch.allclose(
        model.weight.detach(), torch.tensor([3.0], dtype=torch.float32)
    )


def test_unwrap_model_does_not_apply_ema() -> None:
    model = _DummyModel()
    ema = EMAModel(model, decay=0.5)
    model.weight.data.fill_(3.0)
    ema.update(model)  # shadow = 2
    trainer = _make_min_trainer(model, ema)

    eval_model = trainer.runtime.unwrap_model(trainer.model)
    assert eval_model is model
    assert torch.allclose(
        model.weight.detach(), torch.tensor([3.0], dtype=torch.float32)
    )


def test_checkpoint_raises_when_base_policy_has_no_io(tmp_path: Path) -> None:
    model = _DummyPolicyModel()
    ema = EMAModel(model, decay=0.5)
    trainer = _make_min_trainer(model, ema)

    with pytest.raises(RuntimeError, match="has no PolicyIO"):
        trainer._save_checkpoint_dir(tmp_path / "missing_io")


def test_checkpoint_saves_policy_sidecars_when_io_present(tmp_path: Path) -> None:
    model = _DummyPolicyModel()
    model.set_io(
        PolicyIO(
            obs_features=[
                FeatureSpec(
                    key="observation.state",
                    type=FeatureType.STATE,
                    shape=(2,),
                    norm_mode=NormMode.MIN_MAX,
                )
            ],
            action_feature=FeatureSpec(
                key="action",
                type=FeatureType.ACTION,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            ),
        )
    )
    ema = EMAModel(model, decay=0.5)
    trainer = _make_min_trainer(model, ema)
    out_dir = tmp_path / "with_io"
    trainer._save_checkpoint_dir(out_dir)

    assert (out_dir / "pretrained_model" / "policy_io.json").exists()
    assert (out_dir / "pretrained_model" / "norm_stats.pt").exists()


def test_ddp_normalization_restore_uses_model_payload_hook(tmp_path: Path) -> None:
    class _PayloadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.payload: dict | None = None

        def _load_normalization_checkpoint_payload(self, data: dict) -> None:
            self.payload = data

        def set_normalization_stats(self, *args, **kwargs) -> None:
            raise AssertionError("restore must delegate through the payload hook")

    path = tmp_path / "norm_stats.pt"
    payload = {
        "normalization_kind": "test_marker_v1",
        "stats": {"action": {"min": torch.zeros(2), "max": torch.ones(2)}},
        "modes": {"action": NormMode.MIN_MAX.value},
    }
    torch.save(payload, path)
    model = _PayloadModel()

    DdpCheckpointIO._load_norm_stats(model, path)

    assert model.payload is not None
    assert model.payload["normalization_kind"] == "test_marker_v1"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {
                "normalization_kind": "unknown_v2",
                "stats": {},
                "modes": {},
            },
            id="unknown-marker-empty-stats",
        ),
        pytest.param(
            {
                "normalization_kind": "xtokenizer_dual_role_v1",
                "stats": {},
                "modes": {},
            },
            id="dual-role-empty-roles",
        ),
        pytest.param(
            {
                "normalization_kind": "xtokenizer_dual_role_v1",
                "stats": {"action": {}},
                "modes": {"action": NormMode.MIN_MAX.value},
            },
            id="dual-role-missing-observation-role",
        ),
    ],
)
def test_ddp_normalization_restore_dispatches_empty_marked_payloads(
    tmp_path: Path,
    payload: dict,
) -> None:
    class _PayloadModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.payload: dict | None = None

        def _load_normalization_checkpoint_payload(self, data: dict) -> None:
            self.payload = data

    path = tmp_path / "norm_stats.pt"
    torch.save(payload, path)
    model = _PayloadModel()

    DdpCheckpointIO._load_norm_stats(model, path)

    assert model.payload == payload


def test_ddp_normalization_restore_keeps_unmarked_empty_payload_as_noop(
    tmp_path: Path,
) -> None:
    class _PayloadModel(nn.Module):
        def _load_normalization_checkpoint_payload(self, data: dict) -> None:
            raise AssertionError(f"unmarked empty payload must remain a no-op: {data}")

    path = tmp_path / "norm_stats.pt"
    torch.save({"stats": {}, "modes": {}}, path)

    DdpCheckpointIO._load_norm_stats(_PayloadModel(), path)


def test_load_checkpoint_restores_tied_weights_saved_via_save_pretrained(
    tmp_path: Path,
) -> None:
    model = _DummyTiedModel()
    with torch.no_grad():
        model.embed.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ],
                dtype=torch.float32,
            )
        )
    trainer = _make_min_trainer(model, ema=None)
    out_dir = tmp_path / "tied"
    trainer._save_checkpoint_dir(
        out_dir, apply_ema_weights=False, include_training_state=False
    )

    reloaded = _DummyTiedModel()
    trainer2 = _make_min_trainer(reloaded, ema=None)
    trainer2.load_checkpoint(out_dir)

    torch.testing.assert_close(reloaded.embed.weight, model.embed.weight)
    torch.testing.assert_close(reloaded.lm_head.weight, model.lm_head.weight)
