# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

import praxis.training.codecs_trainer as codecs_trainer_module
from praxis.core.types import FeatureSpec, FeatureType, NormMode
from praxis.data.xtokenizer_canonical_stats import (
    write_xtokenizer_quantile_statistics,
)
from praxis.data.xtokenizer_vl import (
    VLFeatureSidecar,
    file_sha256,
    vl_sidecar_generation_fingerprint,
)
from praxis.training.codecs_trainer import (
    CodecsTrainer,
    _bind_xtokenizer_canonical_stats_to_runtime_config,
    _bind_xtokenizer_sidecar_to_runtime_config,
    _open_xtokenizer_canonical_stats,
    _resolve_xtokenizer_action_feature,
)

DATASET_REVISION = "a" * 40
DOF_DIMS = {
    "follow_left_gripper": 1,
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
}
DOF_ORDER = tuple(DOF_DIMS)


def _dataset_cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        name="xtokenizer_bridge",
        adapter="xtokenizer_sidecar",
        repo_id="org/bridge",
        dataset_revision=DATASET_REVISION,
        robot_type_name="WidowX",
        action_horizon=32,
        canonical_stats=str(tmp_path / "canonical_stats.json"),
        action_layout=SimpleNamespace(
            convention="absolute",
            dof_dims=DOF_DIMS,
            dof_order=DOF_ORDER,
        ),
    )


def _codec_cfg(*, action_horizon: int = 32) -> SimpleNamespace:
    return SimpleNamespace(
        name="xtokenizer",
        kind="latents",
        action_horizon=action_horizon,
    )


def _write_stats(
    tmp_path: Path,
    *,
    filename: str = "canonical_stats.json",
    action_min_first: float = -1.0,
    stats_horizon: int = 32,
) -> Path:
    action_minimum = [-1.0] * 10
    action_minimum[0] = action_min_first
    return write_xtokenizer_quantile_statistics(
        tmp_path / filename,
        dataset_repo_id="org/bridge",
        dataset_revision=DATASET_REVISION,
        robot_type="WidowX",
        dof_dims=DOF_DIMS,
        dof_order=DOF_ORDER,
        stats_horizon=stats_horizon,
        predict_action_statistics={
            "q001": action_minimum,
            "q999": [1.0] * 10,
        },
        observation_state_statistics={
            "q001": [-10.0] * 10,
            "q999": [10.0] * 10,
        },
    )


def _write_sidecar(root: Path, *, generation: str) -> VLFeatureSidecar:
    root.mkdir(parents=True, exist_ok=True)
    fill_value = 0.0 if generation == "generation-a" else 1.0
    features = np.full((1, 2, 1, 3), fill_value, dtype=np.float32)
    np.save(root / "features.npy", features)
    np.save(root / "next_features.npy", features)
    np.save(root / "valid.npy", np.ones((1, 2, 1), dtype=np.bool_))
    manifest = {
        "format_version": 3,
        "dataset_repo_id": "org/bridge",
        "dataset_revision": DATASET_REVISION,
        "extractor_revision": "extractor-v1",
        "model_id": "Qwen/test",
        "model_revision": "model-v1",
        "layer": -3,
        "stride": 2,
        "feature_dim": 3,
        "episode_indices": [0],
        "episode_lengths": [4],
        "image_keys": ["observation.images.face"],
        "global_instruction": "global",
        "local_subtask": "local",
        "synthetic": True,
        "features_sha256": file_sha256(root / "features.npy"),
        "next_features_sha256": file_sha256(root / "next_features.npy"),
        "valid_sha256": file_sha256(root / "valid.npy"),
        "features_file": "features.npy",
        "next_features_file": "next_features.npy",
        "valid_file": "valid.npy",
    }
    manifest["generation_fingerprint"] = vl_sidecar_generation_fingerprint(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return VLFeatureSidecar.open(root, dataset_revision=DATASET_REVISION)


def test_xtokenizer_bridge_requires_canonical_stats_before_model_init(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="canonical.*statistics"):
        _open_xtokenizer_canonical_stats(_dataset_cfg(tmp_path), _codec_cfg())


def test_xtokenizer_bridge_rejects_legacy_v1_statistics(tmp_path: Path) -> None:
    path = _write_stats(tmp_path)
    payload = json.loads(path.read_text())
    payload["format_version"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="format_version mismatch.*1 != 3"):
        _open_xtokenizer_canonical_stats(_dataset_cfg(tmp_path), _codec_cfg())


def test_xtokenizer_bridge_rejects_malformed_statistics_json(tmp_path: Path) -> None:
    Path(_dataset_cfg(tmp_path).canonical_stats).write_text("{not-json")

    with pytest.raises(ValueError, match="quantile statistics.*malformed JSON"):
        _open_xtokenizer_canonical_stats(_dataset_cfg(tmp_path), _codec_cfg())


def test_xtokenizer_bridge_rejects_stats_horizon_mismatch_before_training(
    tmp_path: Path,
) -> None:
    _write_stats(tmp_path)
    dataset_cfg = _dataset_cfg(tmp_path)
    dataset_cfg.action_horizon = 16

    with pytest.raises(ValueError, match="stats_horizon mismatch.*32 != 16"):
        _open_xtokenizer_canonical_stats(
            dataset_cfg,
            _codec_cfg(action_horizon=16),
        )


def test_xtokenizer_bridge_rejects_dataset_codec_horizon_mismatch_before_io(
    tmp_path: Path,
) -> None:
    dataset_cfg = _dataset_cfg(tmp_path)
    dataset_cfg.action_horizon = 8

    with pytest.raises(
        ValueError,
        match=(
            r"dataset\.action_horizon=8.*"
            r"codec\.action_horizon=16"
        ),
    ):
        _open_xtokenizer_canonical_stats(
            dataset_cfg,
            _codec_cfg(action_horizon=16),
        )


def test_xtokenizer_bridge_rejects_non_integer_horizon_before_io(
    tmp_path: Path,
) -> None:
    dataset_cfg = _dataset_cfg(tmp_path)
    dataset_cfg.action_horizon = 16.5

    with pytest.raises(ValueError, match="action_horizon.*positive integers"):
        _open_xtokenizer_canonical_stats(
            dataset_cfg,
            _codec_cfg(action_horizon=16),
        )


def test_xtokenizer_stats_horizon_tracks_codec_action_horizon(
    tmp_path: Path,
) -> None:
    _write_stats(tmp_path, stats_horizon=16)
    dataset_cfg = _dataset_cfg(tmp_path)
    dataset_cfg.action_horizon = 16

    artifact = _open_xtokenizer_canonical_stats(
        dataset_cfg,
        _codec_cfg(action_horizon=16),
    )

    assert artifact is not None
    assert artifact.stats_horizon == 16


def test_xtokenizer_bridge_builds_d10_action_feature_from_canonical_stats(
    tmp_path: Path,
) -> None:
    _write_stats(tmp_path)
    artifact = _open_xtokenizer_canonical_stats(
        _dataset_cfg(tmp_path),
        _codec_cfg(),
    )
    raw_feature = FeatureSpec(
        key="action",
        type=FeatureType.ACTION,
        shape=(7,),
        norm_mode=NormMode.MIN_MAX,
    )

    feature = _resolve_xtokenizer_action_feature(raw_feature, artifact)

    assert feature.key == "action"
    assert feature.shape == (10,)
    assert feature.norm_mode is NormMode.MIN_MAX


def test_xtokenizer_bridge_rejects_noncanonical_action_dimension(
    tmp_path: Path,
) -> None:
    dataset_cfg = _dataset_cfg(tmp_path)
    dataset_cfg.action_layout.dof_dims = {
        **DOF_DIMS,
        "unrelated_channel": 1,
    }
    dataset_cfg.action_layout.dof_order = (
        *DOF_ORDER,
        "unrelated_channel",
    )

    with pytest.raises(ValueError, match="canonical.*10-D"):
        _open_xtokenizer_canonical_stats(dataset_cfg, _codec_cfg())


def test_canonical_stats_binding_is_written_exactly_without_path(
    tmp_path: Path,
) -> None:
    _write_stats(tmp_path)
    artifact = _open_xtokenizer_canonical_stats(
        _dataset_cfg(tmp_path),
        _codec_cfg(),
    )
    assert artifact is not None
    cfg = OmegaConf.create({"dataset": {}})

    binding = _bind_xtokenizer_canonical_stats_to_runtime_config(
        cfg,
        artifact,
        resume_checkpoint=None,
    )

    assert binding == artifact.identity
    assert set(binding) == {
        "format_version",
        "dataset_repo_id",
        "dataset_revision",
        "robot_type",
        "action_layout",
        "normalization",
        "content_fingerprint",
    }
    assert binding["normalization"]["quantile_low"] == 0.001
    assert binding["normalization"]["quantile_high"] == 0.999
    assert binding["normalization"]["clamp"] is True
    assert binding["normalization"]["stats_horizon"] == 32
    assert OmegaConf.to_container(cfg.dataset.canonical_stats_identity) == binding
    assert str(tmp_path) not in json.dumps(binding, sort_keys=True)


def test_resume_accepts_identical_canonical_stats_at_relocated_path(
    tmp_path: Path,
) -> None:
    original_path = _write_stats(tmp_path, filename="original.json")
    original_cfg = _dataset_cfg(tmp_path)
    original_cfg.canonical_stats = str(original_path)
    original = _open_xtokenizer_canonical_stats(original_cfg, _codec_cfg())
    assert original is not None
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "pretrained_model"
    model_dir.mkdir(parents=True)
    saved_cfg = OmegaConf.create({"dataset": {}})
    _bind_xtokenizer_canonical_stats_to_runtime_config(
        saved_cfg,
        original,
        resume_checkpoint=None,
    )
    (model_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(saved_cfg))
    relocated_path = tmp_path / "relocated" / "stats.json"
    relocated_path.parent.mkdir()
    shutil.copyfile(original_path, relocated_path)
    relocated_cfg = _dataset_cfg(tmp_path)
    relocated_cfg.canonical_stats = str(relocated_path)
    relocated = _open_xtokenizer_canonical_stats(relocated_cfg, _codec_cfg())
    assert relocated is not None

    binding = _bind_xtokenizer_canonical_stats_to_runtime_config(
        OmegaConf.create({"dataset": {}}),
        relocated,
        resume_checkpoint=str(checkpoint),
    )

    assert binding == original.identity


def test_trainer_binds_v2_predict_and_observation_roles_to_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_stats(tmp_path)
    artifact = _open_xtokenizer_canonical_stats(
        _dataset_cfg(tmp_path),
        _codec_cfg(),
    )
    assert artifact is not None
    calls: list[
        tuple[
            dict[str, dict[str, torch.Tensor]],
            dict[str, dict[str, torch.Tensor]],
            dict[str, Any],
            int,
        ]
    ] = []

    class _Codec:
        action_dim = 10

        def set_io(self, io: object) -> None:
            assert io is trainer.io

        def set_xtokenizer_normalization_stats(
            self,
            *,
            predict_stats: dict[str, dict[str, torch.Tensor]],
            observation_stats: dict[str, dict[str, torch.Tensor]],
            canonical_stats_identity: dict[str, Any],
            stats_horizon: int,
        ) -> None:
            calls.append(
                (
                    predict_stats,
                    observation_stats,
                    canonical_stats_identity,
                    stats_horizon,
                )
            )

        def set_normalization_stats(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("paper recipe must not use dataset normalizers")

    codec = _Codec()
    monkeypatch.setattr(codecs_trainer_module, "make_codec", lambda _cfg: codec)
    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.cfg = SimpleNamespace(
        codec=OmegaConf.create(
            {
                "name": "xtokenizer",
                "kind": "latents",
                "action_dim": 10,
                "action_horizon": 32,
            }
        )
    )
    trainer.dataset_action_dim = 10
    trainer.action_key = "action"
    trainer.dataset_action_layout = {
        "convention": "absolute",
        "dof_dims": DOF_DIMS,
        "dof_order": list(DOF_ORDER),
    }
    trainer.runtime = SimpleNamespace(is_main_process=True)
    trainer.io = object()
    trainer.dataset = SimpleNamespace(
        stats={"action": {"min": [-100.0] * 10, "max": [100.0] * 10}},
        normalization_modes={"action": NormMode.MIN_MAX},
    )
    trainer.xtokenizer_normalization_statistics = artifact

    trainer._init_model()

    assert len(calls) == 1
    predict_stats, observation_stats, canonical_identity, stats_horizon = calls[0]
    expected_predict = artifact.predict_normalization_stats("action")["action"]
    expected_observation = artifact.observation_normalization_stats(
        "observation.state"
    )["observation.state"]
    for name in ("min", "max"):
        torch.testing.assert_close(
            predict_stats["action"][name], expected_predict[name]
        )
        torch.testing.assert_close(
            observation_stats["observation.state"][name],
            expected_observation[name],
        )
    assert canonical_identity == artifact.identity
    assert stats_horizon == artifact.stats_horizon


def test_non_paper_xtokenizer_trainer_uses_generic_mean_std_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[
        tuple[
            dict[str, dict[str, torch.Tensor]],
            dict[str, NormMode] | None,
        ]
    ] = []

    class _Codec:
        action_dim = 10

        def set_io(self, io: object) -> None:
            assert io is trainer.io

        def set_normalization_stats(
            self,
            stats: dict[str, dict[str, torch.Tensor]],
            modes: dict[str, NormMode] | None = None,
        ) -> None:
            calls.append((stats, modes))

        def set_xtokenizer_normalization_stats(self, **_kwargs: object) -> None:
            raise AssertionError("non-paper path must use generic normalization")

    codec = _Codec()
    monkeypatch.setattr(codecs_trainer_module, "make_codec", lambda _cfg: codec)
    trainer: Any = CodecsTrainer.__new__(CodecsTrainer)
    trainer.cfg = SimpleNamespace(
        codec=OmegaConf.create(
            {
                "name": "xtokenizer",
                "kind": "latents",
                "action_dim": 10,
                "action_horizon": 32,
            }
        )
    )
    trainer.dataset_action_dim = 10
    trainer.action_key = "action"
    trainer.dataset_action_layout = None
    trainer.runtime = SimpleNamespace(is_main_process=True)
    trainer.io = object()
    generic_stats = {
        "action": {
            "mean": torch.zeros(10),
            "std": torch.ones(10),
        }
    }
    generic_modes = {"action": NormMode.MEAN_STD}
    trainer.dataset = SimpleNamespace(
        stats=generic_stats,
        normalization_modes=generic_modes,
    )
    trainer.xtokenizer_normalization_statistics = None

    trainer._init_model()

    assert calls == [(generic_stats, generic_modes)]


def test_replaced_canonical_stats_resume_rejects_during_data_init(
    tmp_path: Path,
) -> None:
    active_path = _write_stats(tmp_path, filename="active.json")
    changed_path = _write_stats(
        tmp_path,
        filename="changed.json",
        action_min_first=-0.25,
    )
    active_cfg = _dataset_cfg(tmp_path)
    active_cfg.canonical_stats = str(active_path)
    original = _open_xtokenizer_canonical_stats(active_cfg, _codec_cfg())
    assert original is not None
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "pretrained_model"
    model_dir.mkdir(parents=True)
    saved_cfg = OmegaConf.create({"dataset": {}})
    _bind_xtokenizer_canonical_stats_to_runtime_config(
        saved_cfg,
        original,
        resume_checkpoint=None,
    )
    (model_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(saved_cfg))
    active_path.unlink()
    shutil.copyfile(changed_path, active_path)
    events: list[str] = []

    class _OrderingTrainer(CodecsTrainer):
        def _init_data(self) -> None:
            events.append("data")
            artifact = _open_xtokenizer_canonical_stats(active_cfg, _codec_cfg())
            assert artifact is not None
            _bind_xtokenizer_canonical_stats_to_runtime_config(
                self.cfg,
                artifact,
                resume_checkpoint=self._resume_checkpoint,
            )

        def _init_model(self) -> None:
            events.append("model")
            raise AssertionError("model initialization must not be reached")

        def _build_optimizer(self, optim_params: list[dict[str, Any]]):
            del optim_params
            events.append("optimizer")
            raise AssertionError("optimizer initialization must not be reached")

    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train_codecs",
            overrides=[
                "codec=latents/xtokenizer",
                "dataset=xtokenizer_bridge",
                "train_recipe/codecs=xtokenizer",
                "device=cpu",
                "trainer.use_wandb=false",
                "trainer.mixed_precision=no",
                "run_mode=resume",
                f"resume_checkpoint={checkpoint}",
                f"output_dir={tmp_path / 'run'}",
            ],
        )

    with pytest.raises(ValueError, match="canonical statistics identity mismatch"):
        _OrderingTrainer(cfg)

    assert events == ["data"]


def test_fresh_sidecar_binding_is_written_into_resolved_runtime_config(
    tmp_path: Path,
) -> None:
    sidecar = _write_sidecar(tmp_path / "sidecar", generation="generation-a")
    cfg = OmegaConf.create({"dataset": {}})

    binding = _bind_xtokenizer_sidecar_to_runtime_config(
        cfg,
        sidecar,
        resume_checkpoint=None,
    )

    assert binding["generation_fingerprint"] == sidecar.manifest.generation_fingerprint
    assert OmegaConf.to_container(cfg.dataset.vl_sidecar_identity) == binding


def test_resume_accepts_same_saved_sidecar_identity(tmp_path: Path) -> None:
    sidecar = _write_sidecar(tmp_path / "sidecar", generation="generation-a")
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "pretrained_model"
    model_dir.mkdir(parents=True)
    saved_cfg = OmegaConf.create({"dataset": {}})
    _bind_xtokenizer_sidecar_to_runtime_config(
        saved_cfg,
        sidecar,
        resume_checkpoint=None,
    )
    (model_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(saved_cfg))
    resumed_cfg = OmegaConf.create({"dataset": {}})

    binding = _bind_xtokenizer_sidecar_to_runtime_config(
        resumed_cfg,
        sidecar,
        resume_checkpoint=str(checkpoint),
    )

    assert binding["generation_fingerprint"] == sidecar.manifest.generation_fingerprint


def test_changed_sidecar_resume_rejects_before_model_or_optimizer_init(
    tmp_path: Path,
) -> None:
    original = _write_sidecar(tmp_path / "original", generation="generation-a")
    changed = _write_sidecar(tmp_path / "changed", generation="generation-b")
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "pretrained_model"
    model_dir.mkdir(parents=True)
    saved_cfg = OmegaConf.create({"dataset": {}})
    _bind_xtokenizer_sidecar_to_runtime_config(
        saved_cfg,
        original,
        resume_checkpoint=None,
    )
    (model_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(saved_cfg))

    events: list[str] = []

    class _OrderingTrainer(CodecsTrainer):
        def _init_data(self) -> None:
            events.append("data")
            self.vl_sidecar = changed
            _bind_xtokenizer_sidecar_to_runtime_config(
                self.cfg,
                changed,
                resume_checkpoint=self._resume_checkpoint,
            )

        def _init_model(self) -> None:
            events.append("model")
            raise AssertionError("model initialization must not be reached")

        def _build_optimizer(self, optim_params: list[dict[str, Any]]):
            del optim_params
            events.append("optimizer")
            raise AssertionError("optimizer initialization must not be reached")

    config_dir = Path("src/praxis/configs").resolve()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="train_codecs",
            overrides=[
                "codec=latents/xtokenizer",
                "dataset=xtokenizer_bridge",
                "train_recipe/codecs=xtokenizer",
                "device=cpu",
                "trainer.use_wandb=false",
                "trainer.mixed_precision=no",
                "run_mode=resume",
                f"resume_checkpoint={checkpoint}",
                f"output_dir={tmp_path / 'run'}",
            ],
        )

    with pytest.raises(ValueError, match="sidecar identity mismatch"):
        _OrderingTrainer(cfg)

    assert events == ["data"]


@pytest.mark.parametrize(
    "saved_binding",
    [
        None,
        {},
        {"generation_fingerprint": ""},
        {"generation_fingerprint": "generation-a"},
    ],
)
def test_resume_fails_closed_on_missing_or_malformed_saved_identity(
    tmp_path: Path,
    saved_binding: dict[str, object] | None,
) -> None:
    sidecar = _write_sidecar(tmp_path / "sidecar", generation="generation-a")
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "pretrained_model"
    model_dir.mkdir(parents=True)
    payload: dict[str, object] = {"dataset": {}}
    if saved_binding is not None:
        payload["dataset"] = {"vl_sidecar_identity": saved_binding}
    (model_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(payload))

    with pytest.raises(ValueError, match="saved.*sidecar identity"):
        _bind_xtokenizer_sidecar_to_runtime_config(
            OmegaConf.create({"dataset": {}}),
            sidecar,
            resume_checkpoint=str(checkpoint),
        )


def test_resume_reports_malformed_saved_config_as_identity_error(
    tmp_path: Path,
) -> None:
    sidecar = _write_sidecar(tmp_path / "sidecar", generation="generation-a")
    checkpoint = tmp_path / "checkpoint"
    model_dir = checkpoint / "pretrained_model"
    model_dir.mkdir(parents=True)
    (model_dir / "train_config.yaml").write_text("dataset: [unterminated\n")

    with pytest.raises(ValueError, match="saved.*sidecar identity"):
        _bind_xtokenizer_sidecar_to_runtime_config(
            OmegaConf.create({"dataset": {}}),
            sidecar,
            resume_checkpoint=str(checkpoint),
        )
