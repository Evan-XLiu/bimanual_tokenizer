# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Inference export and Bridge cache compatibility for X-Tokenizer."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor

from praxis.codecs.latents.xtokenizer import XTokenizerCodec, XTokenizerModelConfig
from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.data import bridge_xtokenizer
from praxis.data.action_token_cache import ActionTokenCache
from praxis.data.lerobot_adapter import write_lerobot_cache_provenance
from praxis.data.xtokenizer_vl import vl_sidecar_generation_fingerprint
from praxis.training.core.ema import EMAModel
from scripts.tokenizer.export_xtokenizer import (
    export_xtokenizer_checkpoint,
    load_exported_xtokenizer,
)

DOF_DIMS = {
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
    "follow_left_gripper": 1,
}
REORDERED_DOF_DIMS = {
    "follow_left_gripper": 1,
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
}
DATASET_REPO_ID = "chaoqi-liu/bridge_single_view"
DATASET_REVISION = "a" * 40


def _normalization_statistics() -> dict[str, dict[str, Tensor]]:
    return {
        "predict_action": {
            "min": torch.linspace(-2.0, -1.1, 10, dtype=torch.float32),
            "max": torch.linspace(1.0, 1.9, 10, dtype=torch.float32),
        },
        "observation_state": {
            "min": torch.linspace(-4.0, -3.1, 10, dtype=torch.float32),
            "max": torch.linspace(3.0, 3.9, 10, dtype=torch.float32),
        },
    }


def _training_data_identity(
    dof_dims: dict[str, int] | None = None,
) -> dict[str, Any]:
    resolved_dof_dims = DOF_DIMS if dof_dims is None else dof_dims
    sidecar_identity = {
        "format_version": 3,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "extractor_revision": "extractor-v1",
        "model_id": "Qwen/test",
        "model_revision": "b" * 40,
        "layer": -1,
        "stride": 4,
        "feature_dim": 12,
        "episode_indices": [0, 1],
        "episode_lengths": [33, 33],
        "image_keys": ["observation.images.main"],
        "global_instruction": "global",
        "local_subtask": "local",
        "synthetic": False,
        "features_sha256": "1" * 64,
        "next_features_sha256": "2" * 64,
        "valid_sha256": "3" * 64,
    }
    sidecar_identity["generation_fingerprint"] = vl_sidecar_generation_fingerprint(
        sidecar_identity
    )
    role_stats = _normalization_statistics()
    canonical_stats_identity = {
        "format_version": 3,
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "robot_type": "WidowX",
        "action_layout": {
            "convention": "absolute",
            "dof_dims": resolved_dof_dims,
            "dof_order": list(resolved_dof_dims),
        },
        "normalization": {
            "mode": "quantile_min_max",
            "quantile_low": 0.001,
            "quantile_high": 0.999,
            "clamp": True,
            "stats_horizon": 32,
            "predict_action": {
                "space": "observation_relative",
                "q001": role_stats["predict_action"]["min"].tolist(),
                "q999": role_stats["predict_action"]["max"].tolist(),
            },
            "observation_state": {
                "space": "absolute",
                "q001": role_stats["observation_state"]["min"].tolist(),
                "q999": role_stats["observation_state"]["max"].tolist(),
            },
        },
    }
    encoded = json.dumps(
        canonical_stats_identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    canonical_stats_identity["content_fingerprint"] = hashlib.sha256(
        encoded
    ).hexdigest()
    return {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "canonical_stats_identity": canonical_stats_identity,
        "vl_sidecar_identity": sidecar_identity,
    }


def _tiny_config(
    *,
    dof_dims: dict[str, int] | None = None,
) -> XTokenizerModelConfig:
    config: dict[str, Any] = {
        "action_dim": 10,
        "action_horizon": 32,
        "compression_ratio": 4,
        "hidden_dim": 16,
        "latent_dim": 16,
        "encoder_num_heads": 4,
        "encoder_depth": 1,
        "decoder_num_heads": 4,
        "decoder_depth": 1,
        "mam_num_heads": 4,
        "mam_depth": 2,
        "prediction_num_heads": 4,
        "prediction_depth": 2,
        "dropout": 0.0,
        "codebook_size": 8,
        "num_quantizers": 4,
        "ema_decay": 0.8,
        "dead_code_threshold": 0,
        "embodiment_vocab_size": 4,
        "vl_feature_dim": 12,
        "rotation_weight": 0.0,
        "mam_weight": 0.0,
        "alignment_weight": 0.0,
        "prediction_weight": 0.0,
    }
    if dof_dims is not None:
        config["action_layout"] = {
            "convention": "absolute",
            "dof_dims": dof_dims,
            "dof_order": list(dof_dims),
        }
    return XTokenizerModelConfig.from_dict(config)


def _policy_io() -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(10,),
                norm_mode=NormMode.MIN_MAX,
            )
        ],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(10,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_action_steps=32,
    )


def _codecs_trainer_policy_io() -> PolicyIO:
    return PolicyIO(
        obs_features=[],
        action_feature=FeatureSpec(
            key="action",
            type=FeatureType.ACTION,
            shape=(10,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=0,
        n_action_steps=32,
    )


def _train_and_save_checkpoint(root: Path) -> tuple[Path, Tensor, Tensor, Tensor]:
    torch.manual_seed(19)
    codec = XTokenizerCodec(config=_tiny_config())
    codec.set_io(_policy_io())
    role_stats = _normalization_statistics()
    codec.set_xtokenizer_normalization_stats(
        predict_stats={"action": role_stats["predict_action"]},
        observation_stats={
            "observation.state": role_stats["observation_state"],
        },
        canonical_stats_identity=_training_data_identity()["canonical_stats_identity"],
        stats_horizon=32,
    )
    codec.train()
    actions = torch.randn(2, 32, 10)
    anchors = torch.randn(2, 10)
    robot_type = torch.tensor([0, 0])
    loss, _ = codec(
        {
            "action": actions,
            "observation.state": anchors,
            "robot_type": robot_type,
        }
    )
    optimizer = torch.optim.Adam(codec.get_optim_params(), lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    fixed_actions = torch.linspace(-0.75, 0.75, 320).reshape(1, 32, 10)
    fixed_anchor = torch.linspace(-0.5, 0.5, 10).reshape(1, 10)
    fixed_robot_type = torch.tensor([0])
    codec.eval()
    expected = codec.encode(
        fixed_actions,
        anchor=fixed_anchor,
        robot_type=fixed_robot_type,
    )

    checkpoint_dir = root / "checkpoint"
    model_dir = checkpoint_dir / "pretrained_model"
    codec.save_pretrained(model_dir)
    (model_dir / "train_config.yaml").write_text(
        OmegaConf.to_yaml(
            {
                "codec": {"name": "xtokenizer"},
                "dataset": {
                    "name": "xtokenizer_bridge",
                    "repo_id": DATASET_REPO_ID,
                    "dataset_revision": DATASET_REVISION,
                    "robot_type": 0,
                    "robot_type_name": "WidowX",
                    "embodiment_registry": {"WidowX": 0, "Franka": 1},
                    "action_layout": {
                        "convention": "absolute",
                        "dof_dims": DOF_DIMS,
                        "dof_order": list(DOF_DIMS),
                    },
                    "canonical_stats_identity": _training_data_identity()[
                        "canonical_stats_identity"
                    ],
                    "vl_sidecar_identity": _training_data_identity()[
                        "vl_sidecar_identity"
                    ],
                },
            }
        )
    )
    return checkpoint_dir, fixed_actions, fixed_anchor, expected


def _manifest(artifact_dir: Path) -> dict[str, Any]:
    return json.loads((artifact_dir / "manifest.json").read_text())


def _write_manifest(artifact_dir: Path, manifest: dict[str, Any]) -> None:
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_export_round_trip_is_deterministic_and_training_private_free(
    tmp_path: Path,
) -> None:
    checkpoint_dir, actions, anchor, expected = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )

    reloaded = load_exported_xtokenizer(artifact_dir)
    reloaded_again = load_exported_xtokenizer(artifact_dir)
    actual = reloaded.encode(actions, anchor=anchor, robot_type=torch.tensor([0]))
    repeated = reloaded_again.encode(
        actions,
        anchor=anchor,
        robot_type=torch.tensor([0]),
    )

    assert actual.shape == (1, 8, 4)
    assert actual.dtype == torch.long
    assert bool((actual >= 0).all())
    assert bool((actual < 8).all())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(repeated, actual)

    manifest = _manifest(artifact_dir)
    assert manifest["format_version"] == 3
    assert manifest["training_data_identity"] == _training_data_identity()
    assert manifest["action_horizon"] == 32
    assert manifest["compression_ratio"] == 4
    assert manifest["codebook_size"] == 8
    assert manifest["num_quantizers"] == 4
    assert manifest["training_private_data"] is False
    assert manifest["dof_layout"] == {
        "convention": "absolute",
        "dof_dims": DOF_DIMS,
        "dof_order": list(DOF_DIMS),
    }
    assert manifest["embodiment_registry"] == {"WidowX": 0, "Franka": 1}
    assert manifest["action_semantics"] == {
        "public_action": "absolute",
        "tokenized_action": "observation_relative",
        "observation_anchor": "absolute",
        "relative_reference": "single_pre_chunk_observation",
    }
    assert manifest["normalization"]["modes"] == {
        "action": "min_max",
        "observation.state": "min_max",
    }
    canonical_identity = _training_data_identity()["canonical_stats_identity"]
    assert manifest["normalization"]["kind"] == "xtokenizer_dual_role_v1"
    assert manifest["normalization"]["roles"] == {
        "predict_action": {
            "key": "action",
            "space": "observation_relative",
        },
        "observation_state": {
            "key": "observation.state",
            "space": "absolute",
        },
    }
    assert manifest["normalization"]["identity"] == {
        "format_version": 3,
        "content_fingerprint": canonical_identity["content_fingerprint"],
        "stats_horizon": canonical_identity["normalization"]["stats_horizon"],
    }
    checkpoint_normalization = torch.load(
        checkpoint_dir / "pretrained_model" / "norm_stats.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint_normalization["normalization_kind"] == ("xtokenizer_dual_role_v1")
    assert reloaded.normalize_action is not None
    assert reloaded.normalize_observation is not None
    for key, role, normalizer in (
        ("action", "predict_action", reloaded.normalize_action),
        (
            "observation.state",
            "observation_state",
            reloaded.normalize_observation,
        ),
    ):
        for stat_name in ("min", "max"):
            expected_stat = checkpoint_normalization["stats"][key][stat_name]
            assert torch.equal(normalizer.stats[key][stat_name], expected_stat)
            assert torch.equal(
                expected_stat, _normalization_statistics()[role][stat_name]
            )
    assert manifest["model_config"]["codebook_size"] == 8
    assert len(manifest["source_checkpoint_sha256"]) == 64
    assert len(manifest["artifact_sha256"]) == 64

    payload = torch.load(
        artifact_dir / "xtokenizer_inference.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert payload["metadata"] == {
        key: manifest[key]
        for key in (
            "source_checkpoint_sha256",
            "action_semantics",
            "dof_layout",
            "embodiment_registry",
            "normalization",
            "policy_io",
            "model_config",
            "training_data_identity",
        )
    }
    state_dict = payload["state_dict"]
    assert {key.split(".", 1)[0] for key in state_dict} == {
        "encoder",
        "quantizer",
        "decoder",
    }
    assert state_dict["quantizer.codebooks"].shape == (4, 8, 16)
    assert state_dict["quantizer.codebooks"].dtype == torch.float32


def test_latest_checkpoint_export_applies_saved_ema_weights(tmp_path: Path) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    model_dir = checkpoint_dir / "pretrained_model"
    raw_codec = XTokenizerCodec.from_pretrained(model_dir, local_files_only=True)
    ema = EMAModel(raw_codec)
    for shadow in ema.shadow.values():
        shadow.add_(0.25)
    state_dir = checkpoint_dir / "training_state"
    state_dir.mkdir()
    torch.save(ema.state_dict(), state_dir / "ema.pt")

    parameter_name = next(name for name in ema.shadow if name.startswith("encoder."))
    raw_parameter = raw_codec.state_dict()[parameter_name].detach().clone()
    expected_ema_parameter = ema.shadow[parameter_name].detach().clone()

    artifact_dir = export_xtokenizer_checkpoint(
        model_dir,
        tmp_path / "artifact",
    )

    payload = torch.load(
        artifact_dir / "xtokenizer_inference.pt",
        map_location="cpu",
        weights_only=True,
    )
    exported_parameter = payload["state_dict"][parameter_name]
    torch.testing.assert_close(exported_parameter, expected_ema_parameter)
    assert not torch.equal(exported_parameter, raw_parameter)


def test_source_digest_changes_when_adjacent_ema_bytes_change(tmp_path: Path) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    model_dir = checkpoint_dir / "pretrained_model"
    raw_codec = XTokenizerCodec.from_pretrained(model_dir, local_files_only=True)
    ema = EMAModel(raw_codec)
    state_dir = checkpoint_dir / "training_state"
    state_dir.mkdir()
    torch.save(ema.state_dict(), state_dir / "ema.pt")
    first = export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "first")

    next(iter(ema.shadow.values())).add_(0.5)
    torch.save(ema.state_dict(), state_dir / "ema.pt")
    second = export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "second")

    assert (
        _manifest(first)["source_checkpoint_sha256"]
        != _manifest(second)["source_checkpoint_sha256"]
    )


def test_export_loads_the_same_ema_bytes_bound_by_the_source_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    model_dir = checkpoint_dir / "pretrained_model"
    raw_codec = XTokenizerCodec.from_pretrained(model_dir, local_files_only=True)
    original_ema = EMAModel(raw_codec)
    replacement_ema = EMAModel(raw_codec)
    for shadow in original_ema.shadow.values():
        shadow.add_(0.25)
    for shadow in replacement_ema.shadow.values():
        shadow.add_(0.75)

    state_dir = checkpoint_dir / "training_state"
    state_dir.mkdir()
    ema_path = state_dir / "ema.pt"
    parked_path = state_dir / "ema.original.pt"
    replacement_path = state_dir / "ema.replacement.pt"
    torch.save(original_ema.state_dict(), ema_path)
    torch.save(replacement_ema.state_dict(), replacement_path)

    baseline_dir = export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "baseline")
    baseline_manifest = _manifest(baseline_dir)
    baseline_payload = torch.load(
        baseline_dir / "xtokenizer_inference.pt",
        map_location="cpu",
        weights_only=True,
    )

    load_started = threading.Event()
    replacement_ready = threading.Event()
    load_finished = threading.Event()
    source_restored = threading.Event()

    def _replace_source_ema_while_loading() -> None:
        assert load_started.wait(timeout=10)
        os.replace(ema_path, parked_path)
        os.replace(replacement_path, ema_path)
        replacement_ready.set()
        assert load_finished.wait(timeout=10)
        os.replace(ema_path, replacement_path)
        os.replace(parked_path, ema_path)
        source_restored.set()

    replacement_thread = threading.Thread(
        target=_replace_source_ema_while_loading,
        daemon=True,
    )
    replacement_thread.start()
    inference = importlib.import_module("praxis.codecs.latents.xtokenizer.inference")
    original_torch_load = inference.torch.load

    def _load_during_replacement(*args: Any, **kwargs: Any) -> Any:
        source = args[0]
        if not isinstance(source, (str, os.PathLike)) or Path(source).name != "ema.pt":
            return original_torch_load(*args, **kwargs)
        load_started.set()
        assert replacement_ready.wait(timeout=10)
        try:
            return original_torch_load(*args, **kwargs)
        finally:
            load_finished.set()
            assert source_restored.wait(timeout=10)

    monkeypatch.setattr(inference.torch, "load", _load_during_replacement)
    artifact_dir = export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "artifact")
    replacement_thread.join(timeout=10)
    assert not replacement_thread.is_alive()

    manifest = _manifest(artifact_dir)
    payload = original_torch_load(
        artifact_dir / "xtokenizer_inference.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert (
        manifest["source_checkpoint_sha256"]
        == baseline_manifest["source_checkpoint_sha256"]
    )
    for name, expected in baseline_payload["state_dict"].items():
        torch.testing.assert_close(payload["state_dict"][name], expected)


def test_latest_checkpoint_export_rejects_symlinked_ema_state(tmp_path: Path) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    raw_codec = XTokenizerCodec.from_pretrained(
        checkpoint_dir / "pretrained_model",
        local_files_only=True,
    )
    external_ema = tmp_path / "external-ema.pt"
    torch.save(EMAModel(raw_codec).state_dict(), external_ema)
    state_dir = checkpoint_dir / "training_state"
    state_dir.mkdir()
    (state_dir / "ema.pt").symlink_to(external_ema)

    with pytest.raises(ValueError, match="EMA.*symbolic link"):
        export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "artifact")


def test_export_rejects_model_and_training_action_layout_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    train_config = checkpoint_dir / "pretrained_model" / "train_config.yaml"
    config = OmegaConf.load(train_config)
    config.dataset.action_layout.dof_order = [
        "follow_left_gripper",
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation_6D",
    ]
    config.dataset.canonical_stats_identity.action_layout.dof_order = list(
        config.dataset.action_layout.dof_order
    )
    canonical_identity = OmegaConf.to_container(
        config.dataset.canonical_stats_identity,
        resolve=True,
    )
    assert isinstance(canonical_identity, dict)
    canonical_identity.pop("content_fingerprint")
    config.dataset.canonical_stats_identity.content_fingerprint = hashlib.sha256(
        json.dumps(
            canonical_identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    train_config.write_text(OmegaConf.to_yaml(config))

    with pytest.raises(ValueError, match="model config action layout.*training"):
        export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "artifact")


def test_source_digest_changes_when_checkpoint_bytes_change(tmp_path: Path) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    first = export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "first")
    train_config = checkpoint_dir / "pretrained_model" / "train_config.yaml"
    train_config.write_text(train_config.read_text() + "# same semantics, new bytes\n")
    second = export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "second")

    assert (
        _manifest(first)["source_checkpoint_sha256"]
        != _manifest(second)["source_checkpoint_sha256"]
    )


def test_loader_rejects_manifest_only_source_digest_replacement(
    tmp_path: Path,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    manifest = _manifest(artifact_dir)
    manifest["source_checkpoint_sha256"] = "0" * 64
    _write_manifest(artifact_dir, manifest)

    with pytest.raises(ValueError, match="payload metadata"):
        load_exported_xtokenizer(artifact_dir)


def test_loader_rejects_source_digest_payload_tamper_without_new_artifact_digest(
    tmp_path: Path,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    artifact_path = artifact_dir / "xtokenizer_inference.pt"
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    replacement = "0" * 64
    payload["metadata"]["source_checkpoint_sha256"] = replacement
    torch.save(payload, artifact_path)
    manifest = _manifest(artifact_dir)
    manifest["source_checkpoint_sha256"] = replacement
    _write_manifest(artifact_dir, manifest)

    with pytest.raises(ValueError, match="artifact SHA256"):
        load_exported_xtokenizer(artifact_dir)


def test_direct_export_records_explicit_no_normalization_mode(tmp_path: Path) -> None:
    codec = XTokenizerCodec(config=_tiny_config())
    codec.quantizer.initialized.fill_(True)

    artifact_dir = codec.export_inference_artifact(tmp_path / "artifact")

    assert _manifest(artifact_dir)["normalization"] == {
        "modes": {"action": "none", "observation.state": "none"},
        "stats": {"action": {}, "observation.state": {}},
    }
    assert isinstance(load_exported_xtokenizer(artifact_dir), XTokenizerCodec)


def test_export_rejects_canonical_identity_without_marked_normalization(
    tmp_path: Path,
) -> None:
    codec = XTokenizerCodec(config=_tiny_config())
    codec.quantizer.initialized.fill_(True)
    output = tmp_path / "artifact"

    with pytest.raises(
        ValueError,
        match="[Cc]anonical.*normalization|normalization.*identity",
    ):
        codec.export_inference_artifact(
            output,
            dof_layout={"convention": "absolute", "dof_dims": DOF_DIMS},
            embodiment_registry={"WidowX": 0},
            training_data_identity=_training_data_identity(),
        )

    assert not output.exists()


def test_paper_normalization_requires_canonical_identity() -> None:
    codec = XTokenizerCodec(config=_tiny_config())
    role_stats = _normalization_statistics()

    with pytest.raises(ValueError, match="canonical.*identity"):
        codec.set_xtokenizer_normalization_stats(
            predict_stats={"action": role_stats["predict_action"]},
            observation_stats={
                "observation.state": role_stats["observation_state"],
            },
            canonical_stats_identity=None,
            stats_horizon=32,
        )


def test_paper_export_load_accepts_codecs_trainer_action_only_policy_io(
    tmp_path: Path,
) -> None:
    codec = XTokenizerCodec(config=_tiny_config(dof_dims=DOF_DIMS))
    codec.quantizer.initialized.fill_(True)
    codec.set_io(_codecs_trainer_policy_io())
    role_stats = _normalization_statistics()
    codec.set_xtokenizer_normalization_stats(
        predict_stats={"action": role_stats["predict_action"]},
        observation_stats={
            "observation.state": role_stats["observation_state"],
        },
        canonical_stats_identity=_training_data_identity()["canonical_stats_identity"],
        stats_horizon=32,
    )

    artifact_dir = codec.export_inference_artifact(
        tmp_path / "artifact",
        dof_layout={"convention": "absolute", "dof_dims": DOF_DIMS},
        embodiment_registry={"WidowX": 0},
        training_data_identity=_training_data_identity(),
    )
    loaded = load_exported_xtokenizer(artifact_dir)

    assert loaded.io == _codecs_trainer_policy_io()
    assert loaded.normalize_observation is not None
    assert torch.equal(
        loaded.normalize_observation.stats["observation.state"]["min"],
        role_stats["observation_state"]["min"],
    )


def test_export_failure_preserves_previous_artifact_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    output_dir = export_xtokenizer_checkpoint(checkpoint_dir, tmp_path / "artifact")
    previous_manifest = (output_dir / "manifest.json").read_bytes()
    inference = importlib.import_module("praxis.codecs.latents.xtokenizer.inference")

    def _fail_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected artifact write failure")

    monkeypatch.setattr(inference.torch, "save", _fail_save)
    with pytest.raises(OSError, match="injected artifact write failure"):
        export_xtokenizer_checkpoint(checkpoint_dir, output_dir)

    assert (output_dir / "manifest.json").read_bytes() == previous_manifest
    assert not list(tmp_path.glob(".artifact.tmp-*"))
    assert not (tmp_path / ".artifact.prev").exists()


ManifestMutation = Callable[[dict[str, Any]], None]


def _set_path_traversal(manifest: dict[str, Any]) -> None:
    manifest["artifact_file"] = "../xtokenizer_inference.pt"


def _remove_registry(manifest: dict[str, Any]) -> None:
    del manifest["embodiment_registry"]


def _break_action_layout(manifest: dict[str, Any]) -> None:
    manifest["dof_layout"]["dof_dims"]["follow_left_gripper"] = 2


def _break_normalization(manifest: dict[str, Any]) -> None:
    manifest["normalization"]["stats"]["action"]["min"] = [0.0]


def _break_model_geometry(manifest: dict[str, Any]) -> None:
    manifest["model_config"]["num_quantizers"] = 3


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda manifest: manifest.update(format_version=2), "format_version"),
        (lambda manifest: manifest.update(compression_ratio=2), "compression_ratio"),
        (lambda manifest: manifest.update(training_private_data=True), "private"),
        (_set_path_traversal, "artifact_file"),
        (_remove_registry, "embodiment_registry"),
        (_break_action_layout, "action layout"),
        (_break_normalization, "normalization"),
        (_break_model_geometry, "model_config"),
    ],
)
def test_loader_fails_closed_on_malformed_or_mismatched_manifest(
    tmp_path: Path,
    mutation: ManifestMutation,
    match: str,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    manifest = _manifest(artifact_dir)
    mutation(manifest)
    _write_manifest(artifact_dir, manifest)

    with pytest.raises((TypeError, ValueError), match=match):
        load_exported_xtokenizer(artifact_dir)


def _reorder_dof_layout(manifest: dict[str, Any]) -> None:
    manifest["dof_layout"]["dof_order"] = list(
        reversed(manifest["dof_layout"]["dof_order"])
    )


def _rename_dof_layout_entry(manifest: dict[str, Any]) -> None:
    layout = manifest["dof_layout"]
    old_name = layout["dof_order"][0]
    new_name = f"{old_name}_renamed"
    layout["dof_order"][0] = new_name
    layout["dof_dims"][new_name] = layout["dof_dims"].pop(old_name)


def _change_registry_id(manifest: dict[str, Any]) -> None:
    registry = manifest["embodiment_registry"]
    registry["WidowX"], registry["Franka"] = registry["Franka"], registry["WidowX"]


def _change_normalization_value(manifest: dict[str, Any]) -> None:
    manifest["normalization"]["stats"]["action"]["min"][0] = -1.5


def _change_normalization_mode(manifest: dict[str, Any]) -> None:
    manifest["normalization"] = {
        "modes": {"action": "mean_std", "observation.state": "mean_std"},
        "stats": {
            "action": {
                "mean": [0.0] * 10,
                "std": [1.0] * 10,
            },
            "observation.state": {
                "mean": [0.0] * 10,
                "std": [1.0] * 10,
            },
        },
    }
    manifest["policy_io"]["action_feature"]["norm_mode"] = "mean_std"
    manifest["policy_io"]["obs_features"][0]["norm_mode"] = "mean_std"


def _change_policy_io(manifest: dict[str, Any]) -> None:
    manifest["policy_io"]["fps"] = 31


def _change_model_config(manifest: dict[str, Any]) -> None:
    manifest["model_config"]["dropout"] = 0.1


def _change_training_data_identity(manifest: dict[str, Any]) -> None:
    identity = manifest["training_data_identity"]
    identity["dataset_revision"] = "b" * 40
    identity["canonical_stats_identity"]["dataset_revision"] = "b" * 40
    identity["vl_sidecar_identity"]["dataset_revision"] = "b" * 40
    identity["vl_sidecar_identity"]["generation_fingerprint"] = (
        vl_sidecar_generation_fingerprint(identity["vl_sidecar_identity"])
    )


@pytest.mark.parametrize(
    "mutation",
    [
        _reorder_dof_layout,
        _rename_dof_layout_entry,
        _change_registry_id,
        _change_normalization_value,
        _change_normalization_mode,
        _change_policy_io,
        _change_model_config,
        _change_training_data_identity,
    ],
)
def test_loader_rejects_semantic_manifest_tampering_bound_to_payload(
    tmp_path: Path,
    mutation: ManifestMutation,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    manifest = _manifest(artifact_dir)
    mutation(manifest)
    _write_manifest(artifact_dir, manifest)

    with pytest.raises(
        ValueError,
        match=("payload metadata|action layout|normalization|[Cc]anonical statistics"),
    ):
        load_exported_xtokenizer(artifact_dir)


def test_loader_rejects_path_field_in_training_data_identity(
    tmp_path: Path,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    manifest = _manifest(artifact_dir)
    manifest["training_data_identity"]["canonical_stats_path"] = "/tmp/stats.json"
    _write_manifest(artifact_dir, manifest)

    with pytest.raises(ValueError, match="Training data identity fields"):
        load_exported_xtokenizer(artifact_dir)


def test_loader_accepts_canonical_manifest_with_reordered_object_keys(
    tmp_path: Path,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    _write_manifest(artifact_dir, _manifest(artifact_dir))

    assert isinstance(load_exported_xtokenizer(artifact_dir), XTokenizerCodec)


NormalizationMutation = Callable[[dict[str, Any]], None]


def _none_with_stats(manifest: dict[str, Any]) -> None:
    manifest["normalization"] = {
        "modes": {"action": "none", "observation.state": "none"},
        "stats": {
            "action": {
                "min": [-1.0] * 10,
                "max": [1.0] * 10,
            },
            "observation.state": {},
        },
    }
    manifest["policy_io"]["action_feature"]["norm_mode"] = "none"
    manifest["policy_io"]["obs_features"][0]["norm_mode"] = "none"


def _min_max_with_equal_bound(manifest: dict[str, Any]) -> None:
    manifest["normalization"]["stats"]["action"]["max"][0] = -2.0


def _min_max_with_extra_stat(manifest: dict[str, Any]) -> None:
    manifest["normalization"]["stats"]["action"]["mean"] = [0.0] * 10


def _mean_std_with_nonpositive_std(manifest: dict[str, Any]) -> None:
    manifest["normalization"] = {
        "modes": {"action": "mean_std", "observation.state": "mean_std"},
        "stats": {
            "action": {
                "mean": [0.0] * 10,
                "std": [1.0] * 9 + [0.0],
            },
            "observation.state": {
                "mean": [0.0] * 10,
                "std": [1.0] * 10,
            },
        },
    }
    manifest["policy_io"]["action_feature"]["norm_mode"] = "mean_std"
    manifest["policy_io"]["obs_features"][0]["norm_mode"] = "mean_std"


def _normalization_with_wrong_action_key(manifest: dict[str, Any]) -> None:
    manifest["normalization"] = {
        "modes": {
            "alternate_action": "min_max",
            "observation.state": "min_max",
        },
        "stats": {
            "alternate_action": {
                "min": [-2.0] * 10,
                "max": [2.0] * 10,
            },
            "observation.state": {
                "min": [-2.0] * 10,
                "max": [2.0] * 10,
            },
        },
    }


def _normalization_without_predict_role(manifest: dict[str, Any]) -> None:
    del manifest["normalization"]["roles"]["predict_action"]


def _normalization_without_observation_role(manifest: dict[str, Any]) -> None:
    del manifest["normalization"]["roles"]["observation_state"]


def _normalization_with_swapped_role_spaces(manifest: dict[str, Any]) -> None:
    roles = manifest["normalization"]["roles"]
    roles["predict_action"], roles["observation_state"] = (
        roles["observation_state"],
        roles["predict_action"],
    )


def _normalization_with_swapped_numeric_roles(manifest: dict[str, Any]) -> None:
    stats = manifest["normalization"]["stats"]
    stats["action"], stats["observation.state"] = (
        stats["observation.state"],
        stats["action"],
    )


def _normalization_without_identity(manifest: dict[str, Any]) -> None:
    del manifest["normalization"]["identity"]


def _normalization_with_bad_fingerprint(manifest: dict[str, Any]) -> None:
    manifest["normalization"]["identity"]["content_fingerprint"] = "0" * 64


def _normalization_with_wrong_horizon(manifest: dict[str, Any]) -> None:
    manifest["normalization"]["identity"]["stats_horizon"] = 16


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_none_with_stats, "NONE.*no statistics"),
        (_min_max_with_equal_bound, "min < max"),
        (_min_max_with_extra_stat, "exactly.*max.*min"),
        (_mean_std_with_nonpositive_std, "std > 0"),
        (_normalization_with_wrong_action_key, "normalization keys"),
        (_normalization_without_predict_role, "normalization roles"),
        (_normalization_without_observation_role, "normalization roles"),
        (_normalization_with_swapped_role_spaces, "normalization roles"),
        (_normalization_with_swapped_numeric_roles, "canonical.*statistics"),
        (_normalization_without_identity, "normalization.*identity"),
        (_normalization_with_bad_fingerprint, "content_fingerprint"),
        (_normalization_with_wrong_horizon, "stats_horizon"),
    ],
)
def test_loader_enforces_exact_normalization_contract(
    tmp_path: Path,
    mutation: NormalizationMutation,
    match: str,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    manifest = _manifest(artifact_dir)
    mutation(manifest)
    _write_manifest(artifact_dir, manifest)

    with pytest.raises(ValueError, match=match):
        load_exported_xtokenizer(artifact_dir)


def test_loader_rejects_artifact_digest_tampering(tmp_path: Path) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    with (artifact_dir / "xtokenizer_inference.pt").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="artifact SHA256"):
        load_exported_xtokenizer(artifact_dir)


def test_loader_rejects_undeclared_artifact_files(tmp_path: Path) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    torch.save({"optimizer": "private"}, artifact_dir / "training_state.pt")

    with pytest.raises(ValueError, match="undeclared files"):
        load_exported_xtokenizer(artifact_dir)


def test_loader_rejects_training_head_state_even_with_updated_digest(
    tmp_path: Path,
) -> None:
    checkpoint_dir, _, _, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    artifact_path = artifact_dir / "xtokenizer_inference.pt"
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    payload["state_dict"]["mam_head.predictor.weight"] = torch.zeros(8, 16)
    torch.save(payload, artifact_path)

    inference = importlib.import_module("praxis.codecs.latents.xtokenizer.inference")
    manifest = _manifest(artifact_dir)
    manifest["artifact_sha256"] = inference.sha256_file(artifact_path)
    _write_manifest(artifact_dir, manifest)

    with pytest.raises(ValueError, match="training-only state"):
        load_exported_xtokenizer(artifact_dir)


def test_bridge_generator_loads_export_through_native_cache_adapter(
    tmp_path: Path,
) -> None:
    checkpoint_dir, actions, anchor, _ = _train_and_save_checkpoint(tmp_path)
    artifact_dir = export_xtokenizer_checkpoint(
        checkpoint_dir,
        tmp_path / "artifact",
    )
    generator = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )

    tokenizer = generator._load_cache_tokenizer(
        artifact_dir,
        device="cpu",
        statistics_path=tmp_path / "unused-statistics.json",
    )
    assert not hasattr(tokenizer, "get_normalizer")
    assert not hasattr(tokenizer, "_normalizer_parameters")
    assert not hasattr(tokenizer, "normalize_action")
    result = tokenizer.encode(
        actions,
        absolute_anchor=anchor,
        dof_mask=torch.ones_like(actions, dtype=torch.bool),
        robot_type="WidowX",
        token_order="time_major",
    )

    assert tokenizer.dof_dims == DOF_DIMS
    assert tokenizer.device == torch.device("cpu")
    assert result["indices"].shape == (1, 8, 4)
    direct = load_exported_xtokenizer(artifact_dir)
    torch.testing.assert_close(
        result["indices"],
        direct.encode(actions, anchor=anchor, robot_type=torch.tensor([0])),
    )


def _write_tiny_bridge_dataset(
    root: Path,
    *,
    dataset_repo_id: str = DATASET_REPO_ID,
    dataset_revision: str = DATASET_REVISION,
) -> None:
    frame_count = 66
    states = np.zeros((frame_count, 7), dtype=np.float32)
    states[:, 0] = np.linspace(0.0, 0.2, frame_count)
    states[:, 6] = 0.5
    actions = np.zeros((frame_count, 7), dtype=np.float32)
    actions[:, 0] = 0.002
    actions[:, 5] = 0.001
    actions[:, 6] = np.arange(frame_count) % 2
    table = pa.table(
        {
            "observation.state": pa.array(
                states.tolist(),
                type=pa.list_(pa.float32(), 7),
            ),
            "action": pa.array(
                actions.tolist(),
                type=pa.list_(pa.float32(), 7),
            ),
            "episode_index": pa.array([0] * 33 + [1] * 33, type=pa.int64()),
            "index": pa.array(np.arange(frame_count), type=pa.int64()),
        }
    )
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    pq.write_table(table, data_dir / "part-000.parquet")
    meta_dir = root / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_text(
        json.dumps({"total_episodes": 2, "total_frames": frame_count}) + "\n"
    )
    (meta_dir / "stats.json").write_text("{}\n")
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["bridge task"]),
            }
        ),
        meta_dir / "tasks.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 1], type=pa.int64()),
                "dataset_from_index": pa.array([0, 33], type=pa.int64()),
                "dataset_to_index": pa.array([33, 66], type=pa.int64()),
            }
        ),
        episodes_dir / "file-000.parquet",
    )
    write_lerobot_cache_provenance(
        root=root,
        repo_id=dataset_repo_id,
        revision=dataset_revision,
    )


def _export_tiny_native_artifact(
    root: Path,
    *,
    dof_dims: dict[str, int] | None = None,
) -> Path:
    resolved_dof_dims = DOF_DIMS if dof_dims is None else dof_dims
    codec = XTokenizerCodec(config=_tiny_config(dof_dims=resolved_dof_dims))
    codec.quantizer.initialized.fill_(True)
    codec.set_io(_policy_io())
    role_stats = _normalization_statistics()
    codec.set_xtokenizer_normalization_stats(
        predict_stats={"action": role_stats["predict_action"]},
        observation_stats={
            "observation.state": role_stats["observation_state"],
        },
        canonical_stats_identity=_training_data_identity(resolved_dof_dims)[
            "canonical_stats_identity"
        ],
        stats_horizon=32,
    )
    return codec.export_inference_artifact(
        root,
        dof_layout={
            "convention": "absolute",
            "dof_dims": resolved_dof_dims,
        },
        embodiment_registry={"WidowX": 0},
        training_data_identity=_training_data_identity(resolved_dof_dims),
    )


def _run_native_cache_cli(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper_after_train: bool = False,
    dof_dims: dict[str, int] | None = None,
    dataset_repo_id: str = DATASET_REPO_ID,
    dataset_revision: str = DATASET_REVISION,
    preexisting_output: bool = False,
) -> Path:
    dataset_root = tmp_path / "dataset"
    _write_tiny_bridge_dataset(
        dataset_root,
        dataset_repo_id=dataset_repo_id,
        dataset_revision=dataset_revision,
    )
    artifact_dir = _export_tiny_native_artifact(
        tmp_path / "artifact",
        dof_dims=dof_dims,
    )
    output_root = tmp_path / "cache"
    if preexisting_output:
        output_root.mkdir()
        (output_root / "sentinel.txt").write_text("keep\n")
    generator = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )
    if tamper_after_train:
        original_generate = generator._generate_split_cache

        def _generate_then_tamper(**kwargs: Any) -> None:
            original_generate(**kwargs)
            if kwargs["split_name"] == "train":
                (Path(kwargs["output_root"]).parent / "normalization.json").write_text(
                    "{}\n"
                )

        monkeypatch.setattr(generator, "_generate_split_cache", _generate_then_tamper)

    original_import = builtins.__import__

    def _reject_external_xtokenizer(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "xtokenizer" or name.startswith("xtokenizer."):
            raise AssertionError(f"native cache imported external module {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _reject_external_xtokenizer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_bridge_xtokenizer_cache.py",
            "--dataset-root",
            str(dataset_root),
            "--checkpoint",
            str(artifact_dir),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
            "--batch-size",
            "2",
            "--stats-batch-size",
            "2",
            "--val-ratio",
            "0.5",
            "--split-seed",
            "17",
        ],
    )
    generator.main()
    return output_root


@pytest.mark.parametrize(
    ("dataset_repo_id", "dataset_revision", "match"),
    [
        (DATASET_REPO_ID, "b" * 40, "provenance revision"),
        ("other/bridge", DATASET_REVISION, "provenance repository"),
    ],
)
def test_native_cache_rejects_local_identity_before_loading_data_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_repo_id: str,
    dataset_revision: str,
    match: str,
) -> None:
    generator = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )
    load_calls = 0
    original_load = generator._load_bridge_columns

    def _record_load(dataset_root: Path):
        nonlocal load_calls
        load_calls += 1
        return original_load(dataset_root)

    monkeypatch.setattr(generator, "_load_bridge_columns", _record_load)

    with pytest.raises(ValueError, match=match):
        _run_native_cache_cli(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            dataset_repo_id=dataset_repo_id,
            dataset_revision=dataset_revision,
        )

    assert load_calls == 0
    assert not (tmp_path / "cache").exists()


def test_native_cache_rejects_artifact_directory_swap_before_data_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    _write_tiny_bridge_dataset(dataset_root)
    artifact_dir = _export_tiny_native_artifact(tmp_path / "artifact")
    replacement_dir = _export_tiny_native_artifact(tmp_path / "replacement")
    retired_dir = tmp_path / "retired"
    output_root = tmp_path / "cache"
    inference = importlib.import_module("praxis.codecs.latents.xtokenizer.inference")
    generator = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )
    original_compatibility_load = inference.load_exported_xtokenizer
    original_torch_load = inference.torch.load
    swapped = False
    data_load_calls = 0

    def _swap_published_directory() -> None:
        nonlocal swapped
        os.replace(artifact_dir, retired_dir)
        os.replace(replacement_dir, artifact_dir)
        swapped = True

    def _swap_before_legacy_second_read(path: Path):
        if not swapped:
            _swap_published_directory()
        return original_compatibility_load(path)

    def _swap_after_snapshot_open(*args: Any, **kwargs: Any):
        if not swapped:
            _swap_published_directory()
        return original_torch_load(*args, **kwargs)

    def _reject_data_load(dataset_root: Path):
        nonlocal data_load_calls
        data_load_calls += 1
        raise AssertionError(f"data must not be loaded from {dataset_root}")

    monkeypatch.setattr(
        inference,
        "load_exported_xtokenizer",
        _swap_before_legacy_second_read,
    )
    monkeypatch.setattr(inference.torch, "load", _swap_after_snapshot_open)
    monkeypatch.setattr(generator, "_load_bridge_columns", _reject_data_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_bridge_xtokenizer_cache.py",
            "--dataset-root",
            str(dataset_root),
            "--checkpoint",
            str(artifact_dir),
            "--output-root",
            str(output_root),
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="changed while loading"):
        generator.main()

    assert swapped
    assert data_load_calls == 0
    assert not output_root.exists()


def test_native_cache_cli_is_self_contained_and_records_exact_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = _run_native_cache_cli(monkeypatch=monkeypatch, tmp_path=tmp_path)

    normalization_path = output_root / "normalization.json"
    normalization = json.loads(normalization_path.read_text())
    artifact_normalization = _manifest(tmp_path / "artifact")["normalization"]
    artifact_manifest = _manifest(tmp_path / "artifact")
    training_identity = _training_data_identity()
    assert normalization == artifact_normalization
    assert not (output_root / "statistics.json").exists()
    digest = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )._sha256_file(normalization_path)
    for split_name in ("train", "val"):
        manifest = json.loads((output_root / split_name / "manifest.json").read_text())
        assert manifest["format_version"] == 2
        assert manifest["action_semantics"] == {
            "public_action": "absolute",
            "tokenized_action": "observation_relative",
            "observation_anchor": "absolute",
            "relative_reference": "single_pre_chunk_observation",
        }
        assert manifest["split_seed"] == 17
        assert manifest["val_ratio"] == 0.5
        assert manifest["normalization_source"] == "inference_artifact"
        assert manifest["normalization_sha256"] == digest
        assert manifest["normalization_file"] == "../normalization.json"
        assert manifest["dataset_repo_id"] == training_identity["dataset_repo_id"]
        assert manifest["dataset_revision"] == training_identity["dataset_revision"]
        assert (
            manifest["canonical_stats_identity"]
            == training_identity["canonical_stats_identity"]
        )
        assert (
            manifest["content_fingerprint"]
            == (training_identity["canonical_stats_identity"]["content_fingerprint"])
        )
        assert manifest["stats_horizon"] == 32
        assert manifest["stats_horizon"] == manifest["action_horizon"]
        assert (
            manifest["vl_sidecar_identity"] == training_identity["vl_sidecar_identity"]
        )
        assert manifest["artifact_sha256"] == artifact_manifest["artifact_sha256"]
        assert (
            manifest["source_checkpoint_sha256"]
            == artifact_manifest["source_checkpoint_sha256"]
        )
        assert "checkpoint_sha256" not in manifest
        assert "source_revision" not in manifest
        assert "hf_revision" not in manifest
        assert np.load(output_root / split_name / "tokens.npy").shape == (33, 8, 4)
        ActionTokenCache.open(output_root / split_name)


def test_native_cache_construction_rejects_disabled_normalization() -> None:
    generator = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )
    manifest = {
        "training_data_identity": _training_data_identity(),
        "normalization": {
            "kind": "xtokenizer_dual_role_v1",
            "modes": {"action": "none", "observation.state": "min_max"},
        },
    }

    with pytest.raises(ValueError, match="normalization.*NONE|NONE.*normalization"):
        generator._require_native_training_data_identity(manifest)


def test_native_cache_fails_closed_if_normalization_provenance_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="normalization SHA256"):
        _run_native_cache_cli(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            tamper_after_train=True,
        )


def test_native_cache_refuses_to_overwrite_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(FileExistsError, match="cache output.*already exists"):
        _run_native_cache_cli(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            preexisting_output=True,
        )

    assert (tmp_path / "cache" / "sentinel.txt").read_text() == "keep\n"
    assert not (tmp_path / "cache" / "normalization.json").exists()


def test_native_cache_publish_preserves_output_created_during_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "cache"
    sentinel = output_root / "concurrent-publisher.txt"
    original_rename = Path.rename
    injected_race = False

    def _inject_concurrent_publish(path: Path, target: Path) -> Path:
        nonlocal injected_race
        target = Path(target)
        if target == output_root and path.name.startswith(".cache.tmp-"):
            output_root.mkdir()
            sentinel.write_text("keep concurrent cache\n")
            injected_race = True
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", _inject_concurrent_publish)

    with pytest.raises(FileExistsError, match="cache output.*already exists"):
        _run_native_cache_cli(monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert injected_race
    assert sentinel.read_text() == "keep concurrent cache\n"
    assert not list(tmp_path.glob(".cache.tmp-*"))
    assert not list(tmp_path.glob(".cache.prev-*"))


def test_native_cache_failure_cleans_staging_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )
    original_generate = generator._generate_split_cache
    generated_splits: list[str] = []
    fail_once = True

    def _generate_then_fail_once(**kwargs: Any) -> None:
        nonlocal fail_once
        original_generate(**kwargs)
        generated_splits.append(kwargs["split_name"])
        if kwargs["split_name"] == "train" and fail_once:
            fail_once = False
            raise OSError("injected cache write failure")

    monkeypatch.setattr(generator, "_generate_split_cache", _generate_then_fail_once)

    with pytest.raises(OSError, match="injected cache write failure"):
        _run_native_cache_cli(monkeypatch=monkeypatch, tmp_path=tmp_path)

    output_root = tmp_path / "cache"
    assert generated_splits == ["train"]
    assert not output_root.exists()
    assert not list(tmp_path.glob(".cache.tmp-*"))
    assert not list(tmp_path.glob(".cache.prev-*"))

    generator.main()

    assert generated_splits == ["train", "train", "val"]
    assert (output_root / "summary.json").is_file()
    for split_name in ("train", "val"):
        ActionTokenCache.open(output_root / split_name)
    assert not list(tmp_path.glob(".cache.tmp-*"))
    assert not list(tmp_path.glob(".cache.prev-*"))


def test_native_cache_reordered_layout_matches_direct_named_slice_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = importlib.import_module(
        "scripts.tokenizer.generate_bridge_xtokenizer_cache"
    )
    encode_calls: list[tuple[Tensor, Tensor]] = []
    original_encode = XTokenizerCodec.encode

    def _record_encode(
        codec: XTokenizerCodec,
        absolute_actions: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        absolute_anchor = kwargs["anchor"]
        encode_calls.append(
            (
                absolute_actions.detach().cpu().clone(),
                absolute_anchor.detach().cpu().clone(),
            )
        )
        return original_encode(codec, absolute_actions, **kwargs)

    monkeypatch.setattr(XTokenizerCodec, "encode", _record_encode)
    output_root = _run_native_cache_cli(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        dof_dims=REORDERED_DOF_DIMS,
    )

    states, actions, episode_ids, _ = generator._load_bridge_columns(
        tmp_path / "dataset"
    )
    valid_starts = generator._valid_window_starts(
        episode_ids,
        horizon=generator.HORIZON,
    )
    train_episodes, val_episodes = generator.split_train_val_episodes(
        [0, 1],
        val_ratio=0.5,
        split_seed=17,
    )
    direct = load_exported_xtokenizer(tmp_path / "artifact")
    generation_calls = list(encode_calls)
    assert len(generation_calls) == 2
    absolute_actions = np.zeros((len(states), 10), dtype=np.float32)
    observation_states = np.zeros_like(absolute_actions)
    for episode_index in (0, 1):
        episode_rows = np.flatnonzero(episode_ids == episode_index)
        episode = bridge_xtokenizer.build_bridge_xtokenizer_episode(
            states[episode_rows],
            actions[episode_rows],
            REORDERED_DOF_DIMS,
        )
        absolute_actions[episode_rows] = episode.absolute_actions
        observation_states[episode_rows] = episode.observation_states_absolute

    for call_index, (split_name, selected_episodes) in enumerate(
        (("train", train_episodes), ("val", val_episodes))
    ):
        membership = generator._episode_membership(2, selected_episodes)
        rows = np.flatnonzero(membership[episode_ids])
        starts = valid_starts[membership[episode_ids[valid_starts]]]
        action_rows = starts[:, None] + np.arange(generator.HORIZON)[None, :]
        expected_actions = absolute_actions[action_rows]
        expected_anchors = observation_states[starts]
        seen_actions, seen_anchor = generation_calls[call_index]
        assert torch.equal(seen_actions, torch.from_numpy(expected_actions))
        assert torch.equal(seen_anchor, torch.from_numpy(expected_anchors))
        expected_tokens = direct.encode(
            torch.from_numpy(expected_actions),
            anchor=torch.from_numpy(expected_anchors),
            robot_type=torch.zeros(len(starts), dtype=torch.long),
        ).numpy()
        local_rows = np.searchsorted(rows, starts)
        actual_tokens = np.load(output_root / split_name / "tokens.npy")
        np.testing.assert_array_equal(actual_tokens[local_rows], expected_tokens)
        manifest = json.loads((output_root / split_name / "manifest.json").read_text())
        assert manifest["active_dofs"] == list(REORDERED_DOF_DIMS)
