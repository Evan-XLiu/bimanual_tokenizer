# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from praxis.serving.model_identity import (
    build_checkpoint_identity,
    build_policy_serving_identity,
    validate_robotwin_ar_bin_contract,
)


def _write_sidecars(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text("{}\n")
    (root / "policy_io.json").write_text("{}\n")
    (root / "norm_stats.pt").write_bytes(b"stats")


def _write_robotwin_policy(tmp_path: Path) -> Path:
    tokenizer = tmp_path / "bin" / "pretrained_model"
    tokenizer.parent.mkdir()
    _write_sidecars(tokenizer)
    (tokenizer / "config.json").write_text(
        json.dumps(
            {
                "name": "bin",
                "action_dim": 14,
                "action_horizon": 32,
                "num_bins": 256,
            }
        )
    )
    (tokenizer / "policy_io.json").write_text(
        json.dumps(
            {
                "obs_features": [],
                "action_feature": {"shape": [14]},
                "n_action_steps": 32,
            }
        )
    )
    (tokenizer / "model.safetensors").write_bytes(b"bin")

    policy = tmp_path / "policy" / "pretrained_model"
    policy.parent.mkdir()
    _write_sidecars(policy)
    (policy / "config.json").write_text(
        json.dumps(
            {
                "name": "bar_bin_qwen",
                "action_dim": 14,
                "action_horizon": 32,
                "n_action_steps": 16,
                "state_prefix_num_bins": 256,
                "use_lora": False,
                "freeze_vision_encoder": False,
                "vlm_model_name": "Qwen/Qwen3-VL-2B-Instruct",
                "action_tokenizer": {
                    "name": "bin",
                    "kind": "tokens",
                    "pretrained_path": str(tokenizer),
                },
            }
        )
    )
    (policy / "policy_io.json").write_text(
        json.dumps(
            {
                "obs_features": [
                    {"key": "observation.state", "shape": [14]},
                    {
                        "key": "observation.images.cam_high",
                        "shape": [3, 240, 320],
                    },
                    {
                        "key": "observation.images.cam_left_wrist",
                        "shape": [3, 240, 320],
                    },
                    {
                        "key": "observation.images.cam_right_wrist",
                        "shape": [3, 240, 320],
                    },
                ],
                "action_feature": {"shape": [14]},
                "n_action_steps": 32,
            }
        )
    )
    (policy / "model.safetensors").write_bytes(b"policy")
    return policy


def test_build_checkpoint_identity_covers_single_safetensors_and_sidecars(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "pretrained_model"
    _write_sidecars(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    identity = build_checkpoint_identity(checkpoint)

    assert identity["path"] == str(checkpoint.resolve())
    assert set(identity["files"]) == {
        "config.json",
        "policy_io.json",
        "norm_stats.pt",
        "model.safetensors",
    }
    assert len(identity["artifact_sha256"]) == 64


def test_build_checkpoint_identity_covers_index_and_every_shard(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    _write_sidecars(checkpoint)
    index = {
        "weight_map": {
            "a": "model-00001-of-00002.safetensors",
            "b": "model-00002-of-00002.safetensors",
        }
    }
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"two")

    identity = build_checkpoint_identity(checkpoint)

    assert set(identity["files"]) == {
        "config.json",
        "policy_io.json",
        "norm_stats.pt",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_build_checkpoint_identity_rejects_missing_sidecar(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    _write_sidecars(checkpoint)
    (checkpoint / "policy_io.json").unlink()
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    with pytest.raises(FileNotFoundError, match="policy_io.json"):
        build_checkpoint_identity(checkpoint)


def test_policy_serving_identity_covers_external_action_tokenizer(
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "bin" / "pretrained_model"
    tokenizer.parent.mkdir()
    _write_sidecars(tokenizer)
    (tokenizer / "model.safetensors").write_bytes(b"bin-v1")

    policy = tmp_path / "policy" / "pretrained_model"
    policy.parent.mkdir()
    _write_sidecars(policy)
    (policy / "config.json").write_text(
        json.dumps(
            {
                "action_tokenizer": {
                    "name": "bin",
                    "pretrained_path": str(tokenizer),
                }
            }
        )
    )
    (policy / "model.safetensors").write_bytes(b"policy")

    before = build_policy_serving_identity(policy)
    (tokenizer / "model.safetensors").write_bytes(b"bin-v2")
    after = build_policy_serving_identity(policy)

    assert before["checkpoint"]["artifact_sha256"] == after["checkpoint"][
        "artifact_sha256"
    ]
    assert before["dependencies"]["action_tokenizer"]["artifact_sha256"] != (
        after["dependencies"]["action_tokenizer"]["artifact_sha256"]
    )
    assert before["artifact_sha256"] != after["artifact_sha256"]


def test_robotwin_contract_accepts_full_ar_bin_h32_policy(tmp_path: Path) -> None:
    policy = _write_robotwin_policy(tmp_path)

    contract = validate_robotwin_ar_bin_contract(policy)

    assert contract["policy_name"] == "bar_bin_qwen"
    assert contract["action_horizon"] == 32
    assert contract["tokenizer_name"] == "bin"


def test_robotwin_contract_accepts_canonical_qwen_snapshot_path(tmp_path: Path) -> None:
    policy = _write_robotwin_policy(tmp_path)
    config_path = policy / "config.json"
    config = json.loads(config_path.read_text())
    config["vlm_model_name"] = (
        "/cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/" + "a" * 40
    )
    config_path.write_text(json.dumps(config))

    contract = validate_robotwin_ar_bin_contract(policy)

    assert contract["vlm_model_name"] == config["vlm_model_name"]


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-VL-4B-Instruct",
        "/cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/" + "a" * 40,
        "/cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/not-a-commit",
    ],
)
def test_robotwin_contract_rejects_other_qwen_models(
    tmp_path: Path, model_name: str
) -> None:
    policy = _write_robotwin_policy(tmp_path)
    config_path = policy / "config.json"
    config = json.loads(config_path.read_text())
    config["vlm_model_name"] = model_name
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Qwen3-VL-2B-Instruct"):
        validate_robotwin_ar_bin_contract(policy)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "bar_oat_qwen", "Policy config"),
        ("action_horizon", 8, "Policy config"),
        ("action_dim", 7, "Policy config"),
        ("state_prefix_num_bins", 128, "Policy config"),
        ("use_lora", True, "Policy config"),
        ("freeze_vision_encoder", True, "Policy config"),
    ],
)
def test_robotwin_contract_rejects_wrong_policy_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    policy = _write_robotwin_policy(tmp_path)
    config_path = policy / "config.json"
    config = json.loads(config_path.read_text())
    config[field] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        validate_robotwin_ar_bin_contract(policy)


def test_robotwin_contract_rejects_non_bin_tokenizer(tmp_path: Path) -> None:
    policy = _write_robotwin_policy(tmp_path)
    config_path = policy / "config.json"
    config = json.loads(config_path.read_text())
    config["action_tokenizer"]["name"] = "oat"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="Policy action tokenizer"):
        validate_robotwin_ar_bin_contract(policy)
