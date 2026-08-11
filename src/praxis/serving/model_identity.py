# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Stable identities for local policy checkpoints served during evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_REQUIRED_SIDECARS = ("config.json", "policy_io.json", "norm_stats.pt")
_SAFETENSORS_FILE = "model.safetensors"
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_ROBOTWIN_CAMERA_SHAPE = [3, 240, 320]
_ROBOTWIN_QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_files(root: Path) -> tuple[str, ...]:
    index_path = root / _SAFETENSORS_INDEX
    if not index_path.exists():
        if not (root / _SAFETENSORS_FILE).is_file():
            raise FileNotFoundError(
                f"Checkpoint {root} has neither {_SAFETENSORS_FILE} nor "
                f"{_SAFETENSORS_INDEX}."
            )
        return (_SAFETENSORS_FILE,)

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid safetensors index: {index_path}.") from exc
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Safetensors index has no weight_map: {index_path}.")
    shard_names = set(weight_map.values())
    if not all(isinstance(name, str) and name for name in shard_names):
        raise ValueError(f"Safetensors index has invalid shard names: {index_path}.")
    for name in shard_names:
        candidate = Path(name)
        if candidate.is_absolute() or candidate.parts != (name,):
            raise ValueError(
                f"Safetensors shard must be a root-level filename, got {name!r}."
            )
    return (_SAFETENSORS_INDEX, *sorted(shard_names))


def build_checkpoint_identity(checkpoint_path: str | Path) -> dict[str, Any]:
    """Hash every required serving artifact under a local pretrained_model dir."""
    root = Path(checkpoint_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Policy checkpoint directory not found: {root}.")

    filenames = (*_REQUIRED_SIDECARS, *_weight_files(root))
    files: dict[str, dict[str, int | str]] = {}
    for filename in filenames:
        path = root / filename
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Policy checkpoint artifact is missing or empty: {path}."
            )
        files[filename] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "path": str(root),
        "artifact_sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def _resolve_dependency_path(*, policy_root: Path, dependency_path: str) -> Path:
    candidate = Path(dependency_path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.exists():
        return candidate.resolve()
    for parent in (policy_root, *policy_root.parents):
        if (parent / ".git").exists():
            return (parent / candidate).resolve()
    return (policy_root / candidate).resolve()


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON metadata: {path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON metadata must be an object: {path}.")
    return payload


def _require_fields(
    payload: dict[str, Any], expected: dict[str, Any], *, label: str
) -> None:
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} violates the RoboTwin AR BIN contract: {mismatches}.")


def _is_robotwin_qwen_model(value: Any) -> bool:
    if value == _ROBOTWIN_QWEN_MODEL_ID:
        return True
    if not isinstance(value, str):
        return False
    parts = Path(value).parts
    try:
        model_index = parts.index("models--Qwen--Qwen3-VL-2B-Instruct")
    except ValueError:
        return False
    suffix = parts[model_index + 1 :]
    return (
        len(suffix) == 2
        and suffix[0] == "snapshots"
        and len(suffix[1]) == 40
        and all(character in "0123456789abcdef" for character in suffix[1])
    )


def validate_robotwin_ar_bin_contract(
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Require the full-finetune H32/14D BAR+BIN policy used by this benchmark."""
    policy_root = Path(checkpoint_path).expanduser().resolve()
    policy_config = _json_mapping(policy_root / "config.json")
    _require_fields(
        policy_config,
        {
            "name": "bar_bin_qwen",
            "action_dim": 14,
            "action_horizon": 32,
            "n_action_steps": 16,
            "state_prefix_num_bins": 256,
            "use_lora": False,
            "freeze_vision_encoder": False,
        },
        label="Policy config",
    )
    vlm_model_name = policy_config.get("vlm_model_name")
    if not _is_robotwin_qwen_model(vlm_model_name):
        raise ValueError(
            "Policy config violates the RoboTwin AR BIN contract: "
            "vlm_model_name must identify Qwen/Qwen3-VL-2B-Instruct, got "
            f"{vlm_model_name!r}."
        )
    action_tokenizer = policy_config.get("action_tokenizer")
    if not isinstance(action_tokenizer, dict):
        raise ValueError("Policy config has no action_tokenizer mapping.")
    _require_fields(
        action_tokenizer,
        {"name": "bin", "kind": "tokens"},
        label="Policy action tokenizer",
    )
    tokenizer_path = action_tokenizer.get("pretrained_path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise ValueError("Policy BIN tokenizer has no pretrained_path.")

    policy_io = _json_mapping(policy_root / "policy_io.json")
    _require_fields(policy_io, {"n_action_steps": 32}, label="PolicyIO")
    action_feature = policy_io.get("action_feature")
    if not isinstance(action_feature, dict) or action_feature.get("shape") != [14]:
        raise ValueError(
            "PolicyIO action_feature must have 14D shape for RoboTwin, got "
            f"{action_feature!r}."
        )
    obs_features = policy_io.get("obs_features")
    if not isinstance(obs_features, list):
        raise ValueError("PolicyIO obs_features must be a list.")
    observed_shapes = {
        item.get("key"): item.get("shape")
        for item in obs_features
        if isinstance(item, dict)
    }
    expected_shapes = {
        "observation.state": [14],
        "observation.images.cam_high": _ROBOTWIN_CAMERA_SHAPE,
        "observation.images.cam_left_wrist": _ROBOTWIN_CAMERA_SHAPE,
        "observation.images.cam_right_wrist": _ROBOTWIN_CAMERA_SHAPE,
    }
    if observed_shapes != expected_shapes:
        raise ValueError(
            "PolicyIO observation features violate the RoboTwin contract: "
            f"expected={expected_shapes}, actual={observed_shapes}."
        )

    tokenizer_root = _resolve_dependency_path(
        policy_root=policy_root, dependency_path=tokenizer_path
    )
    tokenizer_config = _json_mapping(tokenizer_root / "config.json")
    _require_fields(
        tokenizer_config,
        {"name": "bin", "action_dim": 14, "action_horizon": 32, "num_bins": 256},
        label="BIN codec config",
    )
    tokenizer_io = _json_mapping(tokenizer_root / "policy_io.json")
    tokenizer_action = tokenizer_io.get("action_feature")
    if not isinstance(tokenizer_action, dict) or tokenizer_action.get("shape") != [14]:
        raise ValueError(
            "BIN PolicyIO action_feature must have 14D shape, got "
            f"{tokenizer_action!r}."
        )
    _require_fields(tokenizer_io, {"n_action_steps": 32}, label="BIN PolicyIO")
    return {
        "policy_name": policy_config["name"],
        "vlm_model_name": str(vlm_model_name),
        "action_dim": 14,
        "action_horizon": 32,
        "execute_steps": 16,
        "tokenizer_name": "bin",
        "tokenizer_kind": "tokens",
        "num_bins": 256,
        "tokenizer_path": str(tokenizer_root),
    }


def build_policy_serving_identity(checkpoint_path: str | Path) -> dict[str, Any]:
    """Hash a policy checkpoint together with its external codec dependency."""
    checkpoint = build_checkpoint_identity(checkpoint_path)
    config_path = Path(checkpoint["path"]) / "config.json"
    config = _json_mapping(config_path)

    dependencies: dict[str, dict[str, Any]] = {}
    action_tokenizer = config.get("action_tokenizer")
    if action_tokenizer is not None:
        if not isinstance(action_tokenizer, dict):
            raise ValueError("Policy action_tokenizer config must be a mapping.")
        pretrained_path = action_tokenizer.get("pretrained_path")
        if not isinstance(pretrained_path, str) or not pretrained_path:
            raise ValueError(
                "Policy action_tokenizer config must contain pretrained_path for "
                "auditable serving."
            )
        dependency_root = _resolve_dependency_path(
            policy_root=Path(checkpoint["path"]),
            dependency_path=pretrained_path,
        )
        dependencies["action_tokenizer"] = build_checkpoint_identity(dependency_root)

    aggregate = {
        "checkpoint_artifact_sha256": checkpoint["artifact_sha256"],
        "dependencies": {
            name: identity["artifact_sha256"]
            for name, identity in sorted(dependencies.items())
        },
    }
    canonical = json.dumps(aggregate, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "artifact_sha256": hashlib.sha256(canonical).hexdigest(),
        "checkpoint": checkpoint,
        "dependencies": dependencies,
    }
