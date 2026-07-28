# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Strict, training-private-free X-Tokenizer inference artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import torch
from omegaconf import OmegaConf
from torch import Tensor

from praxis.codecs.latents.xtokenizer.codec import (
    INFERENCE_ARTIFACT_FILENAME,
    INFERENCE_MANIFEST_FILENAME,
    XTokenizerCodec,
)
from praxis.codecs.latents.xtokenizer.config import XTokenizerModelConfig
from praxis.core.types import CANONICAL_ACTION_KEY, NormMode, PolicyIO

if TYPE_CHECKING:
    from praxis.codecs.latents.xtokenizer.cache import (
        XTokenizerCacheAdapter,
        load_xtokenizer_cache_adapter,
    )

INFERENCE_FORMAT_VERSION = 3
_ACTION_SEMANTICS = {
    "public_action": "absolute",
    "tokenized_action": "observation_relative",
    "observation_anchor": "absolute",
    "relative_reference": "single_pre_chunk_observation",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INFERENCE_PREFIXES = ("encoder.", "quantizer.", "decoder.")
_TRAINING_PREFIXES = (
    "mam_head.",
    "alignment_projection.",
    "logit_scale",
    "prediction_head.",
)
_MANIFEST_FIELDS = {
    "format_version",
    "artifact_file",
    "artifact_sha256",
    "source_checkpoint_sha256",
    "training_private_data",
    "action_horizon",
    "compression_ratio",
    "codebook_size",
    "num_quantizers",
    "action_semantics",
    "dof_layout",
    "embodiment_registry",
    "normalization",
    "model_config",
    "policy_io",
    "training_data_identity",
}
_SEMANTIC_METADATA_FIELDS = (
    "source_checkpoint_sha256",
    "action_semantics",
    "dof_layout",
    "embodiment_registry",
    "normalization",
    "policy_io",
    "model_config",
    "training_data_identity",
)
_TRAINING_DATA_IDENTITY_FIELDS = {
    "dataset_repo_id",
    "dataset_revision",
    "canonical_stats_identity",
    "vl_sidecar_identity",
}
_CANONICAL_STATS_IDENTITY_FIELDS = {
    "format_version",
    "dataset_repo_id",
    "dataset_revision",
    "robot_type",
    "action_layout",
    "normalization",
    "content_fingerprint",
}
_CANONICAL_NORMALIZATION_FIELDS = {
    "mode",
    "quantile_low",
    "quantile_high",
    "clamp",
    "stats_horizon",
    "predict_action",
    "observation_state",
}
_NORMALIZATION_KIND = "xtokenizer_dual_role_v1"
_CACHE_EXPORTS = {
    "XTokenizerCacheAdapter",
    "load_xtokenizer_cache_adapter",
}
_VL_SIDECAR_IDENTITY_FIELDS = {
    "format_version",
    "dataset_repo_id",
    "dataset_revision",
    "extractor_revision",
    "model_id",
    "model_revision",
    "layer",
    "stride",
    "feature_dim",
    "episode_indices",
    "episode_lengths",
    "image_keys",
    "global_instruction",
    "local_subtask",
    "synthetic",
    "generation_fingerprint",
    "features_sha256",
    "next_features_sha256",
    "valid_sha256",
}

__all__ = [
    "INFERENCE_FORMAT_VERSION",
    "XTokenizerCacheAdapter",
    "export_codec_inference_artifact",
    "export_xtokenizer_checkpoint",
    "load_exported_xtokenizer",
    "load_exported_xtokenizer_manifest",
    "load_exported_xtokenizer_snapshot",
    "load_xtokenizer_cache_adapter",
    "sha256_file",
]


def __getattr__(name: str) -> Any:
    if name in _CACHE_EXPORTS:
        cache = import_module("praxis.codecs.latents.xtokenizer.cache")
        for export in _CACHE_EXPORTS:
            globals()[export] = getattr(cache, export)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def export_xtokenizer_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
) -> Path:
    """Export a Praxis trainer checkpoint as an atomic inference artifact."""
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    model_dir = _resolve_pretrained_model_dir(checkpoint_dir)
    ema_path = _resolve_adjacent_ema_path(checkpoint_dir, model_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    digest_before = _source_checkpoint_sha256(model_dir, ema_path)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.source-",
        dir=output_dir.parent,
    ) as snapshot_root_raw:
        snapshot_root = Path(snapshot_root_raw)
        snapshot_model_dir = snapshot_root / "pretrained_model"
        shutil.copytree(
            model_dir,
            snapshot_model_dir,
            copy_function=shutil.copyfile,
        )
        snapshot_ema_path: Path | None = None
        if ema_path is not None:
            snapshot_ema_path = snapshot_root / "training_state" / "ema.pt"
            snapshot_ema_path.parent.mkdir()
            shutil.copyfile(ema_path, snapshot_ema_path)

        digest_after = _source_checkpoint_sha256(model_dir, ema_path)
        ema_path_after = _resolve_adjacent_ema_path(checkpoint_dir, model_dir)
        snapshot_digest = _source_checkpoint_sha256(
            snapshot_model_dir,
            snapshot_ema_path,
        )
        if (
            digest_before != digest_after
            or digest_before != snapshot_digest
            or ema_path_after != ema_path
        ):
            raise RuntimeError("Source checkpoint changed while it was being exported.")

        codec = cast(
            XTokenizerCodec,
            XTokenizerCodec.from_pretrained(
                snapshot_model_dir,
                local_files_only=True,
            ),
        )
        if snapshot_ema_path is not None:
            from praxis.training.core.ema import EMAModel

            ema_state = torch.load(
                snapshot_ema_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(ema_state, dict):
                raise TypeError("X-Tokenizer EMA state must be a parameter mapping.")
            ema = EMAModel(codec)
            if set(ema_state) != set(ema.shadow):
                raise RuntimeError(
                    "X-Tokenizer EMA state parameter names do not match codec."
                )
            ema.load_state_dict(ema_state)
            ema.apply(codec)
        dof_layout, embodiment_registry, training_data_identity = (
            _load_training_metadata(
                snapshot_model_dir,
                expected_action_horizon=codec.action_horizon,
            )
        )
        model_config_layout = codec.config.action_layout
        if model_config_layout is None:
            raise ValueError("X-Tokenizer model config is missing its action layout.")
        model_layout = _resolve_action_layout(model_config_layout)
        training_layout = _resolve_action_layout(dof_layout)
        if not _deep_equal(model_layout, training_layout):
            raise ValueError(
                "X-Tokenizer model config action layout does not match training "
                "dataset action layout."
            )
        return codec.export_inference_artifact(
            output_dir,
            source_checkpoint_sha256=snapshot_digest,
            dof_layout=dof_layout,
            embodiment_registry=embodiment_registry,
            training_data_identity=training_data_identity,
        )


def export_codec_inference_artifact(
    codec: XTokenizerCodec,
    output_dir: Path,
    *,
    source_checkpoint_sha256: str | None = None,
    dof_layout: Mapping[str, Any] | None = None,
    embodiment_registry: Mapping[str, int] | None = None,
    training_data_identity: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically publish one reloadable inference-only codec artifact."""
    from praxis.training.core.checkpoint.staging import (
        publish_staged_directory,
        remove_path,
    )

    checkpoint_normalization = codec._normalization_checkpoint_payload()
    normalization_kind = checkpoint_normalization.get("normalization_kind")
    supplied_canonical_identity = (
        training_data_identity.get("canonical_stats_identity")
        if isinstance(training_data_identity, Mapping)
        else None
    )
    if normalization_kind == _NORMALIZATION_KIND:
        checkpoint_identity = checkpoint_normalization.get("canonical_stats_identity")
        if not isinstance(supplied_canonical_identity, Mapping) or not _deep_equal(
            checkpoint_identity,
            supplied_canonical_identity,
        ):
            raise ValueError(
                "X-Tokenizer checkpoint normalization identity does not match "
                "the canonical training statistics identity."
            )
    elif supplied_canonical_identity is not None:
        raise ValueError(
            "Canonical statistics identity requires marked X-Tokenizer normalization."
        )

    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"Artifact destination must be a directory: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    token = uuid4().hex
    staging_dir = output_dir.parent / f".{output_dir.name}.tmp-{token}"
    backup_dir = output_dir.parent / f".{output_dir.name}.prev-{token}"
    remove_path(staging_dir)
    remove_path(backup_dir)
    staging_dir.mkdir()

    try:
        state_dict = codec.inference_state_dict()
        stats = checkpoint_normalization["stats"]
        modes = checkpoint_normalization["modes"]
        json_stats = _json_normalization_stats(stats)
        normalization = {
            "stats": _select_stats_for_normalization_modes(json_stats, modes),
            "modes": dict(modes),
        }
        if normalization_kind == _NORMALIZATION_KIND:
            if training_data_identity is None:
                raise ValueError(
                    "Dual-role X-Tokenizer inference export requires authenticated "
                    "canonical statistics identity."
                )
            canonical_identity = training_data_identity.get("canonical_stats_identity")
            if not isinstance(canonical_identity, Mapping):
                raise ValueError(
                    "Dual-role X-Tokenizer inference export requires authenticated "
                    "canonical statistics identity."
                )
            canonical_normalization = canonical_identity.get("normalization")
            if not isinstance(canonical_normalization, Mapping):
                raise ValueError(
                    "Dual-role X-Tokenizer inference export requires authenticated "
                    "canonical statistics identity."
                )
            normalization.update(
                {
                    "kind": _NORMALIZATION_KIND,
                    "roles": {
                        "predict_action": {
                            "key": codec.action_key,
                            "space": "observation_relative",
                        },
                        "observation_state": {
                            "key": "observation.state",
                            "space": "absolute",
                        },
                    },
                    "identity": {
                        "format_version": canonical_identity.get("format_version"),
                        "content_fingerprint": canonical_identity.get(
                            "content_fingerprint"
                        ),
                        "stats_horizon": canonical_normalization.get("stats_horizon"),
                    },
                }
            )
        elif normalization_kind is not None or normalization["modes"]:
            raise ValueError(
                "Normalized X-Tokenizer inference export requires explicit "
                f"{_NORMALIZATION_KIND!r} checkpoint state."
            )
        else:
            normalization = {
                "stats": {codec.action_key: {}, "observation.state": {}},
                "modes": {
                    codec.action_key: NormMode.NONE.value,
                    "observation.state": NormMode.NONE.value,
                },
            }
        model_config_layout = codec.config.action_layout
        if model_config_layout is None:
            raise ValueError("X-Tokenizer model config is missing its action layout.")
        resolved_layout = _resolve_action_layout(
            _json_mapping(dof_layout) if dof_layout is not None else model_config_layout
        )
        resolved_registry = (
            dict(embodiment_registry)
            if embodiment_registry is not None
            else {"default": 0}
        )
        source_digest = source_checkpoint_sha256 or _sha256_codec_state(
            codec,
            state_dict,
            normalization=normalization,
        )
        semantic_metadata = {
            "source_checkpoint_sha256": source_digest,
            "action_semantics": dict(_ACTION_SEMANTICS),
            "dof_layout": resolved_layout,
            "embodiment_registry": resolved_registry,
            "normalization": normalization,
            "policy_io": codec.io.to_dict() if codec.io is not None else None,
            "model_config": codec.config.to_dict(),
            "training_data_identity": (
                _json_mapping(training_data_identity)
                if training_data_identity is not None
                else None
            ),
        }
        payload = {
            "format_version": INFERENCE_FORMAT_VERSION,
            "state_dict": state_dict,
            "metadata": semantic_metadata,
        }
        artifact_path = staging_dir / INFERENCE_ARTIFACT_FILENAME
        torch.save(payload, artifact_path)
        manifest = {
            "format_version": INFERENCE_FORMAT_VERSION,
            "artifact_file": INFERENCE_ARTIFACT_FILENAME,
            "artifact_sha256": sha256_file(artifact_path),
            "training_private_data": False,
            "action_horizon": codec.action_horizon,
            "compression_ratio": codec.compression_ratio,
            "codebook_size": codec.codebook_size,
            "num_quantizers": codec.num_quantizers,
            **semantic_metadata,
        }
        _validate_manifest(manifest)
        _write_json(staging_dir / INFERENCE_MANIFEST_FILENAME, manifest)
        load_exported_xtokenizer(staging_dir)
        publish_staged_directory(
            staging_dir=staging_dir,
            checkpoint_dir=output_dir,
            backup_dir=backup_dir,
        )
    except Exception:
        remove_path(staging_dir)
        remove_path(backup_dir)
        raise
    return output_dir


def load_exported_xtokenizer(path: Path) -> XTokenizerCodec:
    """Load a validated inference artifact from one authenticated snapshot."""
    codec, _ = load_exported_xtokenizer_snapshot(path)
    return codec


def load_exported_xtokenizer_snapshot(
    path: Path,
) -> tuple[XTokenizerCodec, dict[str, Any]]:
    """Return a codec and the exact manifest snapshot that authenticated it."""
    artifact_dir = Path(path)
    manifest = load_exported_xtokenizer_manifest(artifact_dir)
    artifact_path = artifact_dir / manifest["artifact_file"]
    if artifact_path.is_symlink():
        raise ValueError("Inference artifact_file must not be a symbolic link.")
    try:
        descriptor = os.open(
            artifact_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ValueError(
            "Exported X-Tokenizer changed while loading its artifact snapshot."
        ) from exc
    with os.fdopen(descriptor, "rb") as artifact_file:
        actual_digest = _sha256_open_file(artifact_file)
        if actual_digest != manifest["artifact_sha256"]:
            raise ValueError(
                "X-Tokenizer artifact SHA256 mismatch: "
                f"expected {manifest['artifact_sha256']}, got {actual_digest}."
            )
        artifact_file.seek(0)
        payload = torch.load(
            artifact_file,
            map_location="cpu",
            weights_only=True,
        )
    codec = _codec_from_export_payload(payload, manifest)
    try:
        current_manifest = load_exported_xtokenizer_manifest(artifact_dir)
    except (
        FileNotFoundError,
        NotADirectoryError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Exported X-Tokenizer changed while loading its artifact snapshot."
        ) from exc
    if not _deep_equal(current_manifest, manifest):
        raise ValueError(
            "Exported X-Tokenizer changed while loading its artifact snapshot."
        )
    return codec, manifest


def _codec_from_export_payload(
    payload: Any,
    manifest: dict[str, Any],
) -> XTokenizerCodec:
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "state_dict",
        "metadata",
    }:
        raise ValueError(
            "Inference payload must contain exactly format_version, state_dict, "
            "and metadata."
        )
    if payload["format_version"] != INFERENCE_FORMAT_VERSION:
        raise ValueError(
            "Inference payload format_version does not match the supported version."
        )
    metadata = payload["metadata"]
    expected_metadata = {key: manifest[key] for key in _SEMANTIC_METADATA_FIELDS}
    if not _deep_equal(metadata, expected_metadata):
        raise ValueError(
            "Inference payload metadata does not exactly match the manifest."
        )
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError("Inference payload state_dict must map strings to tensors.")
    training_keys = sorted(
        key
        for key in state_dict
        if key.startswith(_TRAINING_PREFIXES) or not key.startswith(_INFERENCE_PREFIXES)
    )
    if training_keys:
        raise ValueError(
            "Inference artifact contains unexpected training-only state: "
            + ", ".join(training_keys)
        )

    config = XTokenizerModelConfig.from_dict(manifest["model_config"])
    codec = XTokenizerCodec(config=config)
    expected_state = codec.inference_state_dict()
    missing = sorted(set(expected_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected_state))
    if missing or unexpected:
        raise ValueError(
            "Inference state keys do not match the codec: "
            f"missing={missing}, unexpected={unexpected}."
        )
    for key, expected in expected_state.items():
        value = state_dict[key]
        if value.shape != expected.shape:
            raise ValueError(
                f"Inference state shape mismatch for {key}: "
                f"expected {tuple(expected.shape)}, got {tuple(value.shape)}."
            )
        if value.dtype != expected.dtype:
            raise ValueError(
                f"Inference state dtype mismatch for {key}: "
                f"expected {expected.dtype}, got {value.dtype}."
            )
    codebooks = state_dict["quantizer.codebooks"]
    expected_codebook_shape = (
        config.num_quantizers,
        config.codebook_size,
        config.latent_dim,
    )
    if codebooks.dtype != torch.float32 or codebooks.shape != expected_codebook_shape:
        raise ValueError(
            "Inference artifact must contain all FP32 codebooks with shape "
            f"{expected_codebook_shape}."
        )

    codec.encoder.load_state_dict(_submodule_state(state_dict, "encoder"), strict=True)
    codec.quantizer.load_state_dict(
        _submodule_state(state_dict, "quantizer"),
        strict=True,
    )
    codec.decoder.load_state_dict(_submodule_state(state_dict, "decoder"), strict=True)
    policy_io_raw = manifest["policy_io"]
    if policy_io_raw is not None:
        codec.set_io(PolicyIO.from_dict(policy_io_raw))
    normalization = manifest["normalization"]
    stats = {
        key: {
            stat_name: torch.tensor(values, dtype=torch.float32)
            for stat_name, values in stat_block.items()
        }
        for key, stat_block in normalization["stats"].items()
    }
    if normalization.get("kind") == _NORMALIZATION_KIND:
        roles = normalization["roles"]
        predict_key = roles["predict_action"]["key"]
        observation_key = roles["observation_state"]["key"]
        codec.set_xtokenizer_normalization_stats(
            predict_stats={predict_key: stats[predict_key]},
            observation_stats={
                observation_key: stats[observation_key],
            },
            canonical_stats_identity=manifest["training_data_identity"][
                "canonical_stats_identity"
            ],
            stats_horizon=normalization["identity"]["stats_horizon"],
        )
    codec.eval()
    return codec


def load_exported_xtokenizer_manifest(path: Path) -> dict[str, Any]:
    """Read and validate artifact metadata without loading model tensors."""
    artifact_dir = Path(path)
    if not artifact_dir.is_dir():
        raise NotADirectoryError(
            f"Expected an exported X-Tokenizer directory, got: {artifact_dir}"
        )
    manifest_path = artifact_dir / INFERENCE_MANIFEST_FILENAME
    if manifest_path.is_symlink():
        raise ValueError("Inference manifest must not be a symbolic link.")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Exported X-Tokenizer is missing {INFERENCE_MANIFEST_FILENAME}."
        )
    manifest = _read_json_object(manifest_path)
    _validate_manifest(manifest)
    declared_files = {
        INFERENCE_ARTIFACT_FILENAME,
        INFERENCE_MANIFEST_FILENAME,
    }
    actual_files = {entry.name for entry in artifact_dir.iterdir()}
    undeclared = sorted(actual_files - declared_files)
    if undeclared:
        raise ValueError(
            f"Exported X-Tokenizer contains undeclared files: {undeclared}."
        )
    artifact_path = artifact_dir / manifest["artifact_file"]
    if artifact_path.parent != artifact_dir or not artifact_path.is_file():
        raise FileNotFoundError(
            f"Exported X-Tokenizer is missing {INFERENCE_ARTIFACT_FILENAME}."
        )
    return manifest


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return _sha256_open_file(handle)


def _sha256_open_file(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash names, sizes, and bytes for every regular checkpoint file."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"Expected checkpoint directory, got: {root}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError(
                f"Source checkpoint must not contain symbolic links: {candidate}"
            )
        if candidate.is_file():
            files.append(candidate)
    if not files:
        raise ValueError(f"Source checkpoint directory is empty: {root}")

    digest = hashlib.sha256(b"praxis-xtokenizer-checkpoint-v1\0")
    for file_path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = file_path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = file_path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _resolve_pretrained_model_dir(checkpoint_dir: Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            f"Expected X-Tokenizer checkpoint directory, got: {checkpoint_dir}"
        )
    nested = checkpoint_dir / "pretrained_model"
    model_dir = nested if nested.is_dir() else checkpoint_dir
    required = ("config.json", "policy_io.json", "norm_stats.pt", "train_config.yaml")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"X-Tokenizer checkpoint is missing required files: {missing}."
        )
    return model_dir


def _resolve_adjacent_ema_path(checkpoint_dir: Path, model_dir: Path) -> Path | None:
    checkpoint_root = (
        checkpoint_dir if model_dir != checkpoint_dir else model_dir.parent
    )
    ema_path = checkpoint_root / "training_state" / "ema.pt"
    if ema_path.is_symlink():
        raise ValueError("X-Tokenizer EMA state must not be a symbolic link.")
    return ema_path if ema_path.is_file() else None


def _source_checkpoint_sha256(model_dir: Path, ema_path: Path | None) -> str:
    model_digest = sha256_directory(model_dir)
    if ema_path is None:
        return model_digest
    digest = hashlib.sha256(b"praxis-xtokenizer-checkpoint-with-ema-v1\0")
    digest.update(bytes.fromhex(model_digest))
    digest.update(bytes.fromhex(sha256_file(ema_path)))
    return digest.hexdigest()


def _load_training_metadata(
    model_dir: Path,
    *,
    expected_action_horizon: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    config = OmegaConf.load(model_dir / "train_config.yaml")
    dataset_config = OmegaConf.select(config, "dataset", default=None)
    dataset = OmegaConf.to_container(
        dataset_config,
        resolve=True,
    )
    if not isinstance(dataset, dict):
        raise ValueError("train_config.yaml is missing dataset metadata.")
    action_layout = dataset.get("action_layout")
    if not isinstance(action_layout, dict):
        raise ValueError("train_config.yaml is missing dataset.action_layout.")

    registry_raw = dataset.get("embodiment_registry")
    if registry_raw is None:
        robot_id = _strict_int(dataset.get("robot_type"), label="dataset.robot_type")
        robot_name = dataset.get("robot_type_name")
        if robot_name is None and dataset.get("name") == "xtokenizer_bridge":
            robot_name = "WidowX"
        if not isinstance(robot_name, str) or not robot_name:
            raise ValueError(
                "train_config.yaml must define dataset.embodiment_registry or "
                "dataset.robot_type_name."
            )
        registry = {robot_name: robot_id}
    elif isinstance(registry_raw, dict):
        registry = {
            str(name): _strict_int(value, label=f"embodiment_registry[{name!r}]")
            for name, value in registry_raw.items()
        }
    else:
        raise TypeError("dataset.embodiment_registry must be a mapping.")
    resolved_layout = _json_mapping(action_layout)
    training_data_identity = {
        "dataset_repo_id": dataset.get("repo_id"),
        "dataset_revision": dataset.get("dataset_revision"),
        "canonical_stats_identity": dataset.get("canonical_stats_identity"),
        "vl_sidecar_identity": dataset.get("vl_sidecar_identity"),
    }
    _validate_training_data_identity(
        training_data_identity,
        expected_dof_layout=resolved_layout,
        expected_action_horizon=expected_action_horizon,
        allow_none=False,
    )
    return resolved_layout, registry, training_data_identity


def _validate_manifest(manifest: dict[str, Any]) -> None:
    missing = sorted(_MANIFEST_FIELDS - set(manifest))
    unexpected = sorted(set(manifest) - _MANIFEST_FIELDS)
    if missing or unexpected:
        raise ValueError(
            "Inference manifest fields are malformed: "
            f"missing={missing}, unexpected={unexpected}."
        )
    if manifest["format_version"] != INFERENCE_FORMAT_VERSION:
        raise ValueError(
            "Unsupported X-Tokenizer format_version "
            f"{manifest['format_version']!r}; expected {INFERENCE_FORMAT_VERSION}."
        )
    if manifest["artifact_file"] != INFERENCE_ARTIFACT_FILENAME:
        raise ValueError(
            "Inference manifest artifact_file must be exactly "
            f"{INFERENCE_ARTIFACT_FILENAME!r}."
        )
    _validate_sha256(manifest["artifact_sha256"], label="artifact_sha256")
    _validate_sha256(
        manifest["source_checkpoint_sha256"],
        label="source_checkpoint_sha256",
    )
    if manifest["training_private_data"] is not False:
        raise ValueError(
            "Inference artifacts must declare training private data false."
        )
    if manifest["action_semantics"] != _ACTION_SEMANTICS:
        raise ValueError(
            "Inference action_semantics must declare public absolute I/O and "
            "observation-relative tokenization."
        )

    geometry = {
        name: _strict_int(manifest[name], label=name)
        for name in (
            "action_horizon",
            "compression_ratio",
            "codebook_size",
            "num_quantizers",
        )
    }
    if geometry["action_horizon"] <= 0 or geometry["codebook_size"] <= 0:
        raise ValueError("Inference geometry dimensions must be positive.")
    if geometry["compression_ratio"] != 4:
        raise ValueError("X-Tokenizer compression_ratio must be exactly 4.")
    if geometry["num_quantizers"] != 4:
        raise ValueError("X-Tokenizer num_quantizers must be exactly 4.")

    config_raw = manifest["model_config"]
    if not isinstance(config_raw, dict):
        raise TypeError("Inference model_config must be a mapping.")
    try:
        config = XTokenizerModelConfig.from_dict(config_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Inference model_config is invalid: {exc}") from exc
    if config.to_dict() != config_raw:
        raise ValueError("Inference model_config must be complete and canonical.")
    for name, value in geometry.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"Inference model_config {name} does not match manifest {name}."
            )

    _validate_action_layout(manifest["dof_layout"], config=config)
    _validate_embodiment_registry(
        manifest["embodiment_registry"],
        config=config,
    )
    policy_io = _validate_policy_io(manifest["policy_io"], config=config)
    action_key = (
        policy_io.action_feature.key
        if policy_io is not None and policy_io.action_feature is not None
        else CANONICAL_ACTION_KEY
    )
    normalization = manifest["normalization"]
    paper_mode = (
        isinstance(normalization, dict)
        and normalization.get("kind") == _NORMALIZATION_KIND
    )
    expected_modes: dict[str, NormMode] | None = None
    if policy_io is not None and policy_io.action_feature is not None:
        expected_modes = {action_key: policy_io.action_feature.norm_mode}
        observation_feature = next(
            (
                feature
                for feature in policy_io.obs_features
                if feature.key == "observation.state"
            ),
            None,
        )
        if observation_feature is None:
            if not paper_mode:
                raise ValueError(
                    "Inference policy_io must define absolute observation.state."
                )
            expected_modes["observation.state"] = NormMode.MIN_MAX
        else:
            expected_modes["observation.state"] = observation_feature.norm_mode
    _validate_training_data_identity(
        manifest["training_data_identity"],
        expected_dof_layout=manifest["dof_layout"],
        expected_action_horizon=config.action_horizon,
        allow_none=True,
    )
    training_data_identity = manifest["training_data_identity"]
    canonical_stats_identity = (
        None
        if training_data_identity is None
        else training_data_identity["canonical_stats_identity"]
    )
    _validate_normalization(
        normalization,
        config=config,
        action_key=action_key,
        expected_modes=expected_modes,
        canonical_stats_identity=canonical_stats_identity,
    )
    if paper_mode != (canonical_stats_identity is not None):
        raise ValueError(
            "Canonical statistics identity and marked X-Tokenizer normalization "
            "must be present together."
        )


def _validate_training_data_identity(
    value: Any,
    *,
    expected_dof_layout: Mapping[str, Any],
    expected_action_horizon: int,
    allow_none: bool,
) -> None:
    from praxis.data.xtokenizer_vl import vl_sidecar_generation_fingerprint

    if value is None:
        if allow_none:
            return
        raise ValueError("Training data identity is required for checkpoint export.")
    if not isinstance(value, dict) or set(value) != _TRAINING_DATA_IDENTITY_FIELDS:
        raise ValueError("Training data identity fields are malformed.")
    repo_id = value["dataset_repo_id"]
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("Training data identity dataset_repo_id is malformed.")
    revision = _immutable_revision(
        value["dataset_revision"],
        label="training data identity dataset_revision",
    )

    canonical = value["canonical_stats_identity"]
    if (
        not isinstance(canonical, dict)
        or set(canonical) != _CANONICAL_STATS_IDENTITY_FIELDS
    ):
        raise ValueError("Canonical statistics identity fields are malformed.")
    if canonical["format_version"] != 3:
        raise ValueError("Canonical statistics identity format_version is unsupported.")
    if (
        not isinstance(canonical["robot_type"], str)
        or not canonical["robot_type"].strip()
    ):
        raise ValueError("Canonical statistics identity robot_type is malformed.")
    if (
        canonical["dataset_repo_id"] != repo_id
        or _immutable_revision(
            canonical["dataset_revision"],
            label="canonical statistics dataset_revision",
        )
        != revision
    ):
        raise ValueError(
            "Canonical statistics identity does not match the training dataset."
        )
    if not _deep_equal(canonical["action_layout"], expected_dof_layout):
        raise ValueError(
            "Canonical statistics identity action layout does not match training."
        )
    normalization = canonical["normalization"]
    if (
        not isinstance(normalization, dict)
        or set(normalization) != _CANONICAL_NORMALIZATION_FIELDS
    ):
        raise ValueError("Canonical statistics normalization fields are malformed.")
    if normalization["mode"] != "quantile_min_max":
        raise ValueError("Canonical statistics normalization mode is malformed.")
    if (
        normalization["quantile_low"] != 0.001
        or normalization["quantile_high"] != 0.999
    ):
        raise ValueError("Canonical statistics quantile range is malformed.")
    if normalization["clamp"] is not True:
        raise ValueError("Canonical statistics identity must clamp values.")
    stats_horizon = _strict_int(
        normalization["stats_horizon"],
        label="canonical statistics stats_horizon",
    )
    if stats_horizon <= 0 or stats_horizon != expected_action_horizon:
        raise ValueError(
            "Canonical statistics identity stats_horizon does not match the model."
        )
    action_dim = sum(int(dim) for dim in expected_dof_layout["dof_dims"].values())
    for role, expected_space in (
        ("predict_action", "observation_relative"),
        ("observation_state", "absolute"),
    ):
        block = normalization[role]
        if not isinstance(block, dict) or set(block) != {"space", "q001", "q999"}:
            raise ValueError(f"Canonical statistics {role} role is malformed.")
        if block["space"] != expected_space:
            raise ValueError(f"Canonical statistics {role} space is malformed.")
        for statistic in ("q001", "q999"):
            values = block[statistic]
            if not isinstance(values, list) or len(values) != action_dim:
                raise ValueError(
                    f"Canonical statistics {role}.{statistic} dimension is malformed."
                )
            if not all(
                isinstance(item, int | float)
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in values
            ):
                raise ValueError(
                    f"Canonical statistics {role}.{statistic} is malformed."
                )
        if any(
            float(low) >= float(high)
            for low, high in zip(block["q001"], block["q999"], strict=True)
        ):
            raise ValueError(f"Canonical statistics {role} bounds are malformed.")
    _validate_sha256(
        canonical["content_fingerprint"],
        label="canonical statistics content_fingerprint",
    )
    from praxis.data.xtokenizer_canonical_stats import (
        _content_fingerprint,  # noqa: PLC2701
    )

    if canonical["content_fingerprint"] != _content_fingerprint(canonical):
        raise ValueError("Canonical statistics content_fingerprint mismatch.")

    sidecar = value["vl_sidecar_identity"]
    if not isinstance(sidecar, dict) or set(sidecar) != _VL_SIDECAR_IDENTITY_FIELDS:
        raise ValueError("VL sidecar identity fields are malformed.")
    if sidecar["format_version"] != 3:
        raise ValueError("VL sidecar identity format_version is unsupported.")
    if (
        sidecar["dataset_repo_id"] != repo_id
        or _immutable_revision(
            sidecar["dataset_revision"],
            label="VL sidecar dataset_revision",
        )
        != revision
    ):
        raise ValueError("VL sidecar identity does not match the training dataset.")
    for key in (
        "extractor_revision",
        "model_id",
        "model_revision",
        "global_instruction",
        "local_subtask",
    ):
        if not isinstance(sidecar[key], str) or not sidecar[key].strip():
            raise ValueError(f"VL sidecar identity {key} is malformed.")
    for key in ("layer", "stride", "feature_dim"):
        if not isinstance(sidecar[key], int) or isinstance(sidecar[key], bool):
            raise ValueError(f"VL sidecar identity {key} is malformed.")
    if sidecar["stride"] <= 0 or sidecar["feature_dim"] <= 0:
        raise ValueError("VL sidecar identity geometry must be positive.")
    indices = sidecar["episode_indices"]
    lengths = sidecar["episode_lengths"]
    if (
        not isinstance(indices, list)
        or not isinstance(lengths, list)
        or len(indices) != len(lengths)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in indices
        )
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in lengths
        )
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("VL sidecar identity episode metadata is malformed.")
    image_keys = sidecar["image_keys"]
    if (
        not isinstance(image_keys, list)
        or any(not isinstance(item, str) or not item.strip() for item in image_keys)
        or len(set(image_keys)) != len(image_keys)
    ):
        raise ValueError("VL sidecar identity image_keys are malformed.")
    if not isinstance(sidecar["synthetic"], bool):
        raise ValueError("VL sidecar identity synthetic flag is malformed.")
    for key in (
        "generation_fingerprint",
        "features_sha256",
        "next_features_sha256",
        "valid_sha256",
    ):
        _validate_sha256(sidecar[key], label=f"VL sidecar {key}")
    if sidecar["generation_fingerprint"] != vl_sidecar_generation_fingerprint(sidecar):
        raise ValueError("VL sidecar identity generation fingerprint mismatch.")


def _immutable_revision(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(f"{label} must be an immutable 40-character commit SHA.")
    return value.lower()


def _validate_action_layout(
    value: Any,
    *,
    config: XTokenizerModelConfig,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "convention",
        "dof_dims",
        "dof_order",
    }:
        raise ValueError(
            "Inference action layout must contain convention, dof_dims, and dof_order."
        )
    if value["convention"] != "absolute":
        raise ValueError("Inference action layout convention must be absolute.")
    dof_dims = value["dof_dims"]
    if not isinstance(dof_dims, dict) or not dof_dims:
        raise ValueError(
            "Inference action layout dof_dims must be a non-empty mapping."
        )
    dimensions = [
        _strict_int(dim, label=f"dof_dims[{name!r}]")
        for name, dim in dof_dims.items()
        if isinstance(name, str) and name
    ]
    if len(dimensions) != len(dof_dims) or any(dim <= 0 for dim in dimensions):
        raise ValueError("Inference action layout names and dimensions are invalid.")
    dof_order = value["dof_order"]
    if (
        not isinstance(dof_order, list)
        or not all(isinstance(name, str) and name for name in dof_order)
        or len(dof_order) != len(set(dof_order))
        or set(dof_order) != set(dof_dims)
    ):
        raise ValueError(
            "Inference action layout dof_order must list every DoF exactly once."
        )
    if sum(dimensions) != config.action_dim:
        raise ValueError(
            "Inference action layout dimension does not match model_config action_dim."
        )
    if not _deep_equal(value, config.action_layout):
        raise ValueError(
            "Inference action layout does not match model_config action_layout."
        )


def _validate_embodiment_registry(
    value: Any,
    *,
    config: XTokenizerModelConfig,
) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("Inference embodiment_registry must be a non-empty mapping.")
    ids: list[int] = []
    for name, raw_id in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Inference embodiment_registry names must be non-empty.")
        embodiment_id = _strict_int(
            raw_id,
            label=f"embodiment_registry[{name!r}]",
        )
        if not 0 <= embodiment_id < config.embodiment_vocab_size:
            raise ValueError(
                "Inference embodiment_registry id is outside model_config vocabulary."
            )
        ids.append(embodiment_id)
    if len(ids) != len(set(ids)):
        raise ValueError("Inference embodiment_registry ids must be unique.")


def _validate_normalization(
    value: Any,
    *,
    config: XTokenizerModelConfig,
    action_key: str,
    expected_modes: Mapping[str, NormMode] | None,
    canonical_stats_identity: Mapping[str, Any] | None,
) -> None:
    if not isinstance(value, dict):
        raise ValueError("Inference normalization must be a mapping.")
    paper_mode = value.get("kind") == _NORMALIZATION_KIND
    expected_fields = (
        {"stats", "modes", "kind", "roles", "identity"}
        if paper_mode
        else {"stats", "modes"}
    )
    if set(value) != expected_fields:
        if paper_mode:
            raise ValueError(
                "Inference normalization dual-role metadata must contain stats, "
                "modes, kind, roles, and identity."
            )
        raise ValueError("Inference normalization must contain stats and modes.")
    stats = value["stats"]
    modes = value["modes"]
    if not isinstance(stats, dict) or not isinstance(modes, dict):
        raise TypeError("Inference normalization stats and modes must be mappings.")
    if not stats or not modes:
        raise ValueError("Inference normalization stats and modes must be non-empty.")
    if set(stats) != set(modes):
        raise ValueError("Inference normalization stats and modes keys must match.")
    required_keys = {action_key, "observation.state"}
    if set(modes) != required_keys:
        raise ValueError(
            "Inference normalization keys must exactly match the relative action "
            "and absolute observation.state contracts."
        )
    if paper_mode:
        expected_roles = {
            "predict_action": {
                "key": action_key,
                "space": "observation_relative",
            },
            "observation_state": {
                "key": "observation.state",
                "space": "absolute",
            },
        }
        if not _deep_equal(value["roles"], expected_roles):
            raise ValueError(
                "Inference normalization roles must exactly bind predict_action to "
                "observation-relative action and observation_state to absolute "
                "observation.state."
            )
        identity = value["identity"]
        if not isinstance(identity, dict) or set(identity) != {
            "format_version",
            "content_fingerprint",
            "stats_horizon",
        }:
            raise ValueError("Inference normalization identity fields are malformed.")
        if canonical_stats_identity is None:
            raise ValueError(
                "Inference normalization identity requires canonical statistics."
            )
        canonical_normalization = canonical_stats_identity["normalization"]
        expected_identity = {
            "format_version": canonical_stats_identity["format_version"],
            "content_fingerprint": canonical_stats_identity["content_fingerprint"],
            "stats_horizon": canonical_normalization["stats_horizon"],
        }
        if not _deep_equal(identity, expected_identity):
            raise ValueError(
                "Inference normalization content_fingerprint or stats_horizon does "
                "not match canonical statistics identity."
            )
    for key, mode_raw in modes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Inference normalization keys must be non-empty strings.")
        try:
            mode = NormMode(mode_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Inference normalization mode for {key!r} is invalid."
            ) from exc
        expected_mode = None if expected_modes is None else expected_modes.get(key)
        if expected_mode is not None and mode != expected_mode:
            raise ValueError(
                "Inference normalization mode must match policy_io action feature."
            )
        stat_block = stats[key]
        if not isinstance(stat_block, dict):
            raise TypeError(
                f"Inference normalization stats for {key!r} must be a mapping."
            )
        required = (
            {"min", "max"}
            if mode == NormMode.MIN_MAX
            else {"mean", "std"}
            if mode == NormMode.MEAN_STD
            else set()
        )
        if mode == NormMode.NONE and stat_block:
            raise ValueError(
                f"Inference normalization NONE mode for {key!r} permits no statistics."
            )
        if set(stat_block) != required:
            raise ValueError(
                f"Inference normalization for {key!r} must contain exactly "
                f"{sorted(required)}."
            )
        for stat_name, values in stat_block.items():
            if not isinstance(values, list) or len(values) != config.action_dim:
                raise ValueError(
                    "Inference normalization statistic "
                    f"{key}.{stat_name} must match action_dim."
                )
            if not all(
                isinstance(item, int | float)
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in values
            ):
                raise ValueError(
                    f"Inference normalization statistic {key}.{stat_name} is invalid."
                )
        if mode == NormMode.MIN_MAX and any(
            float(minimum) >= float(maximum)
            for minimum, maximum in zip(
                stat_block["min"],
                stat_block["max"],
                strict=True,
            )
        ):
            raise ValueError(
                f"Inference normalization for {key!r} requires elementwise min < max."
            )
        if mode == NormMode.MEAN_STD and any(
            float(std) <= 0 for std in stat_block["std"]
        ):
            raise ValueError(
                f"Inference normalization for {key!r} requires elementwise std > 0."
            )
    if paper_mode:
        if any(NormMode(mode) is not NormMode.MIN_MAX for mode in modes.values()):
            raise ValueError(
                "Inference dual-role normalization requires min_max for both roles."
            )
        assert canonical_stats_identity is not None
        canonical_normalization = canonical_stats_identity["normalization"]
        expected_stats = {
            action_key: {
                "min": torch.tensor(
                    canonical_normalization["predict_action"]["q001"],
                    dtype=torch.float32,
                ).tolist(),
                "max": torch.tensor(
                    canonical_normalization["predict_action"]["q999"],
                    dtype=torch.float32,
                ).tolist(),
            },
            "observation.state": {
                "min": torch.tensor(
                    canonical_normalization["observation_state"]["q001"],
                    dtype=torch.float32,
                ).tolist(),
                "max": torch.tensor(
                    canonical_normalization["observation_state"]["q999"],
                    dtype=torch.float32,
                ).tolist(),
            },
        }
        if not _deep_equal(stats, expected_stats):
            raise ValueError(
                "Inference normalization numeric roles do not match canonical "
                "statistics identity."
            )
    elif any(NormMode(mode) is not NormMode.NONE for mode in modes.values()):
        raise ValueError(
            "Inference normalization selects paper mode only through explicit "
            f"{_NORMALIZATION_KIND!r} metadata."
        )


def _validate_policy_io(
    value: Any,
    *,
    config: XTokenizerModelConfig,
) -> PolicyIO | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Inference policy_io must be a mapping or null.")
    try:
        policy_io = PolicyIO.from_dict(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Inference policy_io is invalid: {exc}") from exc
    if policy_io.to_dict() != value:
        raise ValueError("Inference policy_io must be complete and canonical.")
    if (
        policy_io.action_feature is None
        or policy_io.action_dim != config.action_dim
        or policy_io.n_action_steps != config.action_horizon
    ):
        raise ValueError(
            "Inference policy_io action layout does not match model_config."
        )
    return policy_io


def _read_json_object(path: Path) -> dict[str, Any]:
    def _reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant {value!r} is not allowed.")

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} is not allowed.")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(),
            parse_constant=_reject_constant,
            object_pairs_hook=_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse inference manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("Inference manifest root must be a JSON object.")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(
        value,
        indent=2,
        allow_nan=False,
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content + "\n")
        handle.flush()


def _json_normalization_stats(
    stats: Mapping[str, Mapping[str, Tensor]],
) -> dict[str, dict[str, list[float]]]:
    return {
        str(key): {
            str(stat_name): [
                float(item)
                for item in torch.as_tensor(values, dtype=torch.float32)
                .reshape(-1)
                .tolist()
            ]
            for stat_name, values in stat_block.items()
        }
        for key, stat_block in stats.items()
    }


def _select_stats_for_normalization_modes(
    stats: Mapping[str, Mapping[str, list[float]]],
    modes: Mapping[str, str],
) -> dict[str, dict[str, list[float]]]:
    selected: dict[str, dict[str, list[float]]] = {}
    for key, mode_raw in modes.items():
        mode = NormMode(mode_raw)
        required = (
            ("min", "max")
            if mode is NormMode.MIN_MAX
            else ("mean", "std")
            if mode is NormMode.MEAN_STD
            else ()
        )
        block = stats.get(key, {})
        selected[key] = {name: list(block[name]) for name in required}
    return selected


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    converted = OmegaConf.to_container(OmegaConf.create(dict(value)), resolve=True)
    if not isinstance(converted, dict):
        raise TypeError("Expected metadata mapping.")
    return cast(dict[str, Any], converted)


def _resolve_action_layout(value: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(value)
    dof_dims = resolved.get("dof_dims")
    if isinstance(dof_dims, dict) and "dof_order" not in resolved:
        resolved["dof_order"] = list(dof_dims)
    return resolved


def _deep_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _deep_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _deep_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)


def _sha256_codec_state(
    codec: XTokenizerCodec,
    state_dict: Mapping[str, Tensor],
    *,
    normalization: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256(b"praxis-xtokenizer-state-v1\0")
    metadata = json.dumps(
        {
            "model_config": codec.config.to_dict(),
            "normalization": normalization,
            "policy_io": codec.io.to_dict() if codec.io is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        encoded_key = key.encode()
        digest.update(len(encoded_key).to_bytes(8, "big"))
        digest.update(encoded_key)
        descriptor = f"{value.dtype}:{tuple(value.shape)}".encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _submodule_state(
    state_dict: Mapping[str, Tensor],
    prefix: str,
) -> dict[str, Tensor]:
    marker = f"{prefix}."
    return {
        key.removeprefix(marker): value
        for key, value in state_dict.items()
        if key.startswith(marker)
    }


def _validate_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Inference {label} must be a lowercase SHA256 digest.")


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    return value
