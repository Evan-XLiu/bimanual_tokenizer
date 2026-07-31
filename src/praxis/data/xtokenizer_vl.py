# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Validated mmap reader for offline X-Tokenizer vision-language features."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

_FORMAT_VERSION = 3
_SNAPSHOT_OPEN_ATTEMPTS = 3
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SEMANTIC_FIELDS = (
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
)
_DIGEST_FIELDS = (
    "features_sha256",
    "next_features_sha256",
    "valid_sha256",
)
_FILE_FIELDS = (
    "features_file",
    "next_features_file",
    "valid_file",
)
_MANIFEST_FIELDS = {
    *_SEMANTIC_FIELDS,
    *_DIGEST_FIELDS,
    *_FILE_FIELDS,
    "generation_fingerprint",
}
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


@dataclass(frozen=True)
class VLFeatureManifest:
    """Metadata that binds a feature sidecar to its source and extractor."""

    format_version: int
    dataset_repo_id: str
    dataset_revision: str
    extractor_revision: str
    model_id: str
    model_revision: str
    layer: int
    stride: int
    feature_dim: int
    episode_indices: tuple[int, ...]
    episode_lengths: tuple[int, ...]
    image_keys: tuple[str, ...]
    global_instruction: str
    local_subtask: str
    synthetic: bool
    generation_fingerprint: str
    features_sha256: str
    next_features_sha256: str
    valid_sha256: str
    features_file: str
    next_features_file: str
    valid_file: str

    @classmethod
    def from_path(cls, path: Path) -> VLFeatureManifest:
        """Load and validate a sidecar manifest."""
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"VL feature manifest does not exist: {path}"
            ) from None
        except json.JSONDecodeError as exc:
            raise ValueError(f"VL feature manifest is not valid JSON: {path}") from exc
        if not isinstance(raw, dict) or set(raw) != _MANIFEST_FIELDS:
            raise ValueError("VL feature manifest fields are malformed.")

        format_version = _required_int(raw, "format_version")
        if format_version != _FORMAT_VERSION:
            raise ValueError(
                "Unsupported VL feature manifest format_version="
                f"{format_version!r}; expected {_FORMAT_VERSION}."
            )

        stride = _required_positive_int(raw, "stride")
        feature_dim = _required_positive_int(raw, "feature_dim")
        episode_indices = _required_int_tuple(raw, "episode_indices")
        episode_lengths = _required_int_tuple(raw, "episode_lengths")
        if len(episode_indices) != len(episode_lengths):
            raise ValueError(
                "VL feature manifest episode_indices and episode_lengths must "
                "have the same length."
            )
        if len(set(episode_indices)) != len(episode_indices):
            raise ValueError("VL feature manifest episode_indices must be unique.")
        if any(index < 0 for index in episode_indices):
            raise ValueError(
                "VL feature manifest episode_indices must be non-negative."
            )
        if any(length <= 0 for length in episode_lengths):
            raise ValueError(
                "VL feature manifest episode_lengths must contain positive values."
            )

        manifest = cls(
            format_version=format_version,
            dataset_repo_id=_required_str(raw, "dataset_repo_id"),
            dataset_revision=_required_str(raw, "dataset_revision"),
            extractor_revision=_required_str(raw, "extractor_revision"),
            model_id=_required_str(raw, "model_id"),
            model_revision=_required_str(raw, "model_revision"),
            layer=_required_int(raw, "layer"),
            stride=stride,
            feature_dim=feature_dim,
            episode_indices=episode_indices,
            episode_lengths=episode_lengths,
            image_keys=_required_str_tuple(raw, "image_keys"),
            global_instruction=_required_str(raw, "global_instruction"),
            local_subtask=_required_str(raw, "local_subtask"),
            synthetic=_required_bool(raw, "synthetic"),
            generation_fingerprint=_required_sha256(raw, "generation_fingerprint"),
            features_sha256=_required_sha256(raw, "features_sha256"),
            next_features_sha256=_required_sha256(raw, "next_features_sha256"),
            valid_sha256=_required_sha256(raw, "valid_sha256"),
            features_file=_required_relative_file(raw, "features_file"),
            next_features_file=_required_relative_file(raw, "next_features_file"),
            valid_file=_required_relative_file(raw, "valid_file"),
        )
        expected_fingerprint = vl_sidecar_generation_fingerprint(raw)
        if manifest.generation_fingerprint != expected_fingerprint:
            raise ValueError("VL feature manifest generation fingerprint mismatch.")
        return manifest

    @property
    def identity(self) -> dict[str, Any]:
        """Return the path-independent, byte-authenticated sidecar identity."""
        return {
            "format_version": self.format_version,
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_revision": self.dataset_revision,
            "extractor_revision": self.extractor_revision,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "layer": self.layer,
            "stride": self.stride,
            "feature_dim": self.feature_dim,
            "episode_indices": list(self.episode_indices),
            "episode_lengths": list(self.episode_lengths),
            "image_keys": list(self.image_keys),
            "global_instruction": self.global_instruction,
            "local_subtask": self.local_subtask,
            "synthetic": self.synthetic,
            "generation_fingerprint": self.generation_fingerprint,
            "features_sha256": self.features_sha256,
            "next_features_sha256": self.next_features_sha256,
            "valid_sha256": self.valid_sha256,
        }

    def validate_expected(
        self,
        *,
        dataset_revision: str,
        dataset_repo_id: str | None = None,
        model_id: str | None = None,
        extractor_revision: str | None = None,
        model_revision: str | None = None,
        image_keys: tuple[str, ...] | None = None,
        global_instruction: str | None = None,
        local_subtask: str | None = None,
        feature_dim: int | None = None,
        stride: int | None = None,
    ) -> None:
        """Reject sidecars that do not match the active training contract."""
        if dataset_repo_id is not None:
            _require_match(
                "dataset repository",
                self.dataset_repo_id,
                dataset_repo_id,
            )
        if model_id is not None:
            _require_match("model identifier", self.model_id, model_id)
        _require_match(
            "dataset revision",
            self.dataset_revision,
            dataset_revision,
        )
        if extractor_revision is not None:
            _require_match(
                "extractor revision",
                self.extractor_revision,
                extractor_revision,
            )
        if model_revision is not None:
            _require_match("model revision", self.model_revision, model_revision)
        if image_keys is not None:
            _require_match("image keys", self.image_keys, image_keys)
        if global_instruction is not None:
            _require_match(
                "global instruction",
                self.global_instruction,
                global_instruction,
            )
        if local_subtask is not None:
            _require_match("local subtask", self.local_subtask, local_subtask)
        if feature_dim is not None:
            _require_match("feature dimension", self.feature_dim, int(feature_dim))
        if stride is not None:
            _require_match("stride", self.stride, int(stride))


@dataclass(frozen=True)
class VLFeatureSidecar:
    """Memory-mapped per-VL-frame, per-view episode features."""

    root: Path
    manifest: VLFeatureManifest
    features: np.ndarray
    next_features: np.ndarray
    valid: np.ndarray
    _episode_rows: dict[int, int]

    @classmethod
    def open(
        cls,
        root: Path,
        dataset_revision: str,
        *,
        expected_dataset_repo_id: str | None = None,
        expected_model_id: str | None = None,
        expected_extractor_revision: str | None = None,
        expected_model_revision: str | None = None,
        expected_image_keys: tuple[str, ...] | None = None,
        expected_global_instruction: str | None = None,
        expected_local_subtask: str | None = None,
        expected_feature_dim: int | None = None,
        expected_stride: int | None = None,
    ) -> VLFeatureSidecar:
        """Open a sidecar and validate provenance and storage before sampling."""
        root = Path(root)
        for attempt in range(_SNAPSHOT_OPEN_ATTEMPTS):
            manifest = VLFeatureManifest.from_path(root / "manifest.json")
            manifest.validate_expected(
                dataset_revision=dataset_revision,
                dataset_repo_id=expected_dataset_repo_id,
                model_id=expected_model_id,
                extractor_revision=expected_extractor_revision,
                model_revision=expected_model_revision,
                image_keys=expected_image_keys,
                global_instruction=expected_global_instruction,
                local_subtask=expected_local_subtask,
                feature_dim=expected_feature_dim,
                stride=expected_stride,
            )
            try:
                _verify_file_sha256(
                    root,
                    manifest.features_file,
                    manifest.features_sha256,
                )
                _verify_file_sha256(
                    root,
                    manifest.next_features_file,
                    manifest.next_features_sha256,
                )
                _verify_file_sha256(
                    root,
                    manifest.valid_file,
                    manifest.valid_sha256,
                )
                features = _load_mmap_array(root, manifest.features_file)
                next_features = _load_mmap_array(root, manifest.next_features_file)
                valid = _load_mmap_array(root, manifest.valid_file)
                if VLFeatureManifest.from_path(root / "manifest.json") != manifest:
                    continue
            except FileNotFoundError:
                if attempt + 1 == _SNAPSHOT_OPEN_ATTEMPTS:
                    raise
                continue
            break
        else:
            raise RuntimeError("Unable to open a stable VL sidecar manifest snapshot.")
        feature_lengths = tuple(
            (length + manifest.stride - 1) // manifest.stride
            for length in manifest.episode_lengths
        )
        expected_prefix = (
            len(manifest.episode_indices),
            max(feature_lengths, default=0),
        )
        expected_features_shape = (
            *expected_prefix,
            len(manifest.image_keys),
            manifest.feature_dim,
        )
        if features.shape != expected_features_shape:
            raise ValueError(
                "VL feature dimension/shape mismatch: manifest requires "
                f"{expected_features_shape}, got {features.shape}."
            )
        if next_features.shape != expected_features_shape:
            raise ValueError(
                "VL next-feature dimension/shape mismatch: manifest requires "
                f"{expected_features_shape}, got {next_features.shape}."
            )
        expected_valid_shape = (*expected_prefix, len(manifest.image_keys))
        if valid.shape != expected_valid_shape:
            raise ValueError(
                "VL valid-mask shape mismatch: manifest requires "
                f"{expected_valid_shape}, got {valid.shape}."
            )
        if not np.issubdtype(features.dtype, np.floating):
            raise ValueError(
                f"VL features must have floating dtype, got {features.dtype}."
            )
        if not np.issubdtype(next_features.dtype, np.floating):
            raise ValueError(
                f"VL next features must have floating dtype, got {next_features.dtype}."
            )
        if valid.dtype != np.bool_:
            raise ValueError(
                f"VL valid mask must have boolean dtype, got {valid.dtype}."
            )
        for row, feature_length in enumerate(feature_lengths):
            if bool(np.asarray(valid[row, feature_length:]).any()):
                episode_index = manifest.episode_indices[row]
                raise ValueError(
                    "VL valid mask marks padded storage as valid for episode "
                    f"{episode_index} after VL frame {feature_length - 1}."
                )

        return cls(
            root=root,
            manifest=manifest,
            features=features,
            next_features=next_features,
            valid=valid,
            _episode_rows={
                episode_index: row
                for row, episode_index in enumerate(manifest.episode_indices)
            },
        )

    @property
    def feature_dim(self) -> int:
        return self.manifest.feature_dim

    @property
    def num_views(self) -> int:
        return len(self.manifest.image_keys)

    @property
    def stride(self) -> int:
        return self.manifest.stride

    def validate_episode_lengths(
        self,
        expected_episode_lengths: dict[int, int],
    ) -> None:
        """Validate selected dataset episode bounds before creating a loader."""
        for episode_index, expected_length in expected_episode_lengths.items():
            row = self._episode_rows.get(int(episode_index))
            if row is None:
                raise ValueError(
                    f"VL sidecar does not contain dataset episode {episode_index}."
                )
            actual_length = self.manifest.episode_lengths[row]
            if actual_length != int(expected_length):
                raise ValueError(
                    f"VL sidecar episode {episode_index} length mismatch: "
                    f"manifest={actual_length}, dataset={int(expected_length)}."
                )

    def lookup(
        self,
        episode_index: Tensor,
        vl_frame_index: Tensor,
    ) -> dict[str, Tensor]:
        """Look up episode/VL-frame pairs without fusing the view dimension."""
        _validate_index_tensor("episode_index", episode_index)
        _validate_index_tensor("vl_frame_index", vl_frame_index)
        if episode_index.shape != vl_frame_index.shape:
            raise ValueError(
                "episode_index and vl_frame_index must have the same shape, got "
                f"{tuple(episode_index.shape)} and {tuple(vl_frame_index.shape)}."
            )

        episodes = episode_index.detach().cpu().numpy().astype(np.int64, copy=False)
        frames = vl_frame_index.detach().cpu().numpy().astype(np.int64, copy=False)
        flat_episodes = episodes.reshape(-1)
        flat_frames = frames.reshape(-1)
        rows = np.empty(flat_episodes.shape, dtype=np.int64)
        next_in_bounds = np.empty(flat_episodes.shape, dtype=np.bool_)
        for position, episode in enumerate(flat_episodes):
            row = self._episode_rows.get(int(episode))
            if row is None:
                raise IndexError(
                    f"VL sidecar episode index {int(episode)} is out of bounds."
                )
            frame = int(flat_frames[position])
            episode_length = self.manifest.episode_lengths[row]
            feature_length = (episode_length + self.stride - 1) // self.stride
            if frame < 0 or frame >= feature_length:
                raise IndexError(
                    "VL sidecar VL frame index "
                    f"{frame} is out of bounds for episode {int(episode)} "
                    f"with feature length {feature_length}."
                )
            rows[position] = row
            next_in_bounds[position] = frame + 1 < feature_length

        selected_features = np.asarray(
            self.features[rows, flat_frames],
        ).reshape(*episodes.shape, self.num_views, self.feature_dim)
        selected_next = np.asarray(
            self.next_features[rows, flat_frames],
        ).reshape(*episodes.shape, self.num_views, self.feature_dim)
        selected_valid = np.asarray(
            self.valid[rows, flat_frames],
        ).reshape(*episodes.shape, self.num_views)
        next_frames = np.minimum(flat_frames + 1, self.valid.shape[1] - 1)
        selected_next_valid = np.asarray(
            self.valid[rows, next_frames],
        ).reshape(*episodes.shape, self.num_views)
        selected_next_is_pad = np.logical_or(
            np.logical_not(selected_next_valid),
            np.logical_not(next_in_bounds.reshape(*episodes.shape, 1)),
        )
        if not np.isfinite(selected_features).all():
            raise ValueError("VL sidecar lookup returned non-finite features.")
        if not np.isfinite(selected_next).all():
            raise ValueError("VL sidecar lookup returned non-finite next features.")

        return {
            "vl_features": torch.from_numpy(selected_features.copy()),
            "next_vl_features": torch.from_numpy(selected_next.copy()),
            "next_vl_feature_is_pad": torch.from_numpy(selected_next_is_pad.copy()),
            "vl_feature_is_pad": torch.from_numpy(
                np.logical_not(selected_valid).copy()
            ),
        }


def vl_sidecar_generation_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Hash all semantic and byte-digest fields, excluding paths and filenames."""
    payload = {key: manifest[key] for key in (*_SEMANTIC_FIELDS, *_DIGEST_FIELDS)}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the exact published file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_sha256(
    root: Path,
    relative_path: str,
    expected_sha256: str,
) -> None:
    path = _confined_path(root, relative_path)
    try:
        actual_sha256 = file_sha256(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"VL sidecar array does not exist: {path}") from None
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"VL sidecar array SHA-256 mismatch for {relative_path!r}: "
            f"manifest={expected_sha256}, actual={actual_sha256}."
        )


def _load_mmap_array(root: Path, relative_path: str) -> np.ndarray:
    path = _confined_path(root, relative_path)
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"VL sidecar array does not exist: {path}") from None
    if not isinstance(array, np.ndarray):
        raise ValueError(f"VL sidecar file must contain one NumPy array: {path}")
    return array


def _confined_path(root: Path, relative_path: str) -> Path:
    root_resolved = root.resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(
            f"VL sidecar file path escapes the configured root: {relative_path!r}."
        )
    return resolved


def _validate_index_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must have integer dtype, got {value.dtype}.")


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"VL feature manifest {key} must be a non-empty string.")
    return value


def _required_sha256(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"VL feature manifest {key} must be a lowercase SHA-256 digest."
        )
    return value


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"VL feature manifest {key} must be an integer.")
    return value


def _required_positive_int(raw: dict[str, Any], key: str) -> int:
    value = _required_int(raw, key)
    if value <= 0:
        raise ValueError(f"VL feature manifest {key} must be positive.")
    return value


def _required_int_tuple(raw: dict[str, Any], key: str) -> tuple[int, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"VL feature manifest {key} must be a list of integers.")
    return tuple(value)


def _required_str_tuple(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(
            f"VL feature manifest {key} must be a non-empty list of strings."
        )
    if len(set(value)) != len(value):
        raise ValueError(f"VL feature manifest {key} must contain unique values.")
    return tuple(value)


def _required_bool(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"VL feature manifest {key} must be a boolean.")
    return value


def _required_relative_file(raw: dict[str, Any], key: str) -> str:
    value = _required_str(raw, key)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"VL feature manifest {key} must be a confined relative path.")
    return value


def _require_match(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            f"VL sidecar {name} mismatch: manifest={actual!r}, expected={expected!r}."
        )
