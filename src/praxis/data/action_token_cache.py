# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Read immutable offline action-token targets."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from lerobot.datasets.sampler import EpisodeAwareSampler
from torch.utils.data import DataLoader, Dataset

from praxis.data.xtokenizer_canonical_stats import (
    validate_xtokenizer_quantile_identity,
)

_ACTION_SEMANTICS = {
    "public_action": "absolute",
    "tokenized_action": "observation_relative",
    "observation_anchor": "absolute",
    "relative_reference": "single_pre_chunk_observation",
}
_COMMON_MANIFEST_FIELDS = {
    "format_version",
    "tokenizer",
    "action_semantics",
    "robot_type_id",
    "codebook_size",
    "num_quantizers",
    "compression_ratio",
    "action_horizon",
    "token_order",
    "num_samples",
    "num_valid_samples",
    "tokens_file",
    "tokens_sha256",
    "split",
    "split_seed",
    "val_ratio",
    "episode_ids_sha256",
    "global_indices_sha256",
    "first_global_index",
    "last_global_index",
    "active_dofs",
    "inactive_normalized_fill",
    "tail_token_fill",
    "gripper_clip",
    "max_valid_windows",
}
_LEGACY_MANIFEST_FIELDS = _COMMON_MANIFEST_FIELDS | {
    "source_revision",
    "hf_revision",
    "checkpoint_sha256",
    "dataset_repo_id",
    "statistics_sha256",
    "statistics_key",
}
_NATIVE_MANIFEST_FIELDS = _COMMON_MANIFEST_FIELDS | {
    "dataset_repo_id",
    "dataset_revision",
    "canonical_stats_identity",
    "content_fingerprint",
    "stats_horizon",
    "vl_sidecar_identity",
    "artifact_sha256",
    "source_checkpoint_sha256",
    "normalization_source",
    "normalization_file",
    "normalization_sha256",
    "source_action_sha256",
    "source_observation_state_sha256",
}
_NATIVE_ONLY_FIELDS = _NATIVE_MANIFEST_FIELDS - _LEGACY_MANIFEST_FIELDS
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
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


@dataclass(frozen=True)
class _BoundNativeActionTokenCache:
    _tokens: np.ndarray = field(repr=False)
    _valid_rows: np.ndarray = field(repr=False)

    def token_ids(self, index: int) -> np.ndarray:
        if index < 0:
            index += len(self._valid_rows)
        if index < 0 or index >= len(self._valid_rows):
            raise IndexError("Native action-token cache row is out of range.")
        if not self._valid_rows[index]:
            raise IndexError(
                "Native action-token cache row is not a complete action window."
            )
        return np.asarray(self._tokens[index]).reshape(-1)


@dataclass(frozen=True)
class ActionTokenCache:
    root: Path
    manifest: dict[str, Any]
    _tokens: np.ndarray = field(repr=False)
    _is_native: bool = field(repr=False)

    @classmethod
    def open(cls, root: str | Path) -> ActionTokenCache:
        root = Path(root)
        manifest = json.loads((root / "manifest.json").read_text())
        if not isinstance(manifest, dict):
            raise ValueError("Action-token cache manifest must be an object.")
        if manifest.get("format_version") != 2:
            raise ValueError(
                "Unsupported action-token cache "
                f"format_version={manifest.get('format_version')!r}"
            )
        is_native = bool(set(manifest) & _NATIVE_ONLY_FIELDS)
        expected_fields = (
            _NATIVE_MANIFEST_FIELDS if is_native else _LEGACY_MANIFEST_FIELDS
        )
        missing = sorted(expected_fields - set(manifest))
        unexpected = sorted(set(manifest) - expected_fields)
        if missing or unexpected:
            raise ValueError(
                "Action-token cache manifest fields are malformed: "
                f"missing={missing}, unexpected={unexpected}."
            )
        if manifest["tokenizer"] != "x-tokenizer":
            raise ValueError(f"Unsupported tokenizer={manifest['tokenizer']!r}")
        if manifest["action_semantics"] != _ACTION_SEMANTICS:
            raise ValueError(
                "X-Tokenizer cache action_semantics must declare public absolute "
                "I/O and observation-relative tokenization."
            )
        if manifest["token_order"] != "time_major":
            raise ValueError("X-Tokenizer cache token_order must be 'time_major'.")
        action_horizon = _positive_int(manifest["action_horizon"], "action_horizon")
        compression_ratio = _positive_int(
            manifest["compression_ratio"],
            "compression_ratio",
        )
        num_quantizers = _positive_int(
            manifest["num_quantizers"],
            "num_quantizers",
        )
        codebook_size = _positive_int(manifest["codebook_size"], "codebook_size")
        num_samples = _nonnegative_int(manifest["num_samples"], "num_samples")
        num_valid_samples = _nonnegative_int(
            manifest["num_valid_samples"],
            "num_valid_samples",
        )
        if num_valid_samples > num_samples:
            raise ValueError(
                "Action-token cache num_valid_samples exceeds num_samples."
            )
        _validate_sha256(manifest["tokens_sha256"], label="tokens_sha256")
        tokens_path = _confined_token_path(root, manifest["tokens_file"])
        actual_tokens_sha256 = _sha256_file(tokens_path)
        if actual_tokens_sha256 != manifest["tokens_sha256"]:
            raise ValueError(
                "Action-token cache tokens_sha256 mismatch: "
                f"expected {manifest['tokens_sha256']}, got {actual_tokens_sha256}."
            )
        if is_native:
            _validate_native_manifest(root, manifest)
        else:
            _validate_legacy_manifest(manifest)

        tokens = np.load(tokens_path, mmap_mode="r")
        latent_horizon, remainder = divmod(
            action_horizon,
            compression_ratio,
        )
        expected = (
            num_samples,
            latent_horizon,
            num_quantizers,
        )
        if remainder or tokens.shape != expected:
            raise ValueError(
                f"Action-token cache token shape must be {expected}, got {tokens.shape}."
            )
        if is_native and tokens.dtype != np.dtype(np.uint16):
            raise ValueError(
                f"Native action-token cache dtype must be uint16, got {tokens.dtype}."
            )
        if not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError(
                f"Action-token cache dtype must be integer, got {tokens.dtype}."
            )
        if tokens.size and (
            int(tokens.min()) < 0 or int(tokens.max()) >= codebook_size
        ):
            raise ValueError(
                "Action-token cache contains ids outside the configured codebook."
            )
        return cls(
            root=root,
            manifest=manifest,
            _tokens=tokens,
            _is_native=is_native,
        )

    def __len__(self) -> int:
        return int(self._tokens.shape[0])

    @property
    def tokens(self) -> np.ndarray:
        self._require_legacy_public_access()
        return self._tokens

    def token_ids(self, index: int) -> np.ndarray:
        self._require_legacy_public_access()
        return np.asarray(self._tokens[index]).reshape(-1)

    def _require_legacy_public_access(self) -> None:
        if self._is_native:
            raise RuntimeError(
                "Public native token access requires successful consumer binding."
            )

    def _bound_native_view(
        self,
        valid_rows: np.ndarray,
    ) -> _BoundNativeActionTokenCache:
        if not self._is_native:
            raise AssertionError("Only native caches require a bound token view.")
        immutable_valid_rows = np.array(valid_rows, dtype=np.bool_, copy=True)
        immutable_valid_rows.setflags(write=False)
        return _BoundNativeActionTokenCache(
            _tokens=self._tokens,
            _valid_rows=immutable_valid_rows,
        )


def _validate_native_manifest(root: Path, manifest: dict[str, Any]) -> None:
    if manifest["normalization_source"] != "inference_artifact":
        raise ValueError(
            "Native action-token cache normalization_source must be "
            "'inference_artifact'."
        )
    if manifest["max_valid_windows"] is not None:
        raise ValueError("Production native action-token caches must not be capped.")
    split = manifest["split"]
    if split not in {"train", "val"} or root.name != split:
        raise ValueError(
            "Native action-token cache split must match its cache directory."
        )
    _nonnegative_int(manifest["split_seed"], "split_seed")
    val_ratio = manifest["val_ratio"]
    if (
        not isinstance(val_ratio, int | float)
        or isinstance(val_ratio, bool)
        or not math.isfinite(float(val_ratio))
        or not 0.0 <= float(val_ratio) <= 1.0
    ):
        raise ValueError("Native action-token cache val_ratio must be within [0, 1].")
    for key in (
        "artifact_sha256",
        "source_checkpoint_sha256",
        "normalization_sha256",
        "content_fingerprint",
        "episode_ids_sha256",
        "global_indices_sha256",
        "source_action_sha256",
        "source_observation_state_sha256",
    ):
        _validate_sha256(manifest[key], label=key)
    stats_horizon = _positive_int(manifest["stats_horizon"], "stats_horizon")
    if stats_horizon != manifest["action_horizon"]:
        raise ValueError(
            "Native action-token cache stats_horizon must match action_horizon."
        )
    repo_id = manifest["dataset_repo_id"]
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("Native action-token cache dataset repository is malformed.")
    revision = manifest["dataset_revision"]
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Native action-token cache dataset revision is malformed.")

    canonical_identity = manifest["canonical_stats_identity"]
    try:
        dof_dims = canonical_identity["action_layout"]["dof_dims"]
        action_dim = sum(
            _positive_int(value, "dof dimension") for value in dof_dims.values()
        )
    except (AttributeError, KeyError, TypeError):
        raise ValueError(
            "Native action-token cache canonical statistics identity is malformed."
        ) from None
    canonical_identity = validate_xtokenizer_quantile_identity(
        canonical_identity,
        expected_action_dim=action_dim,
        expected_stats_horizon=stats_horizon,
    )
    if (
        canonical_identity["content_fingerprint"] != manifest["content_fingerprint"]
        or canonical_identity["dataset_repo_id"] != repo_id
        or canonical_identity["dataset_revision"] != revision
    ):
        raise ValueError(
            "Native action-token cache canonical statistics fingerprint or dataset "
            "identity does not match the manifest."
        )
    if manifest["normalization_file"] != "../normalization.json":
        raise ValueError(
            "Native action-token cache normalization_file is not the supported path."
        )
    normalization_path = root / manifest["normalization_file"]
    normalization = _authenticated_json(
        normalization_path,
        expected_sha256=manifest["normalization_sha256"],
        label="normalization",
    )
    _validate_native_normalization(
        normalization,
        canonical_identity=canonical_identity,
    )
    _validate_vl_sidecar_identity(
        manifest["vl_sidecar_identity"],
        dataset_repo_id=repo_id,
        dataset_revision=revision,
    )


def _validate_vl_sidecar_identity(
    value: Any,
    *,
    dataset_repo_id: str,
    dataset_revision: str,
) -> None:
    if not isinstance(value, dict) or set(value) != _VL_SIDECAR_IDENTITY_FIELDS:
        raise ValueError("Native cache VL sidecar identity fields are malformed.")
    if value["format_version"] != 3:
        raise ValueError("Native cache VL sidecar format_version is unsupported.")
    if (
        value["dataset_repo_id"] != dataset_repo_id
        or value["dataset_revision"] != dataset_revision
    ):
        raise ValueError(
            "Native cache VL sidecar dataset identity does not match the manifest."
        )
    for key in (
        "dataset_repo_id",
        "dataset_revision",
        "extractor_revision",
        "model_id",
        "model_revision",
        "global_instruction",
        "local_subtask",
    ):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"Native cache VL sidecar {key} is malformed.")
    for key in ("layer", "stride", "feature_dim"):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            raise ValueError(f"Native cache VL sidecar {key} is malformed.")
    if value["stride"] <= 0 or value["feature_dim"] <= 0:
        raise ValueError("Native cache VL sidecar geometry must be positive.")
    episode_indices = value["episode_indices"]
    episode_lengths = value["episode_lengths"]
    if (
        not isinstance(episode_indices, list)
        or not isinstance(episode_lengths, list)
        or len(episode_indices) != len(episode_lengths)
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 0
            for item in episode_indices
        )
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in episode_lengths
        )
        or len(set(episode_indices)) != len(episode_indices)
    ):
        raise ValueError("Native cache VL sidecar episode metadata is malformed.")
    image_keys = value["image_keys"]
    if (
        not isinstance(image_keys, list)
        or any(not isinstance(item, str) or not item.strip() for item in image_keys)
        or len(set(image_keys)) != len(image_keys)
    ):
        raise ValueError("Native cache VL sidecar image_keys are malformed.")
    if not isinstance(value["synthetic"], bool):
        raise ValueError("Native cache VL sidecar synthetic flag is malformed.")
    for key in (
        "generation_fingerprint",
        "features_sha256",
        "next_features_sha256",
        "valid_sha256",
    ):
        _validate_sha256(value[key], label=f"VL sidecar {key}")
    from praxis.data.xtokenizer_vl import vl_sidecar_generation_fingerprint

    if value["generation_fingerprint"] != vl_sidecar_generation_fingerprint(value):
        raise ValueError("Native cache VL sidecar generation fingerprint mismatch.")


def _validate_native_normalization(
    value: Any,
    *,
    canonical_identity: dict[str, Any],
) -> None:
    expected_fields = {"kind", "roles", "stats", "modes", "identity"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("Native cache normalization fields are malformed.")
    if value["kind"] != "xtokenizer_dual_role_v1":
        raise ValueError("Native cache normalization requires xtokenizer_dual_role_v1.")
    expected_modes = {"action": "min_max", "observation.state": "min_max"}
    if value["modes"] != expected_modes:
        raise ValueError("Native cache normalization rejects NONE normalization roles.")
    if value["roles"] != {
        "predict_action": {
            "key": "action",
            "space": "observation_relative",
        },
        "observation_state": {
            "key": "observation.state",
            "space": "absolute",
        },
    }:
        raise ValueError("Native cache normalization roles are malformed.")
    canonical_normalization = canonical_identity["normalization"]
    if value["identity"] != {
        "format_version": canonical_identity["format_version"],
        "content_fingerprint": canonical_identity["content_fingerprint"],
        "stats_horizon": canonical_normalization["stats_horizon"],
    }:
        raise ValueError("Native cache normalization identity is malformed.")
    expected_stats = {
        "action": {
            "min": canonical_normalization["predict_action"]["q001"],
            "max": canonical_normalization["predict_action"]["q999"],
        },
        "observation.state": {
            "min": canonical_normalization["observation_state"]["q001"],
            "max": canonical_normalization["observation_state"]["q999"],
        },
    }
    if value["stats"] != expected_stats:
        raise ValueError(
            "Native cache normalization statistics do not match the canonical identity."
        )


def _validate_legacy_manifest(manifest: dict[str, Any]) -> None:
    for key in (
        "tokens_sha256",
        "checkpoint_sha256",
        "statistics_sha256",
        "episode_ids_sha256",
        "global_indices_sha256",
    ):
        _validate_sha256(manifest[key], label=key)


def _authenticated_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Native cache {label} file is unavailable.") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Native cache {label} SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual}."
        )
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Native cache {label} JSON is malformed.") from exc


def _confined_token_path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Action-token cache tokens_file must be a relative path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            "Action-token cache tokens_file must be a confined relative path."
        )
    resolved_root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        raise ValueError(
            "Action-token cache tokens_file must remain confined to the cache root."
        ) from None
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Action-token cache token file is missing: {candidate}"
        )
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"Action-token cache {label} must be a lowercase SHA256 digest."
        )


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Action-token cache {label} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Action-token cache {label} must be a non-negative integer.")
    return value


class _DatasetWithLength(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> dict[str, Any]: ...


class CachedActionTokenDataset(Dataset[dict[str, Any]]):
    """Attach cached token targets while preserving the continuous action sample."""

    def __init__(
        self,
        dataset: _DatasetWithLength,
        cache: ActionTokenCache,
        *,
        expected_split: str | None = None,
        expected_split_seed: int | None = None,
        expected_val_ratio: float | None = None,
        expected_artifact_sha256: str | None = None,
    ) -> None:
        if len(dataset) != len(cache):
            raise ValueError(
                f"Dataset/cache length mismatch: {len(dataset)} != {len(cache)}"
            )
        self.dataset = dataset
        self.cache = cache
        self._bound_native_cache: _BoundNativeActionTokenCache | None = None
        if cache.manifest.get("normalization_source") == "inference_artifact":
            self._bound_native_cache = _bind_native_cache_to_dataset(
                dataset,
                cache,
                expected_split=expected_split,
                expected_split_seed=expected_split_seed,
                expected_val_ratio=expected_val_ratio,
                expected_artifact_sha256=expected_artifact_sha256,
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        token_values = (
            self._bound_native_cache.token_ids(index)
            if self._bound_native_cache is not None
            else self.cache.token_ids(index)
        )
        sample = dict(self.dataset[index])
        token_ids = torch.from_numpy(token_values.copy()).long()
        sample["action_token_ids"] = token_ids
        sample["action_token_length"] = torch.tensor(
            token_ids.numel(), dtype=torch.long
        )
        return sample

    def make_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle: bool = True,
        pin_memory: bool = True,
        prefetch_factor: int | None = None,
        use_episode_aware_sampler: bool = True,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        drop_last: bool = True,
    ) -> DataLoader:
        """Create a loader whose dataset remains this cache wrapper."""
        sampler = None
        if self._bound_native_cache is not None:
            if not use_episode_aware_sampler:
                raise ValueError(
                    "Native cached loading requires the episode-aware sampler to "
                    "exclude incomplete tail windows."
                )
            drop_n_last_frames = max(
                drop_n_last_frames,
                int(self.cache.manifest["action_horizon"]) - 1,
            )
        if use_episode_aware_sampler:
            source = getattr(self.dataset, "lerobot_dataset", None)
            if source is None:
                raise ValueError(
                    "Episode-aware cached loading requires a LeRobot-backed dataset."
                )
            episode_indices = source.hf_dataset["episode_index"]
            from_indices, to_indices = _episode_bounds(episode_indices)
            sampler = EpisodeAwareSampler(
                dataset_from_indices=from_indices,
                dataset_to_indices=to_indices,
                drop_n_first_frames=drop_n_first_frames,
                drop_n_last_frames=drop_n_last_frames,
                shuffle=shuffle,
            )
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last,
            prefetch_factor=prefetch_factor,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def _bind_native_cache_to_dataset(
    dataset: _DatasetWithLength,
    cache: ActionTokenCache,
    *,
    expected_split: str | None,
    expected_split_seed: int | None,
    expected_val_ratio: float | None,
    expected_artifact_sha256: str | None,
) -> _BoundNativeActionTokenCache:
    _validate_sha256(
        expected_artifact_sha256,
        label="expected_artifact_sha256",
    )
    if expected_artifact_sha256 != cache.manifest["artifact_sha256"]:
        raise ValueError(
            "Native action-token cache artifact_sha256 does not match the trusted "
            "expected_artifact_sha256."
        )
    source = getattr(dataset, "lerobot_dataset", None)
    if source is None:
        raise ValueError(
            "Native action-token caches require a LeRobot-backed consumer dataset."
        )
    manifest = cache.manifest
    repo_id = getattr(source, "repo_id", None)
    if not isinstance(repo_id, str) or repo_id != manifest["dataset_repo_id"]:
        raise ValueError(
            "Native action-token cache dataset repository does not match the consumer."
        )
    expected_revision = manifest["dataset_revision"]
    root = getattr(source, "root", None)
    from praxis.data.lerobot_adapter import read_lerobot_cache_provenance

    revision = read_lerobot_cache_provenance(root=root, repo_id=repo_id)
    if revision != expected_revision:
        raise ValueError(
            "Native action-token cache dataset revision does not match the consumer."
        )
    if expected_split != manifest["split"]:
        raise ValueError(
            "Native action-token cache split does not match the consumer split."
        )
    if expected_split_seed != manifest["split_seed"]:
        raise ValueError(
            "Native action-token cache split_seed does not match the consumer split."
        )
    if expected_val_ratio != manifest["val_ratio"]:
        raise ValueError(
            "Native action-token cache val_ratio does not match the consumer split."
        )

    hf_dataset = getattr(source, "hf_dataset", None)
    if hf_dataset is None:
        raise ValueError(
            "Native action-token cache consumer is missing index/episode_index."
        )
    try:
        global_indices = _int64_values(hf_dataset["index"])
        episode_indices = _int64_values(hf_dataset["episode_index"])
        source_action_sha256 = sha256_float32_matrix(hf_dataset["action"])
        source_observation_state_sha256 = sha256_float32_matrix(
            hf_dataset["observation.state"]
        )
    except (KeyError, TypeError):
        raise ValueError(
            "Native action-token cache consumer is missing source identity columns."
        ) from None
    if len(global_indices) != len(cache) or len(episode_indices) != len(cache):
        raise ValueError("Native action-token cache consumer row count is malformed.")
    if _sha256_ints(global_indices) != manifest["global_indices_sha256"]:
        raise ValueError(
            "Native action-token cache global row order does not match the consumer."
        )
    if source_action_sha256 != manifest["source_action_sha256"]:
        raise ValueError(
            "Native action-token cache source action values do not match the consumer."
        )
    if source_observation_state_sha256 != manifest["source_observation_state_sha256"]:
        raise ValueError(
            "Native action-token cache source observation.state values do not match "
            "the consumer."
        )
    episode_order = np.asarray(
        list(dict.fromkeys(int(value) for value in episode_indices)),
        dtype=np.int64,
    )
    if _sha256_ints(episode_order) != manifest["episode_ids_sha256"]:
        raise ValueError(
            "Native action-token cache episode identity does not match the consumer."
        )
    expected_first = int(global_indices[0]) if len(global_indices) else None
    expected_last = int(global_indices[-1]) if len(global_indices) else None
    if (
        manifest["first_global_index"] != expected_first
        or manifest["last_global_index"] != expected_last
    ):
        raise ValueError(
            "Native action-token cache global index bounds do not match the consumer."
        )
    io = getattr(dataset, "io", None)
    consumer_horizon = getattr(io, "n_action_steps", None)
    if consumer_horizon != manifest["action_horizon"]:
        raise ValueError(
            "Native action-token cache horizon does not match the consumer dataset."
        )

    valid_rows = np.zeros(len(cache), dtype=np.bool_)
    boundaries = np.flatnonzero(episode_indices[1:] != episode_indices[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(episode_indices)]))
    sidecar = manifest["vl_sidecar_identity"]
    sidecar_lengths = dict(
        zip(sidecar["episode_indices"], sidecar["episode_lengths"], strict=True)
    )
    for start, end in zip(starts, ends, strict=True):
        episode_index = int(episode_indices[int(start)])
        if sidecar_lengths.get(episode_index) != int(end - start):
            raise ValueError(
                "Native action-token cache VL sidecar episode identity does not "
                "match the consumer."
            )
    horizon = int(manifest["action_horizon"])
    for start, end in zip(starts, ends, strict=True):
        valid_end = max(int(start), int(end) - horizon + 1)
        valid_rows[int(start) : valid_end] = True
    if int(valid_rows.sum()) != manifest["num_valid_samples"]:
        raise ValueError(
            "Native action-token cache valid-window count does not match the consumer."
        )
    return cache._bound_native_view(valid_rows)


def _int64_values(values: Any) -> np.ndarray:
    return np.asarray(
        [
            int(value.item()) if isinstance(value, torch.Tensor) else int(value)
            for value in values
        ],
        dtype=np.int64,
    )


def _sha256_ints(values: np.ndarray) -> str:
    payload = np.ascontiguousarray(values, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def sha256_float32_matrix(values: Any) -> str:
    """Hash one finite float32 matrix with shape and canonical C-order bytes."""
    if isinstance(values, torch.Tensor):
        raw = values.detach().cpu().numpy()
    elif isinstance(values, np.ndarray):
        raw = values
    else:
        rows = [
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else np.asarray(value)
            for value in values
        ]
        raw = np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32)
    matrix = np.ascontiguousarray(raw, dtype=np.dtype("<f4"))
    if matrix.ndim != 2:
        raise ValueError("Source identity values must be a rank-2 float32 matrix.")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError("Source identity float32 matrix must contain finite values.")
    digest = hashlib.sha256(b"praxis-action-token-source-matrix-v1\0")
    for dimension in matrix.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _episode_bounds(
    episode_indices: list[int] | list[torch.Tensor],
) -> tuple[list[int], list[int]]:
    if not episode_indices:
        return [], []
    episode_ids = [
        int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        for value in episode_indices
    ]
    from_indices = [0]
    to_indices: list[int] = []
    current = episode_ids[0]
    for index, episode_id in enumerate(episode_ids[1:], start=1):
        if episode_id != current:
            to_indices.append(index)
            from_indices.append(index)
            current = episode_id
    to_indices.append(len(episode_ids))
    return from_indices, to_indices
