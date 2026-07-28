# SPDX-License-Identifier: Apache-2.0

"""Strict offline normalization statistics for canonical X-Tokenizer inputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import torch

from praxis.core.types import NormMode

_FORMAT_VERSION = 2
_QUANTILE_RANGE = (0.001, 0.999)
_REPRESENTATIONS = {
    "public_action": "absolute",
    "action": "observation_relative",
    "observation.state": "absolute",
    "relative_reference": "single_pre_chunk_observation",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FIELDS = {
    "format_version",
    "dataset_repo_id",
    "dataset_revision",
    "action_layout",
    "representations",
    "mode",
    "quantile_range",
    "statistics",
    "content_fingerprint",
}
_STATISTIC_GROUPS = {"action", "observation.state"}
_STATISTIC_FIELDS = {"min", "max"}
_QUANTILE_FORMAT_VERSION = 3
_QUANTILE_LEVELS = (0.001, 0.999)
_QUANTILE_MODE = "quantile_min_max"
_QUANTILE_FIELDS = {
    "format_version",
    "dataset_repo_id",
    "dataset_revision",
    "robot_type",
    "action_layout",
    "normalization",
    "content_fingerprint",
}
_QUANTILE_NORMALIZATION_FIELDS = {
    "mode",
    "quantile_low",
    "quantile_high",
    "clamp",
    "stats_horizon",
    "predict_action",
    "observation_state",
}
_QUANTILE_STATISTIC_FIELDS = {"q001", "q999"}
_QUANTILE_ROLE_SPACES = {
    "predict_action": "observation_relative",
    "observation_state": "absolute",
}


@dataclass(frozen=True)
class CanonicalActionStatistics:
    """Validated relative-action and absolute-anchor Bridge statistics."""

    dataset_repo_id: str
    dataset_revision: str
    dof_dims: dict[str, int]
    dof_order: tuple[str, ...]
    mode: NormMode
    action_statistics: dict[str, tuple[float, ...]]
    observation_statistics: dict[str, tuple[float, ...]]
    content_fingerprint: str

    @property
    def action_dim(self) -> int:
        return sum(self.dof_dims.values())

    @property
    def action_representation(self) -> str:
        return _REPRESENTATIONS["action"]

    @property
    def observation_representation(self) -> str:
        return _REPRESENTATIONS["observation.state"]

    @property
    def public_action_representation(self) -> str:
        return _REPRESENTATIONS["public_action"]

    @property
    def relative_reference(self) -> str:
        return _REPRESENTATIONS["relative_reference"]

    @property
    def quantile_range(self) -> tuple[float, float]:
        return _QUANTILE_RANGE

    @property
    def identity(self) -> dict[str, Any]:
        """Return the exact path-independent identity persisted in checkpoints."""
        return {
            "format_version": _FORMAT_VERSION,
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_revision": self.dataset_revision,
            "action_layout": {
                "convention": "absolute",
                "dof_dims": dict(self.dof_dims),
                "dof_order": list(self.dof_order),
            },
            "representations": dict(_REPRESENTATIONS),
            "mode": self.mode.value,
            "quantile_range": list(_QUANTILE_RANGE),
            "content_fingerprint": self.content_fingerprint,
        }

    def normalization_stats(
        self,
        action_key: str,
        observation_key: str = "observation.state",
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            action_key: {
                name: torch.tensor(values, dtype=torch.float32)
                for name, values in self.action_statistics.items()
            },
            observation_key: {
                name: torch.tensor(values, dtype=torch.float32)
                for name, values in self.observation_statistics.items()
            },
        }

    def normalization_modes(
        self,
        action_key: str,
        observation_key: str = "observation.state",
    ) -> dict[str, NormMode]:
        return {action_key: self.mode, observation_key: self.mode}

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        expected_dataset_repo_id: str,
        expected_dataset_revision: str,
        expected_dof_dims: Mapping[str, int],
        expected_dof_order: Sequence[str],
        expected_action_dim: int,
    ) -> CanonicalActionStatistics:
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Canonical X-Tokenizer statistics artifact does not exist: {path}"
            ) from None
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Canonical X-Tokenizer statistics artifact is malformed JSON: {path}"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != _FIELDS:
            raise ValueError(
                "Canonical X-Tokenizer statistics artifact fields are malformed."
            )
        if raw["format_version"] != _FORMAT_VERSION:
            raise ValueError(
                "Canonical X-Tokenizer statistics format_version mismatch: "
                f"{raw['format_version']!r} != {_FORMAT_VERSION}."
            )
        if raw.get("representations") != _REPRESENTATIONS:
            raise ValueError(
                "Canonical X-Tokenizer statistics representation contract is malformed."
            )
        if raw.get("quantile_range") != list(_QUANTILE_RANGE):
            raise ValueError(
                "Canonical X-Tokenizer statistics quantile range must be "
                "[0.001, 0.999]."
            )

        repo_id = _required_string(raw, "dataset_repo_id")
        if repo_id != expected_dataset_repo_id:
            raise ValueError(
                "Canonical X-Tokenizer statistics dataset repository mismatch: "
                f"{repo_id!r} != {expected_dataset_repo_id!r}."
            )
        revision = _immutable_revision(raw.get("dataset_revision"))
        expected_revision = _immutable_revision(expected_dataset_revision)
        if revision != expected_revision:
            raise ValueError(
                "Canonical X-Tokenizer statistics dataset revision mismatch: "
                f"{revision!r} != {expected_revision!r}."
            )

        expected_dims = {name: int(dim) for name, dim in expected_dof_dims.items()}
        expected_order = tuple(expected_dof_order)
        _validate_layout(expected_dims, expected_order)
        layout = raw.get("action_layout")
        if not isinstance(layout, dict) or set(layout) != {
            "convention",
            "dof_dims",
            "dof_order",
        }:
            raise ValueError(
                "Canonical X-Tokenizer statistics action layout is malformed."
            )
        raw_dims = layout.get("dof_dims")
        raw_order = layout.get("dof_order")
        if (
            layout.get("convention") != "absolute"
            or not isinstance(raw_dims, dict)
            or not isinstance(raw_order, list)
        ):
            raise ValueError(
                "Canonical X-Tokenizer statistics action layout is malformed."
            )
        try:
            dims = {str(name): int(dim) for name, dim in raw_dims.items()}
        except (TypeError, ValueError):
            raise ValueError(
                "Canonical X-Tokenizer statistics action layout is malformed."
            ) from None
        order = tuple(raw_order)
        _validate_layout(dims, order)
        if dims != expected_dims or order != expected_order:
            raise ValueError("Canonical X-Tokenizer statistics action layout mismatch.")
        action_dim = sum(dims.values())
        if action_dim != int(expected_action_dim):
            raise ValueError(
                "Canonical X-Tokenizer statistics action dimension mismatch: "
                f"{action_dim} != {int(expected_action_dim)}."
            )

        try:
            mode = NormMode(raw.get("mode"))
        except ValueError:
            raise ValueError(
                "Canonical X-Tokenizer statistics normalization mode is malformed."
            ) from None
        if mode is not NormMode.MIN_MAX:
            raise ValueError("Canonical X-Tokenizer statistics mode must be min_max.")
        raw_statistics = raw.get("statistics")
        if (
            not isinstance(raw_statistics, dict)
            or set(raw_statistics) != _STATISTIC_GROUPS
        ):
            raise ValueError("Canonical X-Tokenizer statistics vectors are malformed.")
        statistics = {
            group: _validate_statistics_block(
                raw_statistics[group],
                action_dim=action_dim,
                group=group,
            )
            for group in sorted(_STATISTIC_GROUPS)
        }

        fingerprint = raw.get("content_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or _SHA256_PATTERN.fullmatch(fingerprint) is None
            or fingerprint != _content_fingerprint(raw)
        ):
            raise ValueError(
                "Canonical X-Tokenizer statistics content fingerprint mismatch."
            )
        return cls(
            dataset_repo_id=repo_id,
            dataset_revision=revision,
            dof_dims=dims,
            dof_order=order,
            mode=mode,
            action_statistics=statistics["action"],
            observation_statistics=statistics["observation.state"],
            content_fingerprint=fingerprint,
        )


@dataclass(frozen=True)
class XTokenizerQuantileStatistics:
    """Fixed-horizon quantiles for X-Tokenizer prediction and observation roles."""

    dataset_repo_id: str
    dataset_revision: str
    robot_type: str
    dof_dims: Mapping[str, int]
    dof_order: tuple[str, ...]
    stats_horizon: int
    predict_action_statistics: Mapping[str, tuple[float, ...]]
    observation_state_statistics: Mapping[str, tuple[float, ...]]
    content_fingerprint: str
    clamp: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "dof_dims", MappingProxyType(dict(self.dof_dims)))
        object.__setattr__(self, "dof_order", tuple(self.dof_order))
        object.__setattr__(
            self,
            "predict_action_statistics",
            _immutable_statistics(self.predict_action_statistics),
        )
        object.__setattr__(
            self,
            "observation_state_statistics",
            _immutable_statistics(self.observation_state_statistics),
        )

    @property
    def action_dim(self) -> int:
        return sum(self.dof_dims.values())

    @property
    def quantile_levels(self) -> tuple[float, float]:
        return _QUANTILE_LEVELS

    @property
    def predict_space(self) -> str:
        return _QUANTILE_ROLE_SPACES["predict_action"]

    @property
    def observation_space(self) -> str:
        return _QUANTILE_ROLE_SPACES["observation_state"]

    @property
    def identity(self) -> dict[str, Any]:
        """Return the path-independent contract used to bind runtime artifacts."""
        return {
            "format_version": _QUANTILE_FORMAT_VERSION,
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_revision": self.dataset_revision,
            "robot_type": self.robot_type,
            "action_layout": {
                "convention": "absolute",
                "dof_dims": dict(self.dof_dims),
                "dof_order": list(self.dof_order),
            },
            "normalization": {
                "mode": _QUANTILE_MODE,
                "quantile_low": _QUANTILE_LEVELS[0],
                "quantile_high": _QUANTILE_LEVELS[1],
                "clamp": self.clamp,
                "stats_horizon": self.stats_horizon,
                "predict_action": {
                    "space": self.predict_space,
                    **{
                        name: list(values)
                        for name, values in self.predict_action_statistics.items()
                    },
                },
                "observation_state": {
                    "space": self.observation_space,
                    **{
                        name: list(values)
                        for name, values in self.observation_state_statistics.items()
                    },
                },
            },
            "content_fingerprint": self.content_fingerprint,
        }

    def predict_normalization_stats(
        self,
        action_key: str,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return _normalization_tensors(action_key, self.predict_action_statistics)

    def observation_normalization_stats(
        self,
        observation_key: str,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return _normalization_tensors(
            observation_key,
            self.observation_state_statistics,
        )

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        expected_dataset_repo_id: str,
        expected_dataset_revision: str,
        expected_robot_type: str,
        expected_dof_dims: Mapping[str, int],
        expected_dof_order: Sequence[str],
        expected_action_dim: int,
        expected_stats_horizon: int,
    ) -> XTokenizerQuantileStatistics:
        try:
            raw = json.loads(Path(path).read_text())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"X-Tokenizer quantile statistics artifact does not exist: {path}"
            ) from None
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"X-Tokenizer quantile statistics artifact is malformed JSON: {path}"
            ) from exc
        if not isinstance(raw, dict) or set(raw) != _QUANTILE_FIELDS:
            raise ValueError("X-Tokenizer quantile statistics fields are malformed.")
        if raw["format_version"] != _QUANTILE_FORMAT_VERSION:
            raise ValueError(
                "X-Tokenizer quantile statistics format_version mismatch: "
                f"{raw['format_version']!r} != {_QUANTILE_FORMAT_VERSION}."
            )

        repo_id = _required_string(raw, "dataset_repo_id")
        if repo_id != expected_dataset_repo_id:
            raise ValueError(
                "X-Tokenizer quantile statistics dataset repository mismatch: "
                f"{repo_id!r} != {expected_dataset_repo_id!r}."
            )
        revision = _immutable_revision(raw.get("dataset_revision"))
        expected_revision = _immutable_revision(expected_dataset_revision)
        if revision != expected_revision:
            raise ValueError(
                "X-Tokenizer quantile statistics dataset revision mismatch: "
                f"{revision!r} != {expected_revision!r}."
            )
        robot_type = _required_string(raw, "robot_type")
        if robot_type != expected_robot_type:
            raise ValueError(
                "X-Tokenizer quantile statistics robot type mismatch: "
                f"{robot_type!r} != {expected_robot_type!r}."
            )

        expected_dims = _validated_dof_dims(
            expected_dof_dims,
            label="expected_dof_dims",
        )
        expected_order = tuple(expected_dof_order)
        _validate_layout(expected_dims, expected_order)
        dims, order = _quantile_action_layout(raw.get("action_layout"))
        if dims != expected_dims or order != expected_order:
            raise ValueError("X-Tokenizer quantile statistics action layout mismatch.")
        action_dim = sum(dims.values())
        expected_dim = _positive_int(
            expected_action_dim,
            label="expected_action_dim",
        )
        if action_dim != expected_dim:
            raise ValueError(
                "X-Tokenizer quantile statistics action dimension mismatch: "
                f"{action_dim} != {expected_dim}."
            )

        normalization = raw.get("normalization")
        if (
            not isinstance(normalization, dict)
            or set(normalization) != _QUANTILE_NORMALIZATION_FIELDS
        ):
            raise ValueError(
                "X-Tokenizer quantile statistics normalization is malformed."
            )
        if normalization.get("mode") != _QUANTILE_MODE:
            raise ValueError(
                "X-Tokenizer quantile statistics normalization mode must be "
                f"{_QUANTILE_MODE}."
            )
        if (
            normalization.get("quantile_low") != _QUANTILE_LEVELS[0]
            or normalization.get("quantile_high") != _QUANTILE_LEVELS[1]
        ):
            raise ValueError(
                "X-Tokenizer quantile statistics quantile levels must be "
                "0.001 and 0.999."
            )
        if normalization.get("clamp") is not True:
            raise ValueError(
                "X-Tokenizer quantile statistics normalization must clamp values."
            )
        stats_horizon = _positive_int(
            normalization.get("stats_horizon"),
            label="X-Tokenizer quantile statistics stats_horizon",
        )
        expected_horizon = _positive_int(
            expected_stats_horizon,
            label="expected_stats_horizon",
        )
        if stats_horizon != expected_horizon:
            raise ValueError(
                "X-Tokenizer quantile statistics stats_horizon mismatch: "
                f"{stats_horizon} != {expected_horizon}."
            )
        statistics = {
            role: _validate_quantile_artifact_block(
                normalization[role],
                action_dim=action_dim,
                role=role,
            )
            for role in _QUANTILE_ROLE_SPACES
        }

        fingerprint = raw.get("content_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or _SHA256_PATTERN.fullmatch(fingerprint) is None
            or fingerprint != _content_fingerprint(raw)
        ):
            raise ValueError(
                "X-Tokenizer quantile statistics content fingerprint mismatch."
            )
        return cls(
            dataset_repo_id=repo_id,
            dataset_revision=revision,
            robot_type=robot_type,
            dof_dims=dims,
            dof_order=order,
            stats_horizon=stats_horizon,
            predict_action_statistics=statistics["predict_action"],
            observation_state_statistics=statistics["observation_state"],
            content_fingerprint=fingerprint,
        )


def validate_xtokenizer_quantile_identity(
    value: Mapping[str, Any],
    *,
    expected_action_dim: int,
    expected_stats_horizon: int,
) -> dict[str, Any]:
    """Validate and detach a path-independent quantile statistics identity."""
    if not isinstance(value, dict) or set(value) != _QUANTILE_FIELDS:
        raise ValueError("Canonical statistics identity fields are malformed.")
    if value["format_version"] != _QUANTILE_FORMAT_VERSION:
        raise ValueError("Canonical statistics identity format_version is unsupported.")
    _required_string(value, "dataset_repo_id")
    _immutable_revision(value.get("dataset_revision"))
    _required_string(value, "robot_type")
    dims, _ = _quantile_action_layout(value.get("action_layout"))
    action_dim = sum(dims.values())
    expected_dim = _positive_int(
        expected_action_dim,
        label="expected_action_dim",
    )
    if action_dim != expected_dim:
        raise ValueError(
            "Canonical statistics identity action dimension mismatch: "
            f"{action_dim} != {expected_dim}."
        )

    normalization = value.get("normalization")
    if (
        not isinstance(normalization, dict)
        or set(normalization) != _QUANTILE_NORMALIZATION_FIELDS
    ):
        raise ValueError("Canonical statistics identity normalization is malformed.")
    if normalization.get("mode") != _QUANTILE_MODE:
        raise ValueError(
            "Canonical statistics identity normalization mode is malformed."
        )
    if (
        normalization.get("quantile_low") != _QUANTILE_LEVELS[0]
        or normalization.get("quantile_high") != _QUANTILE_LEVELS[1]
        or normalization.get("clamp") is not True
    ):
        raise ValueError(
            "Canonical statistics identity quantile contract is malformed."
        )
    stats_horizon = _positive_int(
        normalization.get("stats_horizon"),
        label="canonical statistics identity stats_horizon",
    )
    expected_horizon = _positive_int(
        expected_stats_horizon,
        label="expected_stats_horizon",
    )
    if stats_horizon != expected_horizon:
        raise ValueError(
            "Canonical statistics identity stats_horizon mismatch: "
            f"{stats_horizon} != {expected_horizon}."
        )
    for role in _QUANTILE_ROLE_SPACES:
        _validate_quantile_artifact_block(
            normalization[role],
            action_dim=action_dim,
            role=role,
        )

    fingerprint = value.get("content_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or _SHA256_PATTERN.fullmatch(fingerprint) is None
        or fingerprint != _content_fingerprint(value)
    ):
        raise ValueError("Canonical statistics identity content fingerprint mismatch.")
    return json.loads(json.dumps(value, allow_nan=False))


def write_canonical_action_statistics(
    path: Path,
    *,
    dataset_repo_id: str,
    dataset_revision: str,
    dof_dims: Mapping[str, int],
    dof_order: Sequence[str],
    mode: NormMode,
    action_statistics: Mapping[str, Sequence[float]],
    observation_statistics: Mapping[str, Sequence[float]],
) -> Path:
    """Write one deterministic artifact after applying the same strict checks."""
    if not isinstance(dataset_repo_id, str) or not dataset_repo_id.strip():
        raise ValueError("dataset_repo_id must be a non-empty string.")
    revision = _immutable_revision(dataset_revision)
    dims = _validated_dof_dims(dof_dims, label="dof_dims")
    order = tuple(dof_order)
    _validate_layout(dims, order)
    if NormMode(mode) is not NormMode.MIN_MAX:
        raise ValueError("Canonical X-Tokenizer statistics mode must be min_max.")
    raw = {
        "format_version": _FORMAT_VERSION,
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": revision,
        "action_layout": {
            "convention": "absolute",
            "dof_dims": dims,
            "dof_order": list(order),
        },
        "representations": dict(_REPRESENTATIONS),
        "mode": NormMode(mode).value,
        "quantile_range": list(_QUANTILE_RANGE),
        "statistics": {
            "action": {
                name: [float(value) for value in action_statistics[name]]
                for name in sorted(_STATISTIC_FIELDS)
            },
            "observation.state": {
                name: [float(value) for value in observation_statistics[name]]
                for name in sorted(_STATISTIC_FIELDS)
            },
        },
    }
    raw["content_fingerprint"] = _content_fingerprint(raw)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(raw, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                existing_payload = destination.read_bytes()
            except FileNotFoundError:
                os.link(temporary, destination)
            else:
                if existing_payload != payload:
                    raise ValueError(
                        "Canonical X-Tokenizer statistics artifact already exists "
                        f"with different content: {destination}"
                    ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    CanonicalActionStatistics.from_path(
        destination,
        expected_dataset_repo_id=dataset_repo_id,
        expected_dataset_revision=revision,
        expected_dof_dims=dims,
        expected_dof_order=order,
        expected_action_dim=sum(dims.values()),
    )
    return destination


def write_xtokenizer_quantile_statistics(
    path: Path,
    *,
    dataset_repo_id: str,
    dataset_revision: str,
    robot_type: str,
    dof_dims: Mapping[str, int],
    dof_order: Sequence[str],
    stats_horizon: int,
    predict_action_statistics: Mapping[str, Sequence[float]],
    observation_state_statistics: Mapping[str, Sequence[float]],
) -> Path:
    """Atomically publish one fixed-horizon dual-role quantile artifact."""
    if not isinstance(dataset_repo_id, str) or not dataset_repo_id.strip():
        raise ValueError("dataset_repo_id must be a non-empty string.")
    revision = _immutable_revision(dataset_revision)
    if not isinstance(robot_type, str) or not robot_type.strip():
        raise ValueError("robot_type must be a non-empty string.")
    dims = _validated_dof_dims(dof_dims, label="dof_dims")
    order = tuple(dof_order)
    _validate_layout(dims, order)
    action_dim = sum(dims.values())
    horizon = _positive_int(stats_horizon, label="stats_horizon")
    statistics = {
        "predict_action": _validate_quantile_values(
            predict_action_statistics,
            action_dim=action_dim,
            role="predict_action",
        ),
        "observation_state": _validate_quantile_values(
            observation_state_statistics,
            action_dim=action_dim,
            role="observation_state",
        ),
    }
    raw = {
        "format_version": _QUANTILE_FORMAT_VERSION,
        "dataset_repo_id": dataset_repo_id,
        "dataset_revision": revision,
        "robot_type": robot_type,
        "action_layout": {
            "convention": "absolute",
            "dof_dims": dims,
            "dof_order": list(order),
        },
        "normalization": {
            "mode": _QUANTILE_MODE,
            "quantile_low": _QUANTILE_LEVELS[0],
            "quantile_high": _QUANTILE_LEVELS[1],
            "clamp": True,
            "stats_horizon": horizon,
            **{
                role: {
                    "space": _QUANTILE_ROLE_SPACES[role],
                    **{name: list(values) for name, values in role_statistics.items()},
                }
                for role, role_statistics in statistics.items()
            },
        },
    }
    raw["content_fingerprint"] = _content_fingerprint(raw)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(raw, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                existing_payload = destination.read_bytes()
            except FileNotFoundError:
                os.link(temporary, destination)
            else:
                if existing_payload != payload:
                    raise ValueError(
                        "X-Tokenizer quantile statistics artifact already exists "
                        f"with different content: {destination}"
                    ) from None
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    XTokenizerQuantileStatistics.from_path(
        destination,
        expected_dataset_repo_id=dataset_repo_id,
        expected_dataset_revision=revision,
        expected_robot_type=robot_type,
        expected_dof_dims=dims,
        expected_dof_order=order,
        expected_action_dim=action_dim,
        expected_stats_horizon=horizon,
    )
    return destination


def _validate_statistics_block(
    value: Any,
    *,
    action_dim: int,
    group: str,
) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, dict) or set(value) != _STATISTIC_FIELDS:
        raise ValueError("Canonical X-Tokenizer statistics vectors are malformed.")
    statistics: dict[str, tuple[float, ...]] = {}
    for name in sorted(_STATISTIC_FIELDS):
        values = value[name]
        if not isinstance(values, list) or len(values) != action_dim:
            raise ValueError(
                "Canonical X-Tokenizer statistics vector dimension mismatch "
                f"for {group}.{name}: expected {action_dim}."
            )
        try:
            vector = tuple(float(item) for item in values)
        except (TypeError, ValueError):
            raise ValueError(
                f"Canonical X-Tokenizer statistics {group}.{name} must be numeric."
            ) from None
        if not all(math.isfinite(item) for item in vector):
            raise ValueError(
                f"Canonical X-Tokenizer statistics {group}.{name} must be finite."
            )
        statistics[name] = vector
    if any(
        minimum >= maximum
        for minimum, maximum in zip(
            statistics["min"],
            statistics["max"],
            strict=True,
        )
    ):
        raise ValueError(
            "Canonical X-Tokenizer statistics require min < max in every "
            f"{group} channel."
        )
    return statistics


def _quantile_action_layout(
    value: Any,
) -> tuple[dict[str, int], tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != {
        "convention",
        "dof_dims",
        "dof_order",
    }:
        raise ValueError("X-Tokenizer quantile statistics action layout is malformed.")
    raw_dims = value.get("dof_dims")
    raw_order = value.get("dof_order")
    if (
        value.get("convention") != "absolute"
        or not isinstance(raw_dims, dict)
        or not isinstance(raw_order, list)
    ):
        raise ValueError("X-Tokenizer quantile statistics action layout is malformed.")
    dims = _validated_dof_dims(
        raw_dims,
        label="X-Tokenizer quantile statistics action_layout.dof_dims",
    )
    order = tuple(raw_order)
    _validate_layout(dims, order)
    return dims, order


def _validate_quantile_artifact_block(
    value: Any,
    *,
    action_dim: int,
    role: str,
) -> dict[str, tuple[float, ...]]:
    expected_fields = {"space", *_QUANTILE_STATISTIC_FIELDS}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"X-Tokenizer quantile statistics {role} block is malformed.")
    expected_space = _QUANTILE_ROLE_SPACES[role]
    if value.get("space") != expected_space:
        raise ValueError(
            f"X-Tokenizer quantile statistics {role} space must be {expected_space!r}."
        )
    return _validate_quantile_values(
        {name: value[name] for name in _QUANTILE_STATISTIC_FIELDS},
        action_dim=action_dim,
        role=role,
        require_json_arrays=True,
    )


def _validate_quantile_values(
    value: Any,
    *,
    action_dim: int,
    role: str,
    require_json_arrays: bool = False,
) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or set(value) != _QUANTILE_STATISTIC_FIELDS:
        raise ValueError(
            f"X-Tokenizer quantile statistics {role} vectors are malformed."
        )
    statistics: dict[str, tuple[float, ...]] = {}
    for name in sorted(_QUANTILE_STATISTIC_FIELDS):
        values = value[name]
        if require_json_arrays:
            if not isinstance(values, list):
                raise ValueError(
                    "X-Tokenizer quantile statistics "
                    f"{role}.{name} must be a JSON array."
                )
        elif not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(
                "X-Tokenizer quantile statistics "
                f"{role}.{name} must be a numeric sequence."
            )
        if len(values) != action_dim:
            raise ValueError(
                "X-Tokenizer quantile statistics vector dimension mismatch "
                f"for {role}.{name}: expected {action_dim}."
            )
        if any(isinstance(item, bool) or not isinstance(item, Real) for item in values):
            raise ValueError(
                "X-Tokenizer quantile statistics "
                f"{role}.{name} must contain only real numbers."
            )
        vector = tuple(float(item) for item in values)
        if not all(math.isfinite(item) for item in vector):
            raise ValueError(
                f"X-Tokenizer quantile statistics {role}.{name} must be finite."
            )
        statistics[name] = vector
    if any(
        low >= high
        for low, high in zip(
            statistics["q001"],
            statistics["q999"],
            strict=True,
        )
    ):
        raise ValueError(
            "X-Tokenizer quantile statistics require q001 < q999 in every "
            f"{role} channel."
        )
    return statistics


def _normalization_tensors(
    key: str,
    statistics: Mapping[str, Sequence[float]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        key: {
            "min": torch.tensor(statistics["q001"], dtype=torch.float32),
            "max": torch.tensor(statistics["q999"], dtype=torch.float32),
        }
    }


def _positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def _validated_dof_dims(
    value: Mapping[str, int],
    *,
    label: str,
) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping.")
    dimensions: dict[str, int] = {}
    for name, dimension in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} keys must be non-empty strings.")
        dimensions[name] = _positive_int(
            dimension,
            label=f"{label}[{name!r}]",
        )
    return dimensions


def _immutable_statistics(
    value: Mapping[str, Sequence[float]],
) -> Mapping[str, tuple[float, ...]]:
    return MappingProxyType(
        {name: tuple(statistics) for name, statistics in value.items()}
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_layout(dof_dims: Mapping[str, int], dof_order: Sequence[str]) -> None:
    order = tuple(dof_order)
    if (
        not order
        or not all(isinstance(name, str) and name for name in order)
        or len(order) != len(set(order))
        or set(order) != set(dof_dims)
    ):
        raise ValueError(
            "Canonical X-Tokenizer statistics action layout is invalid: "
            "dof_order must list every dof_dims key exactly once."
        )

    canonical_dims: dict[str, int] = {}
    for raw_name in order:
        dimension = _positive_int(
            dof_dims[raw_name],
            label=f"dof_dims[{raw_name!r}]",
        )
        canonical_name = raw_name.replace("_relative", "")
        if canonical_name in canonical_dims:
            raise ValueError(
                "Canonical X-Tokenizer statistics action layout is invalid: "
                f"duplicate canonical DoF {canonical_name!r}."
            )
        canonical_dims[canonical_name] = dimension

    required_dims = {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
        "follow_left_gripper": 1,
    }
    for name, expected_dimension in required_dims.items():
        if canonical_dims.get(name) != expected_dimension:
            raise ValueError(
                "Canonical X-Tokenizer statistics action layout is invalid: "
                f"dof_dims must contain {name!r} with dim {expected_dimension}."
            )


def _immutable_revision(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise ValueError(
            "Canonical X-Tokenizer statistics dataset revision must be an "
            "immutable 40-character hexadecimal commit SHA."
        )
    return value.lower()


def _required_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Canonical X-Tokenizer statistics {key} must be a non-empty string."
        )
    return value


def _content_fingerprint(raw: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in raw.items() if key != "content_fingerprint"}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
