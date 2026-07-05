# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for config object serialization and comparison."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, cast


def config_to_dict(config_obj: Any) -> dict[str, Any] | None:
    """Serialize common config object shapes into a plain dict."""
    if config_obj is None:
        return None

    to_dict = getattr(config_obj, "to_dict", None)
    if callable(to_dict):
        data = to_dict()
        if isinstance(data, dict):
            return cast(dict[str, Any], dict(data))

    if not isinstance(config_obj, type) and is_dataclass(config_obj):
        return cast(dict[str, Any], asdict(cast(Any, config_obj)))

    if isinstance(config_obj, dict):
        return dict(config_obj)

    return None


def require_config_dict(config_obj: Any) -> dict[str, Any]:
    """Serialize a config object, failing when no supported representation exists."""
    payload = config_to_dict(config_obj)
    if payload is None:
        raise TypeError(
            f"Unsupported config object type for serialization: {type(config_obj)!r}"
        )
    return payload


def normalize_config_value(value: Any) -> Any:
    """Normalize config payload values for stable semantic equality checks."""
    if isinstance(value, dict):
        return {str(key): normalize_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(normalize_config_value(item) for item in value)
    return value


def config_values_equal(left: Any, right: Any) -> bool:
    """Compare config values while ignoring mapping-key/list-vs-tuple noise."""
    return normalize_config_value(left) == normalize_config_value(right)
