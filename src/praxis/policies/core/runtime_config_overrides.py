# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Pretrained policy config override helpers."""

from __future__ import annotations

from typing import Any

from praxis.core.config import config_to_dict, config_values_equal
from praxis.policies.core import BasePolicy

PRETRAINED_RUNTIME_OVERRIDE_KEYS_ATTR = "PRETRAINED_RUNTIME_OVERRIDE_KEYS"


def apply_pretrained_config_overrides(
    policy: BasePolicy,
    cfg_dict: dict[str, Any],
    *,
    explicit_override_keys: set[str] | None,
) -> frozenset[str]:
    """Patch a loaded policy config with explicit runtime-safe overrides."""
    config_obj = getattr(policy, "config", None)
    if config_obj is None:
        return frozenset()

    overrides: dict[str, Any] = {}
    for key, value in cfg_dict.items():
        if key in ("name", "pretrained_path"):
            continue
        if explicit_override_keys is not None and key not in explicit_override_keys:
            continue
        if not hasattr(config_obj, key):
            if explicit_override_keys is not None:
                raise ValueError(
                    f"Unknown policy config key for pretrained override: policy.{key}."
                )
            continue
        checkpoint_value = getattr(config_obj, key)
        if config_values_equal(checkpoint_value, value):
            continue
        _validate_pretrained_config_override(config_obj=config_obj, key=key)
        overrides[key] = value

    if not overrides:
        return frozenset()

    validated = _validate_pretrained_config_overrides(
        config_obj=config_obj,
        overrides=overrides,
    )
    for key, value in validated.items():
        setattr(config_obj, key, value)
    return frozenset(validated)


def _pretrained_runtime_override_keys(config_obj: Any) -> frozenset[str] | None:
    keys = getattr(
        type(config_obj),
        PRETRAINED_RUNTIME_OVERRIDE_KEYS_ATTR,
        getattr(config_obj, PRETRAINED_RUNTIME_OVERRIDE_KEYS_ATTR, None),
    )
    if keys is None:
        return None
    return frozenset(str(key) for key in keys)


def _validate_pretrained_config_override(
    *,
    config_obj: Any,
    key: str,
) -> None:
    runtime_keys = _pretrained_runtime_override_keys(config_obj)
    if runtime_keys is None or key in runtime_keys:
        return
    allowed = ", ".join(sorted(runtime_keys)) if runtime_keys else "none"
    raise ValueError(
        f"Cannot override policy.{key} when loading policy.pretrained_path. "
        f"{type(config_obj).__name__} treats that field as checkpoint-owned. "
        f"Allowed runtime override keys for this config: {allowed}."
    )


def _validate_pretrained_config_overrides(
    *,
    config_obj: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    payload = config_to_dict(config_obj)
    if payload is None:
        return dict(overrides)
    payload.update(overrides)

    parser = getattr(type(config_obj), "from_dict", None)
    if callable(parser):
        validated_config = parser(payload)
    else:
        try:
            validated_config = type(config_obj)(**payload)
        except TypeError:
            return dict(overrides)

    return {key: getattr(validated_config, key) for key in overrides}
