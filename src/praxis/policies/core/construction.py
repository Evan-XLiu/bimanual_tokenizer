# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for config-backed policy construction."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

from praxis.core.types import PolicyIO


def coerce_policy_config(
    config: Any,
    *,
    config_cls: type | None,
    owner_name: str,
) -> Any:
    """Resolve a config dict through a policy's declared CONFIG_CLASS."""
    if not isinstance(config, dict):
        return config
    if config_cls is None:
        raise RuntimeError(
            f"{owner_name} must set CONFIG_CLASS to accept config dicts."
        )
    from_dict = getattr(config_cls, "from_dict", None)
    if callable(from_dict):
        return from_dict(config)
    return config_cls(**config)


def build_config_policy(
    policy_cls,
    *,
    io: PolicyIO,
    config_cls,
    resolved_overrides: dict,
):
    """Build a config-backed policy from resolved config kwargs.

    This is intentionally small and generic: callers remain responsible for
    deriving ``resolved_overrides`` from ``PolicyIO``.  The shared helper only
    owns the common ``config dataclass -> policy(config=...) -> set_io(...)``
    construction pattern.
    """
    if config_cls is None:
        raise RuntimeError(f"{policy_cls.__name__} must set CONFIG_CLASS.")
    from_dict = getattr(config_cls, "from_dict", None)
    if callable(from_dict):
        config = from_dict(resolved_overrides)
    else:
        config = config_cls(**resolved_overrides)
    policy = policy_cls(config=asdict(cast(Any, config)))
    policy.set_io(io)
    return policy
