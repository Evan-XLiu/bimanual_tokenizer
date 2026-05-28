"""Small helpers for codec dataclass configs."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from typing import Any, TypeVar, cast

T = TypeVar("T")


def build_dataclass_config(
    config_cls: type[T],
    cfg: dict[str, Any],
    *,
    tuple_int_fields: tuple[str, ...] = (),
) -> T:
    """Build a dataclass config while rejecting unknown keys."""
    if not is_dataclass(config_cls):
        raise TypeError(f"{config_cls!r} is not a dataclass config.")
    known = {f.name for f in fields(cast(Any, config_cls))}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(
            "Unknown codec config key(s) for "
            f"{config_cls.__name__}: {', '.join(unknown)}."
        )
    filtered = {k: v for k, v in cfg.items() if k in known}
    for key in tuple_int_fields:
        value = filtered.get(key)
        if isinstance(value, list):
            filtered[key] = tuple(int(v) for v in value)
    return cast(T, config_cls(**filtered))


def dataclass_config_to_dict(config: object) -> dict[str, Any]:
    """Serialize a codec dataclass config to a plain dict."""
    if not is_dataclass(config):
        raise TypeError(f"{type(config)!r} is not a dataclass config.")
    return cast(dict[str, Any], asdict(cast(Any, config)))
