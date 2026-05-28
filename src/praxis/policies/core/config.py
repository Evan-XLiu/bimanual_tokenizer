"""Shared config validation helpers for policy dataclasses."""

from __future__ import annotations


def positive_int(value: int, name: str) -> int:
    resolved = int(value)
    if isinstance(value, bool) or resolved < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return resolved


def non_negative_int(value: int, name: str) -> int:
    resolved = int(value)
    if isinstance(value, bool) or resolved < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return resolved


def positive_float(value: float, name: str) -> float:
    resolved = float(value)
    if resolved <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return resolved


def non_negative_float(value: float, name: str) -> float:
    resolved = float(value)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return resolved


def tuple_of_str(value: tuple[str, ...] | list[str], name: str) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    raise TypeError(f"{name} must be a tuple/list of strings, got {type(value)!r}.")
