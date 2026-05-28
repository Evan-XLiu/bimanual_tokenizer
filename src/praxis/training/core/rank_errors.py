"""Rank-aware error propagation helpers for distributed trainer services."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any


def rank0_error_message(context: str, exc: BaseException) -> str:
    """Format an exception that is expected to originate on rank 0."""
    return f"{context} failed on rank 0: {type(exc).__name__}: {exc}"


def rank_error_message(runtime: Any, context: str, exc: BaseException) -> str:
    """Format a rank-local exception with the runtime's global rank."""
    rank = int(getattr(runtime, "global_rank", 0))
    return f"{context} failed on rank {rank}: {type(exc).__name__}: {exc}"


def collect_rank_errors(runtime: Any, error: str | None) -> list[str]:
    """Gather optional rank-local error strings from every rank."""
    gather_object = getattr(runtime, "gather_object", None)
    raw_gathered = gather_object(error) if callable(gather_object) else [error]
    gathered = (
        raw_gathered if isinstance(raw_gathered, list | tuple) else [raw_gathered]
    )
    return [str(item) for item in gathered if item is not None]


def raise_collected_rank_errors(errors: list[str]) -> None:
    """Raise one RuntimeError containing all gathered rank-local errors."""
    if errors:
        raise RuntimeError("; ".join(errors))


def raise_if_rank0_error(runtime: Any, error: str | None) -> None:
    """Broadcast a rank-0 error string and raise it on every rank."""
    error = runtime.broadcast_object(error, src=0)
    if error is not None:
        raise RuntimeError(str(error))


@contextlib.contextmanager
def rank0_error_scope(runtime: Any, context: str) -> Iterator[None]:
    """Turn rank-0 exceptions into a broadcast error while peers participate."""
    error: str | None = None
    try:
        yield
    except BaseException as exc:
        if getattr(runtime, "is_main_process", False):
            error = rank0_error_message(context, exc)
        else:
            raise
    raise_if_rank0_error(runtime, error)
