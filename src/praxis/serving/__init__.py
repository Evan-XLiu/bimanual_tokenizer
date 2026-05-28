from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from praxis.serving.server import PolicyServer

__all__ = ["PolicyServer"]


def __getattr__(name: str) -> Any:
    if name == "PolicyServer":
        module = import_module("praxis.serving.server")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
