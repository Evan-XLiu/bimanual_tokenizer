"""Shared path helpers for tests."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root containing ``pyproject.toml``."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Failed to locate repository root from tests/_paths.py")
