# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import subprocess
import tarfile
from pathlib import Path

import pytest

from praxis.scripts.serve import _verify_runtime_overlay


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True)


def _dirty_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    source = repo / "src" / "praxis" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("VERSION = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "src/praxis/runtime.py")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    source.write_text("VERSION = 2\n")
    return repo, source


def _write_overlay(path: Path, repo: Path, *members: str) -> str:
    with tarfile.open(path, mode="w") as archive:
        for member in members:
            archive.add(repo / member, arcname=member, recursive=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_overlay_verifies_membership_hash_and_current_bytes(
    tmp_path: Path,
) -> None:
    repo, _source = _dirty_repo(tmp_path)
    overlay = tmp_path / "overlay.tar"
    digest = _write_overlay(overlay, repo, "src/praxis/runtime.py")

    identity = _verify_runtime_overlay(
        repo_root=repo, overlay_path=overlay, expected_sha256=digest
    )

    assert identity == {
        "path": str(overlay),
        "sha256": digest,
        "file_count": 1,
        "verified": True,
    }


def test_runtime_overlay_rejects_stale_member_bytes(tmp_path: Path) -> None:
    repo, source = _dirty_repo(tmp_path)
    overlay = tmp_path / "overlay.tar"
    digest = _write_overlay(overlay, repo, "src/praxis/runtime.py")
    source.write_text("VERSION = 3\n")

    with pytest.raises(RuntimeError, match="bytes do not match"):
        _verify_runtime_overlay(
            repo_root=repo, overlay_path=overlay, expected_sha256=digest
        )


def test_runtime_overlay_rejects_incomplete_membership(tmp_path: Path) -> None:
    repo, _source = _dirty_repo(tmp_path)
    extra = repo / "scripts" / "tool.py"
    extra.parent.mkdir()
    extra.write_text("ENABLED = True\n")
    overlay = tmp_path / "overlay.tar"
    digest = _write_overlay(overlay, repo, "src/praxis/runtime.py")

    with pytest.raises(RuntimeError, match="membership does not match"):
        _verify_runtime_overlay(
            repo_root=repo, overlay_path=overlay, expected_sha256=digest
        )
