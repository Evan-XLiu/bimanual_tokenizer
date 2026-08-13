# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""gRPC serving entry point."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import hydra
from omegaconf import DictConfig, OmegaConf

from praxis.core.devices import require_available_device
from praxis.serving.model_identity import build_policy_serving_identity

CONFIG_PATH = str(Path(__file__).parents[1] / "configs")


_OVERLAY_ROOTS = ("src", "scripts", "tests")
_INSTALLED_SOURCE_COMMIT = (
    Path(__file__).resolve().parents[1] / "serving" / "SOURCE_COMMIT"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_paths(repo_root: Path, *args: str) -> set[str]:
    command_args = list(args)
    try:
        separator_index = command_args.index("--")
    except ValueError:
        command_args.append("-z")
    else:
        command_args.insert(separator_index, "-z")
    result = subprocess.run(
        ["git", "-C", str(repo_root), *command_args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot inspect Praxis source overlay with git {' '.join(command_args)}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return {item.decode() for item in result.stdout.split(b"\0") if item}


def _dirty_overlay_paths(repo_root: Path) -> set[str]:
    tracked = _git_paths(
        repo_root, "diff", "--name-only", "HEAD", "--", *_OVERLAY_ROOTS
    )
    untracked = _git_paths(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *_OVERLAY_ROOTS,
    )
    paths = tracked | untracked
    missing = sorted(path for path in paths if not (repo_root / path).is_file())
    if missing:
        raise RuntimeError(
            "Praxis runtime overlay cannot represent deleted or non-file paths: "
            f"{missing}."
        )
    return paths


def _verify_runtime_overlay(
    *, repo_root: Path, overlay_path: Path, expected_sha256: str
) -> dict[str, object]:
    if not overlay_path.is_file():
        raise FileNotFoundError(f"Praxis runtime overlay not found: {overlay_path}.")
    actual_sha256 = _sha256(overlay_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Praxis runtime overlay SHA256 mismatch: "
            f"actual={actual_sha256}, expected={expected_sha256}."
        )
    expected_paths = _dirty_overlay_paths(repo_root)
    archive_paths: set[str] = set()
    with tarfile.open(overlay_path, mode="r:*") as archive:
        for member in archive.getmembers():
            pure_path = PurePosixPath(member.name)
            if (
                pure_path.is_absolute()
                or ".." in pure_path.parts
                or not pure_path.parts
                or pure_path.parts[0] not in _OVERLAY_ROOTS
                or not member.isfile()
            ):
                raise RuntimeError(
                    f"Unsafe or unsupported Praxis runtime overlay member: {member.name!r}."
                )
            relative_path = pure_path.as_posix()
            if relative_path in archive_paths:
                raise RuntimeError(
                    f"Duplicate Praxis runtime overlay member: {relative_path}."
                )
            archive_paths.add(relative_path)
            archived = archive.extractfile(member)
            if archived is None or archived.read() != (repo_root / relative_path).read_bytes():
                raise RuntimeError(
                    "Praxis runtime overlay bytes do not match the current checkout: "
                    f"{relative_path}."
                )
    if archive_paths != expected_paths:
        raise RuntimeError(
            "Praxis runtime overlay membership does not match dirty src/scripts/tests "
            f"paths: missing={sorted(expected_paths - archive_paths)}, "
            f"extra={sorted(archive_paths - expected_paths)}."
        )
    return {
        "path": str(overlay_path),
        "sha256": actual_sha256,
        "file_count": len(archive_paths),
        "verified": True,
    }


def _source_metadata() -> dict[str, object]:
    if _INSTALLED_SOURCE_COMMIT.is_file():
        commit = _INSTALLED_SOURCE_COMMIT.read_text(encoding="utf-8").strip()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise RuntimeError(
                f"Invalid installed Praxis source commit: {_INSTALLED_SOURCE_COMMIT}."
            )
        return {
            "repo_root": None,
            "git_commit": commit,
            "git_dirty": False,
            "source_snapshot_sha256": None,
            "runtime_overlay_path": None,
            "runtime_overlay_sha256": None,
            "runtime_overlay_file_count": None,
            "runtime_overlay_verified": False,
        }

    repo_root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=all")
    git_dirty = bool(status) if status is not None else None
    overlay_path_value = os.environ.get("PRAXIS_RUNTIME_OVERLAY_PATH")
    overlay_sha256 = os.environ.get("PRAXIS_RUNTIME_OVERLAY_SHA256")
    if (overlay_path_value is None) != (overlay_sha256 is None):
        raise RuntimeError(
            "PRAXIS_RUNTIME_OVERLAY_PATH and PRAXIS_RUNTIME_OVERLAY_SHA256 must be "
            "provided together."
        )
    runtime_overlay: dict[str, object] | None = None
    if overlay_path_value is not None:
        assert overlay_sha256 is not None
        runtime_overlay = _verify_runtime_overlay(
            repo_root=repo_root,
            overlay_path=Path(overlay_path_value).expanduser().resolve(),
            expected_sha256=overlay_sha256,
        )
    if git_dirty and runtime_overlay is None:
        raise RuntimeError(
            "Dirty Praxis serving source requires a verified PRAXIS_RUNTIME_OVERLAY_PATH "
            "and PRAXIS_RUNTIME_OVERLAY_SHA256."
        )
    return {
        "repo_root": str(repo_root),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": git_dirty,
        "source_snapshot_sha256": os.environ.get("PRAXIS_SOURCE_SNAPSHOT_SHA256"),
        "runtime_overlay_path": (
            None if runtime_overlay is None else runtime_overlay["path"]
        ),
        "runtime_overlay_sha256": (
            None if runtime_overlay is None else runtime_overlay["sha256"]
        ),
        "runtime_overlay_file_count": (
            None if runtime_overlay is None else runtime_overlay["file_count"]
        ),
        "runtime_overlay_verified": (
            False if runtime_overlay is None else runtime_overlay["verified"]
        ),
    }


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="serve")
def main(cfg: DictConfig) -> None:
    """Start the gRPC policy server.

    Policy is loaded self-contained from checkpoint (weights + PolicyIO + norm stats).
    No dataset download is required.
    """
    print(OmegaConf.to_yaml(cfg))
    source_metadata_before = _source_metadata()

    from praxis.policies.factory import make_policy
    from praxis.serving.server import PolicyServer

    device = require_available_device(cfg.device, purpose="Serving")

    checkpoint_path = OmegaConf.select(cfg, "policy.pretrained_path") or None
    serving_identity_before = (
        build_policy_serving_identity(checkpoint_path)
        if checkpoint_path is not None
        else None
    )

    # Build policy via factory — self-contained from checkpoint.
    policy = make_policy(cfg.policy)
    serving_identity_after = (
        build_policy_serving_identity(checkpoint_path)
        if checkpoint_path is not None
        else None
    )
    if serving_identity_before != serving_identity_after:
        raise RuntimeError(
            "Policy or action-tokenizer artifacts changed while the policy was loading."
        )
    source_metadata_after = _source_metadata()
    if source_metadata_before != source_metadata_after:
        raise RuntimeError("Praxis serving source changed while the policy was loading.")

    # Start server
    server = PolicyServer(
        policy=policy,
        device=str(device),
        host=cfg.host,
        port=cfg.port,
        serving_identity=serving_identity_after,
        source_metadata=source_metadata_after,
    )
    server.serve(block=True)


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
