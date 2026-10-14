"""Copy a completed recent checkpoint before the recent pool evicts it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("--timeout-hours", type=float, default=12)
    args = parser.parse_args()

    root = args.checkpoint_root.resolve()
    source = root / "recent" / f"step_{args.step}"
    manifest = root / "recent" / "recent_manifest.json"
    target = root / "protected" / source.name
    staging = root / "protected" / f".{source.name}.copying"
    if target.exists() or staging.exists():
        raise SystemExit(f"Refusing to overwrite {target} or {staging}")

    deadline = time.monotonic() + args.timeout_hours * 3600
    while time.monotonic() < deadline:
        try:
            entries = json.loads(manifest.read_text())
        except (OSError, ValueError):
            entries = []
        if source.is_dir() and any(
            entry.get("step") == args.step and entry.get("file") == source.name
            for entry in entries
        ):
            break
        time.sleep(5)
    else:
        raise SystemExit(f"Timed out waiting for completed {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    files = [path.relative_to(source) for path in source.rglob("*") if path.is_file()]
    if not files or not (staging / "pretrained_model" / "model.safetensors").is_file():
        raise SystemExit(f"Incomplete checkpoint copy at {staging}")
    for relative in files:
        if file_hash(source / relative) != file_hash(staging / relative):
            raise SystemExit(f"SHA-256 mismatch for {relative}")
    staging.rename(target)
    print(f"Archived and verified {len(files)} files: {target}", flush=True)


if __name__ == "__main__":
    main()
