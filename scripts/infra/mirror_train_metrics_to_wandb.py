#!/usr/bin/env python3
"""Copy a live training metrics file to a second W&B account.

This does not modify the source training process or its W&B run. Re-running the
command resumes the destination run and skips steps already logged there.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--expected-username", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--final-step", type=int, default=30000)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    return parser.parse_args()


def _read_key(path: Path) -> str:
    key_stat = path.stat()
    if key_stat.st_uid != os.getuid():
        raise PermissionError(f"Key file must be owned by the current user: {path}")
    if stat.S_IMODE(key_stat.st_mode) & 0o077:
        raise PermissionError(f"Key file must not be accessible by others: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"Key file is empty: {path}")
    return key


def _new_entries(path: Path, after_step: int) -> list[tuple[int, dict]]:
    entries: dict[int, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # The writer may still be appending this line.
            if not isinstance(record, dict):
                continue
            step = record.get("step")
            if isinstance(step, int) and step > after_step:
                entries[step] = {key: value for key, value in record.items() if key != "step"}
    return sorted(entries.items())


def main() -> None:
    args = _arguments()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    key = _read_key(args.key_file.expanduser())
    os.environ["WANDB_API_KEY"] = key
    import wandb

    api = wandb.Api(api_key=key, timeout=30)
    viewer = api.viewer
    username = (
        viewer.get("username")
        if isinstance(viewer, dict)
        else getattr(viewer, "username", None)
    )
    if username != args.expected_username:
        raise PermissionError(
            f"W&B key belongs to {username!r}, expected {args.expected_username!r}; no run was created"
        )

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        resume="allow",
        mode="online",
        config={"source_run": args.source_run, "metrics_source": str(args.metrics)},
    )
    try:
        last_step = max(-1, run.step - 1)
        print(f"Mirroring to {run.url}; resuming after step {last_step}", flush=True)
        while True:
            for step, metrics in _new_entries(args.metrics, last_step):
                run.log(metrics, step=step)
                last_step = step
                print(f"Mirrored step {step}", flush=True)
            if not args.follow or last_step >= args.final_step:
                break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
