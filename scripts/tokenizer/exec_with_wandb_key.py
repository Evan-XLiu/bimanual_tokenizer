#!/usr/bin/env python3
"""Read a private W&B API key and replace this process with a command."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: exec_with_wandb_key.py KEY_FILE COMMAND [ARG ...]")
    key = Path(sys.argv[1]).read_text().strip()
    if not key:
        raise SystemExit("W&B API key file is empty")
    environment = dict(os.environ)
    environment["WANDB_API_KEY"] = key
    os.execvpe(sys.argv[2], sys.argv[2:], environment)


if __name__ == "__main__":
    main()
