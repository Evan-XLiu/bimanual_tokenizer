#!/usr/bin/env python3
"""Securely save a Hugging Face upload token without shell history or echo."""

from __future__ import annotations

import getpass
import os
from pathlib import Path


def main() -> None:
    destination = Path.home() / ".config/bimanual_tokenizer/hf_evanxw0715.key"
    token = getpass.getpass("New Hugging Face write token: ").strip()
    if not token.startswith("hf_") or any(char.isspace() for char in token):
        raise SystemExit("Invalid token format; nothing was saved.")

    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(token)
        os.replace(temporary, destination)
        destination.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved token to {destination} with mode 600.")


if __name__ == "__main__":
    main()
