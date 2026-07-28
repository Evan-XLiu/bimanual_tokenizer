#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Export a Praxis X-Tokenizer checkpoint for offline cache inference."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from praxis.codecs.latents.xtokenizer.inference import (
    export_xtokenizer_checkpoint,
    load_exported_xtokenizer,
)

__all__ = ["export_xtokenizer_checkpoint", "load_exported_xtokenizer", "main"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    artifact_dir = export_xtokenizer_checkpoint(
        args.checkpoint_dir,
        args.output_dir,
    )
    print(artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
