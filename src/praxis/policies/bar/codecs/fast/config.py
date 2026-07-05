# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared FAST policy config helpers."""

from __future__ import annotations

from praxis.codecs.tokens.fast.config import (
    default_fast_tokenizer_config,
)

FAST_RUNTIME_OVERRIDE_KEYS = frozenset({"decode_failure_mode", "decode_strict"})


def default_fast_action_tokenizer_config() -> dict[str, object]:
    """Return a fresh default FAST action-tokenizer config payload."""
    return default_fast_tokenizer_config()
