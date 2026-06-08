# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""FAST inference helpers shared by FAST policy families."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from praxis.policies.bar.core.token_sequence import pad_variable_token_id_lists

_VALID_FAST_DECODE_FAILURE_MODES = frozenset({"raise", "warn_zero"})


def validate_fast_decode_failure_mode(mode: str) -> str:
    """Validate configured FAST decode failure mode."""
    if mode not in _VALID_FAST_DECODE_FAILURE_MODES:
        choices = ", ".join(sorted(_VALID_FAST_DECODE_FAILURE_MODES))
        raise ValueError(
            f"decode_failure_mode must be one of [{choices}], got {mode!r}."
        )
    return mode


def detokenize_fast_actions(
    *,
    action_tokenizer: Any,
    tokens: Tensor,
    decode_strict: bool | None,
    decode_failure_mode: str,
    action_horizon: int,
    action_dim: int,
    policy_name: str,
    logger: logging.Logger | None = None,
) -> Tensor:
    """Detokenize FAST action tokens with configurable strictness/failure policy."""
    mode = validate_fast_decode_failure_mode(decode_failure_mode)
    try:
        return action_tokenizer.detokenize(tokens, strict_decode=decode_strict)
    except (ValueError, RuntimeError) as exc:
        if mode == "raise":
            raise
        log = logger or logging.getLogger(__name__)
        token_len = int(tokens.shape[1]) if tokens.ndim >= 2 else -1
        log.warning(
            "FAST decode failed in %s (strict=%s, mode=%s, batch=%d, token_len=%d): %s. "
            "Returning zero actions.",
            policy_name,
            decode_strict,
            mode,
            int(tokens.shape[0]),
            token_len,
            exc,
        )
        return torch.zeros(
            (int(tokens.shape[0]), int(action_horizon), int(action_dim)),
            dtype=torch.float32,
            device=tokens.device,
        )


def detokenize_fast_action_id_lists(
    *,
    action_tokenizer: Any,
    batch_token_ids: list[list[int]],
    device: torch.device,
    pad_token_id: int,
    decode_strict: bool | None,
    decode_failure_mode: str,
    action_horizon: int,
    action_dim: int,
    policy_name: str,
    logger: logging.Logger | None = None,
) -> Tensor:
    """Pad FAST token-id lists, then detokenize them with FAST failure handling."""
    tokens = pad_variable_token_id_lists(
        batch_token_ids,
        device=device,
        pad_value=int(pad_token_id),
    )
    with torch.inference_mode():
        return detokenize_fast_actions(
            action_tokenizer=action_tokenizer,
            tokens=tokens,
            decode_strict=decode_strict,
            decode_failure_mode=decode_failure_mode,
            action_horizon=action_horizon,
            action_dim=action_dim,
            policy_name=policy_name,
            logger=logger,
        )
