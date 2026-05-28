"""Shared stop-controller config types for action-token policies."""

from __future__ import annotations

from typing import Final, Literal, cast

DecodeStopMode = Literal["eos", "fixed_budget"]

_DECODE_STOP_MODES: Final[frozenset[str]] = frozenset({"eos", "fixed_budget"})


def validate_decode_stop_mode(value: object) -> DecodeStopMode:
    """Validate and normalize a policy-side decode stop mode."""
    if not isinstance(value, str):
        raise TypeError(
            f"decode_stop_mode must be a string, got {type(value).__name__}."
        )
    if value not in _DECODE_STOP_MODES:
        expected = ", ".join(sorted(_DECODE_STOP_MODES))
        raise ValueError(
            f"Unsupported decode_stop_mode={value!r}. Expected one of: {expected}."
        )
    return cast(DecodeStopMode, value)


def validate_decode_keep_k_for_stop_mode(
    *,
    decode_stop_mode: DecodeStopMode,
    decode_keep_k: int | None,
) -> None:
    """Ensure an explicit keep budget is only used by fixed-budget decode."""
    if decode_stop_mode == "fixed_budget" or decode_keep_k is None:
        return
    raise ValueError(
        "policy.decode_keep_k is only valid when "
        "policy.decode_stop_mode='fixed_budget'."
    )


def validate_decode_stop_tokenizer_contract(
    *,
    decode_stop_mode: DecodeStopMode,
    action_tokenizer: object,
) -> None:
    """Validate tokenizer capabilities required by a policy stop controller."""
    if decode_stop_mode == "eos":
        if not bool(getattr(action_tokenizer, "variable_length_codes", False)):
            raise ValueError(
                "policy.decode_stop_mode='eos' requires an action tokenizer "
                "with variable_length_codes=True."
            )
        return


def validate_prefix_decode_budget(
    *,
    action_tokenizer: object,
    keep_k: int,
    token_horizon: int,
) -> None:
    """Validate that a tokenizer can detokenize the requested prefix budget."""
    if int(keep_k) >= int(token_horizon):
        return
    if bool(getattr(action_tokenizer, "supports_prefix_decode", False)):
        return
    raise ValueError(
        "decode_keep_k shorter than token_horizon requires an action tokenizer "
        "with supports_prefix_decode=True."
    )
