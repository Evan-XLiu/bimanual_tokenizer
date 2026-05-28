"""Shared decode metadata and decode-parameter helpers for action-token policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from praxis.policies.bar.core.decode_schedule import plan_generation_schedule


class TokenRegime(str, Enum):
    FIXED_LENGTH = "fixed_length"
    VARIABLE_LENGTH = "variable_length"


@dataclass
class DecodeMetadata:
    pred_action_token_ids: list[list[int]]
    decode_steps: int
    token_regime: str
    block_schedule: list[int] | None = None
    decode_keep_k: int | None = None
    stopped_on_eos: bool | None = None


def resolve_max_new_tokens(
    configured: int | None,
    *,
    codec_max_code_seq_len: int | None,
    owner_name: str,
    minimum: int | None = None,
) -> int:
    """Resolve generation budget from config or codec contract."""
    if configured is None:
        if codec_max_code_seq_len is None:
            raise ValueError(
                f"{owner_name} requires max_new_tokens for action codecs without max_code_seq_len."
            )
        resolved = int(codec_max_code_seq_len)
    else:
        resolved = int(configured)

    if minimum is not None and resolved < int(minimum):
        raise ValueError(
            f"{owner_name} max_new_tokens={resolved} must be >= {minimum}."
        )
    return resolved


def resolve_decode_sampling(
    *,
    default_temperature: float | None,
    default_top_k: int | None,
    override_temperature: float | None = None,
    override_top_k: int | None = None,
) -> tuple[float | None, int | None]:
    """Resolve call-time decode sampling overrides against config defaults."""
    temperature = (
        default_temperature if override_temperature is None else override_temperature
    )
    top_k = default_top_k if override_top_k is None else override_top_k

    if temperature is not None:
        temperature = float(temperature)
        if temperature < 0.0:
            raise ValueError(f"decode_temperature must be >= 0, got {temperature}.")
    if top_k is not None:
        top_k = int(top_k)
        if top_k < 1:
            raise ValueError(f"decode_top_k must be >= 1, got {top_k}.")
    return temperature, top_k


def build_decode_info(
    pred_action_token_ids: list[list[int]],
    *,
    token_regime: TokenRegime,
    decode_steps: int,
    block_schedule: list[int] | None = None,
    decode_keep_k: int | None = None,
    stopped_on_eos: bool | None = None,
) -> dict:
    """Build a stable decode metadata dict for policy inference outputs."""
    metadata = DecodeMetadata(
        pred_action_token_ids=pred_action_token_ids,
        decode_steps=int(decode_steps),
        token_regime=token_regime.value,
        block_schedule=block_schedule,
        decode_keep_k=decode_keep_k,
        stopped_on_eos=stopped_on_eos,
    )
    return asdict(metadata)


def build_variable_length_decode_info(
    *,
    pred_action_token_ids: list[list[int]],
    block_schedule: tuple[int, ...] | None = None,
    decode_keep_k: int | None = None,
    stopped_on_eos: bool | None = None,
) -> dict:
    """Build decode metadata for variable-length action-token policies."""
    decode_steps = max((len(ids) for ids in pred_action_token_ids), default=0)
    return build_decode_info(
        pred_action_token_ids=pred_action_token_ids,
        token_regime=TokenRegime.VARIABLE_LENGTH,
        decode_steps=decode_steps,
        block_schedule=list(block_schedule) if block_schedule is not None else None,
        decode_keep_k=decode_keep_k,
        stopped_on_eos=stopped_on_eos,
    )


def build_fixed_length_decode_info(
    *,
    pred_action_token_ids: list[list[int]],
    block_schedule: tuple[int, ...],
) -> dict:
    """Build decode metadata for fixed-length action-token policies."""
    decode_keep_k = int(len(pred_action_token_ids[0])) if pred_action_token_ids else 0
    decode_steps = (
        int(len(plan_generation_schedule(block_schedule, keep_k=decode_keep_k)))
        if decode_keep_k > 0
        else 0
    )
    return build_decode_info(
        pred_action_token_ids,
        token_regime=TokenRegime.FIXED_LENGTH,
        block_schedule=list(block_schedule),
        decode_steps=decode_steps,
        decode_keep_k=decode_keep_k,
    )
