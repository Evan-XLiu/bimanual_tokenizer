"""Canonical core primitives for BAR action-token policies."""

from praxis.policies.bar.core.decode_info import (
    DecodeMetadata,
    TokenRegime,
    build_decode_info,
    build_fixed_length_decode_info,
    build_variable_length_decode_info,
    resolve_decode_sampling,
    resolve_max_new_tokens,
)
from praxis.policies.bar.core.decode_schedule import (
    build_step_attention_mask,
    plan_generation_schedule,
    resolve_decode_keep_k,
)
from praxis.policies.bar.core.sampling import (
    sample_next_token_ids,
)
from praxis.policies.bar.core.stop import (
    DecodeStopMode,
    validate_decode_keep_k_for_stop_mode,
    validate_decode_stop_mode,
    validate_decode_stop_tokenizer_contract,
    validate_prefix_decode_budget,
)
from praxis.policies.bar.core.vlm import (
    run_blockwise_decode,
)

__all__ = [
    "DecodeMetadata",
    "DecodeStopMode",
    "TokenRegime",
    "build_decode_info",
    "build_fixed_length_decode_info",
    "build_variable_length_decode_info",
    "build_step_attention_mask",
    "plan_generation_schedule",
    "resolve_decode_keep_k",
    "resolve_decode_sampling",
    "resolve_max_new_tokens",
    "run_blockwise_decode",
    "sample_next_token_ids",
    "validate_decode_keep_k_for_stop_mode",
    "validate_decode_stop_mode",
    "validate_decode_stop_tokenizer_contract",
    "validate_prefix_decode_budget",
]
