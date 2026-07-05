# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""VLM-specific action-token helpers."""

from praxis.policies.bar.core.stop import DecodeStopMode
from praxis.policies.bar.core.vlm.base import VlmActionTokenPolicyBase
from praxis.policies.bar.core.vlm.config import VlmActionTokenPolicyConfig
from praxis.policies.bar.core.vlm.decode_tokens import (
    build_allowed_decode_token_ids,
    build_prediction_inputs,
    extract_contiguous_action_token_ids,
    raw_token_ids_to_vlm_tensor,
)
from praxis.policies.bar.core.vlm.decoding import (
    build_vlm_cached_step_inputs,
    build_vlm_prefill_inputs,
    run_blockwise_decode,
)

__all__ = [
    "VlmActionTokenPolicyBase",
    "DecodeStopMode",
    "VlmActionTokenPolicyConfig",
    "build_allowed_decode_token_ids",
    "build_prediction_inputs",
    "build_vlm_cached_step_inputs",
    "build_vlm_prefill_inputs",
    "extract_contiguous_action_token_ids",
    "raw_token_ids_to_vlm_tensor",
    "run_blockwise_decode",
]
