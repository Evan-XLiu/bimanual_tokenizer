# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared VLM-conditioned action-expert modules for VLA policies."""

from praxis.policies.core.action_expert.expert import ActionExpert, ActionExpertLayer
from praxis.policies.core.action_expert.flow import (
    FlowSample,
    sample_flow_training_point,
)
from praxis.policies.core.action_expert.integrated import (
    IntegratedVlmActionExpertForward,
    VlmActionExpertModel,
    VlmActionExpertPrefixAdapter,
    context_position_offsets,
)
from praxis.policies.core.action_expert.policy import ActionExpertPolicyMixin
from praxis.policies.core.action_expert.prefix_cache import (
    PrefixCache,
    build_prefix_cache,
    cache_to_device,
    flatten_cache_tensor,
    validate_cache_matches_mask,
)
from praxis.policies.core.action_expert.sampling import sample_actions_euler
from praxis.policies.core.vlm.prefix import LayerKV, VlmAttentionGeometry

__all__ = [
    "ActionExpert",
    "ActionExpertLayer",
    "ActionExpertPolicyMixin",
    "FlowSample",
    "IntegratedVlmActionExpertForward",
    "LayerKV",
    "PrefixCache",
    "VlmAttentionGeometry",
    "VlmActionExpertModel",
    "VlmActionExpertPrefixAdapter",
    "build_prefix_cache",
    "cache_to_device",
    "context_position_offsets",
    "flatten_cache_tensor",
    "sample_actions_euler",
    "sample_flow_training_point",
    "validate_cache_matches_mask",
]
