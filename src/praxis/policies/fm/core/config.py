"""Config dataclass for flow-matching VLM/action-expert policies."""

from __future__ import annotations

from dataclasses import dataclass

from praxis.policies.core.action_expert.config import ActionExpertPolicyConfig


@dataclass
class PolicyConfig(ActionExpertPolicyConfig):
    """Hyperparameters shared by FM VLM/action-expert policies."""

    name: str = "fm_pali"
