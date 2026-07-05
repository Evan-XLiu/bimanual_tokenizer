# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Base policy for end-to-end flow-matching VLM/action-expert training."""

from __future__ import annotations

from typing import Any, ClassVar, cast

from torch import Tensor

from praxis.core.types import PolicyIO
from praxis.policies.core import BasePolicy
from praxis.policies.core.action_expert import (
    ActionExpert,
    ActionExpertPolicyMixin,
    VlmActionExpertModel,
)
from praxis.policies.core.construction import (
    build_config_policy,
    coerce_policy_config,
)
from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.vlm.checkpoint import VlmCheckpointFilterMixin
from praxis.policies.core.vlm.optim import build_vlm_param_groups
from praxis.policies.core.vlm.prefix import VlmInputBatch
from praxis.policies.fm.core.config import PolicyConfig


class PolicyBase(VlmCheckpointFilterMixin, ActionExpertPolicyMixin, BasePolicy):
    """Shared lifecycle and objective wiring for FM policies."""

    CONFIG_CLASS: ClassVar[type[PolicyConfig] | None] = PolicyConfig
    config: PolicyConfig
    vlm: Any
    processor: Any
    action_expert: ActionExpert | None
    vlm_action_expert: VlmActionExpertModel | None

    def __init__(self, config: PolicyConfig | dict[str, Any], **kwargs) -> None:
        config = coerce_policy_config(
            config,
            config_cls=self.CONFIG_CLASS,
            owner_name=self.__class__.__name__,
        )
        super().__init__(**kwargs)
        self.config = cast(PolicyConfig, config)
        self._initialize_action_expert_members()
        self._load_configured_vlm_runtime()
        self._apply_configured_lora(require_vocab_modules=False)
        self._enable_configured_gradient_checkpointing()

    @classmethod
    def from_io(cls, io: PolicyIO, **config_overrides):
        resolved_overrides = dict(config_overrides)
        resolved_overrides["action_dim"] = cls._resolve_action_dim_for_from_io(
            configured_action_dim=resolved_overrides.get("action_dim"),
            io=io,
            owner=cls.__name__,
        )
        return build_config_policy(
            cls,
            io=io,
            config_cls=cls.CONFIG_CLASS,
            resolved_overrides=resolved_overrides,
        )

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        raise NotImplementedError

    def set_io(self, io: PolicyIO) -> None:
        super().set_io(io)
        self._configure_action_queue(n_action_steps=self.config.n_action_steps)
        self._build_action_expert(io)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        raw_batch = dict(batch)
        norm_batch = self._prepare_normalized_batch(raw_batch)
        raw_actions = self._actions_from_batch(raw_batch)
        norm_actions = self._actions_from_batch(norm_batch).to(
            device=raw_actions.device,
            dtype=raw_actions.dtype,
        )
        instructions = extract_instructions(
            raw_batch,
            language_key=None if self.io is None else self.io.language_key,
        )
        vlm_inputs = self._build_vlm_inputs_for_action_expert(
            self._observations_for_state_prefix(raw_batch, norm_batch=norm_batch),
            instructions=instructions,
        )
        with self._autocast_context():
            flow_loss = self.compute_integrated_flow_loss(
                norm_batch,
                actions=norm_actions,
                vlm_inputs=vlm_inputs,
                detach_prefix=False,
            )
        total = float(self.config.flow_loss_weight) * flow_loss
        return total, {
            "flow_loss": float(flow_loss.detach().item()),
        }

    def get_optim_params(self) -> list[dict]:
        return build_vlm_param_groups(
            named_parameters=self.named_parameters(),
            use_lora=bool(self.config.use_lora),
            is_vocab_matrix_param_name=self._is_vocab_matrix_param_name,
        )

    @staticmethod
    def _is_vocab_matrix_param_name(name: str) -> bool:
        base_name = name[4:] if name.startswith("vlm.") else name
        if not base_name.endswith(".weight"):
            return False
        return ("embed_tokens" in base_name) or ("lm_head" in base_name)
