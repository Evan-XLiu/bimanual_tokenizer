# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Config dataclass for knowledge-insulated VLM/action-expert policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from praxis.policies.core.action_expert.config import ActionExpertPolicyConfig
from praxis.policies.core.action_tokens import action_tokenizer_config_factory
from praxis.policies.core.config import (
    non_negative_float,
    non_negative_int,
    positive_int,
)
from praxis.policies.core.vlm.loss import (
    validate_block_ce_weighting,
    validate_ce_vocab_scope,
)


@dataclass
class PolicyConfig(ActionExpertPolicyConfig):
    """Hyperparameters shared by KI VLM/action-expert policies."""

    PRETRAINED_RUNTIME_OVERRIDE_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "block_ce_weighting",
            "action_expert_detach_vlm",
            "ce_warmstart_steps",
            "ki_end_step",
            "ce_loss_weight",
            "ce_vocab_scope",
            "default_instruction",
            "flow_loss_weight",
            "flow_sampling_steps",
            "n_action_steps",
        }
    )

    name: str = "ki_oat_pali"
    lora_modules_to_save: tuple[str, ...] = ("lm_head", "embed_tokens")

    action_tokenizer: dict[str, Any] = field(
        default_factory=action_tokenizer_config_factory("oat")
    )
    freeze_action_tokenizer: bool = True
    action_token_target_key: str | None = None
    action_token_length_key: str | None = None
    num_action_tokens: int | None = None
    action_token_template: str = "<|action_{i}|>"
    action_mask_token_template: str = "<|action_mask|>"
    action_embed_init: Literal["normal", "avg", "zero"] = "normal"
    ce_loss_weight: float = 1.0
    ce_warmstart_steps: int = 0
    ki_end_step: int | None = None
    # KI insulates the VLM from continuous-action gradients; CE still trains it.
    action_expert_detach_vlm: bool = True
    block_ce_weighting: str = "block_mean"
    ce_vocab_scope: str = "action"
    ce_supervise_eos: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.ce_loss_weight = non_negative_float(self.ce_loss_weight, "ce_loss_weight")
        self.ce_warmstart_steps = non_negative_int(
            self.ce_warmstart_steps,
            "ce_warmstart_steps",
        )
        self.block_ce_weighting = validate_block_ce_weighting(self.block_ce_weighting)
        if self.ki_end_step is not None:
            self.ki_end_step = non_negative_int(self.ki_end_step, "ki_end_step")
            if self.ki_end_step < self.ce_warmstart_steps:
                raise ValueError("ki_end_step must be >= ce_warmstart_steps")
        self.ce_vocab_scope = validate_ce_vocab_scope(self.ce_vocab_scope)
        self.ce_supervise_eos = bool(self.ce_supervise_eos)
        if (
            self.action_token_target_key is not None
            and not self.action_token_target_key.strip()
        ):
            raise ValueError(
                "action_token_target_key must be non-empty when configured."
            )
        if self.num_action_tokens is not None:
            self.num_action_tokens = positive_int(
                self.num_action_tokens, "num_action_tokens"
            )
