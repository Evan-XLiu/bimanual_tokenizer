# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared FAST variable-length policy mixin for VLM backbones."""

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from praxis.policies.bar.codecs.fast.decode import (
    detokenize_fast_action_id_lists,
    validate_fast_decode_failure_mode,
)
from praxis.policies.bar.core.token_sequence import (
    resolve_variable_action_token_budget,
    resolve_variable_decode_budget,
    resolve_variable_pad_token_id,
)
from praxis.policies.core.action_tokens import VariableLengthActionTokenMixin

logger = logging.getLogger(__name__)


class FastVlmPolicyMixin(VariableLengthActionTokenMixin):
    """Shared variable-length FAST behavior for VLM-backed action-token policies."""

    CONFIG_CLASS: ClassVar[type[Any] | None] = None
    decode_strict: bool | None
    decode_failure_mode: str
    pad_token_id: int

    def __init__(self, config: Any, **kwargs: Any) -> None:
        config_cls = self.CONFIG_CLASS
        if isinstance(config, dict):
            if config_cls is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} must define CONFIG_CLASS to accept config dictionaries."
                )
            resolved_config = config_cls.from_dict(config)
        else:
            resolved_config = config
        self.decode_strict = cast(bool | None, resolved_config.decode_strict)
        self.decode_failure_mode = validate_fast_decode_failure_mode(
            cast(str, resolved_config.decode_failure_mode)
        )
        cast(Any, super()).__init__(config=resolved_config, **kwargs)

    def _post_set_io(self) -> None:
        self_any = cast(Any, self)
        cfg = self_any.config
        self_any.max_action_tokens = resolve_variable_action_token_budget(
            cfg.max_new_tokens,
            action_tokenizer=self_any.action_tokenizer,
            owner_name=self.__class__.__name__,
        )
        self_any.max_decode_tokens = resolve_variable_decode_budget(
            cfg.max_new_tokens,
            action_tokenizer=self_any.action_tokenizer,
            owner_name=self.__class__.__name__,
            eos_headroom=1,
        )
        self_any._mask_sentinel = int(self_any.tokenizer_vocab_size)
        self_any._validate_decode_stop_configuration()

    def _init_token_modeling(self) -> None:
        super_any = cast(Any, super())
        super_any._init_token_modeling()
        self_any = cast(Any, self)
        self.pad_token_id = resolve_variable_pad_token_id(
            action_tokenizer=self_any.action_tokenizer,
            tokenizer_vocab_size=self_any.tokenizer_vocab_size,
        )

    def _require_max_decode_tokens(self) -> int:
        max_decode_tokens = cast(Any, self).max_decode_tokens
        if max_decode_tokens is None:
            raise RuntimeError(
                "max_decode_tokens must be initialized before FAST decode."
            )
        return int(max_decode_tokens)

    def _require_max_code_seq_len(self) -> int:
        max_action_tokens = getattr(self, "max_action_tokens", None)
        if max_action_tokens is None:
            raise RuntimeError(
                "max_action_tokens must be initialized before FAST decode."
            )
        return int(max_action_tokens)

    def _resolve_block_schedule(self) -> tuple[int, ...]:
        return (1,) * self._require_max_decode_tokens()

    def _decode_generated_action_tokens(
        self,
        batch_token_ids: list[list[int]],
        *,
        device: torch.device,
        horizon: int | None = None,
    ) -> Tensor:
        del horizon
        self_any = cast(Any, self)
        return detokenize_fast_action_id_lists(
            action_tokenizer=self_any.action_tokenizer,
            batch_token_ids=batch_token_ids,
            device=device,
            pad_token_id=self.pad_token_id,
            decode_strict=self.decode_strict,
            decode_failure_mode=self.decode_failure_mode,
            action_horizon=self_any.config.action_horizon,
            action_dim=self_any.config.action_dim,
            policy_name=self.__class__.__name__,
            logger=logger,
        )
