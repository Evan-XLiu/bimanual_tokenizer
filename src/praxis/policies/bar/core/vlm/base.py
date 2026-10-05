# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared backbone-agnostic base for VLM action-token policies."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from praxis.codecs.factory import make_codec
from praxis.core.normalizer import Normalize
from praxis.core.types import NormMode
from praxis.policies.bar.core.decode_info import (
    build_fixed_length_decode_info,
    build_variable_length_decode_info,
    resolve_max_new_tokens,
)
from praxis.policies.bar.core.decode_schedule import resolve_decode_keep_k
from praxis.policies.bar.core.stop import (
    validate_decode_keep_k_for_stop_mode,
    validate_decode_stop_tokenizer_contract,
    validate_prefix_decode_budget,
)
from praxis.policies.bar.core.token_sequence import (
    detokenize_fixed_action_ids,
    resolve_max_code_seq_len,
    resolve_pad_horizon,
    tokenize_fixed_actions,
    swap_btt_arm_token_blocks,
)
from praxis.policies.bar.core.vlm.decoding import run_blockwise_decode
from praxis.policies.core.action_tokens import (
    ActionTokenPolicyBase,
)
from praxis.policies.core.construction import coerce_policy_config
from praxis.policies.core.state_prefix import StatePrefixPolicyMixin
from praxis.policies.core.vlm.checkpoint import VlmCheckpointFilterMixin
from praxis.policies.core.vlm.dtypes import parse_vlm_dtype
from praxis.policies.core.vlm.fsdp import select_vlm_fsdp_wrap_modules
from praxis.policies.core.vlm.lora import (
    apply_lora_to_vlm,
)
from praxis.policies.core.vlm.optim import build_vlm_param_groups
from praxis.policies.core.vlm.prefix import VlmInputBatch, VlmPrefixLogitsAdapter
from praxis.policies.core.vlm.schedule import (
    PrefixAttentionMode,
    resolve_block_schedule,
)
from praxis.policies.core.vlm.supervision import IGNORE_INDEX
from praxis.policies.core.vlm.tokenization import require_token_codec
from praxis.policies.core.vlm.training import (
    run_action_token_ce_prefix_training_step,
)
from praxis.policies.core.vlm.vocab import (
    ActionTokenVocabulary,
    build_trainable_vocab_rows,
    install_vocab_row_gradient_masks,
    resolve_num_action_tokens,
    setup_action_token_vocabulary,
)


class VlmActionTokenPolicyBase(
    StatePrefixPolicyMixin,
    VlmCheckpointFilterMixin,
    ActionTokenPolicyBase,
):
    """Common VLM policy behavior shared across Smol/Qwen/Pali families."""

    CONFIG_CLASS: ClassVar[type[Any] | None] = None
    ACTION_TOKEN_OWNER_NAME: str | None = None
    _SCHEDULE_STRIP_KEYS: ClassVar[tuple[str, ...]] = ("labels",)
    _PREFILL_EXTRA_KEYS: ClassVar[tuple[str, ...]] = ()
    processor: Any

    def __init__(self, config: Any, **kwargs) -> None:
        config = coerce_policy_config(
            config,
            config_cls=self.CONFIG_CLASS,
            owner_name=self.__class__.__name__,
        )
        self.max_code_seq_len: int | None = None
        self.max_decode_tokens: int | None = None
        self.action_token_vocabulary: ActionTokenVocabulary | None = None
        self._action_mask_token_id: int | None = None
        self._action_mask_token: str | None = None
        self._mask_sentinel: int | None = None
        self._vlm_ce_prefix_adapter: VlmPrefixLogitsAdapter | None = None
        self._initialize_state_prefix_members()

        super().__init__(**kwargs)
        self.config = config
        self._vocab_row_grad_mask_hooks: list[RemovableHandle] = []

        self._load_vlm(config.vlm_model_name, config.vlm_dtype)
        if config.freeze_vision_encoder:
            self._freeze_vision_encoder_modules()
        self._setup_state_prefix_token()

        self.action_tokenizer = require_token_codec(
            make_codec(config.action_tokenizer),
            owner_name=self.__class__.__name__,
        )
        self._validate_frozen_tokenizer_contract()
        if config.freeze_action_tokenizer:
            self.action_tokenizer.requires_grad_(False)
            self.action_tokenizer.eval()

        owner_name = self.ACTION_TOKEN_OWNER_NAME or self.__class__.__name__
        num_action_tokens = resolve_num_action_tokens(
            codebook_size=self.action_tokenizer.codebook_size,
            configured=self.config.num_action_tokens,
            owner_name=owner_name,
        )
        self._setup_action_tokens(num_action_tokens)
        self._eos_token_id: int | None = self._resolve_eos_token_id()

        if config.use_lora:
            self.vlm = apply_lora_to_vlm(
                vlm=self.vlm,
                lora_r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_target_modules=self.config.lora_target_modules,
                lora_dropout=self.config.lora_dropout,
                lora_modules_to_save=self.config.lora_modules_to_save,
            )

        if config.gradient_checkpointing:
            cast(Any, self.vlm).gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        self._init_token_modeling()
        self._validate_eos_configuration()
        self._install_vocab_row_gradient_masks()

    def train(self, mode: bool = True) -> VlmActionTokenPolicyBase:
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self._ensure_frozen_vision_eval()
        if self.config.freeze_action_tokenizer:
            self.action_tokenizer.eval()
        return self

    @torch.inference_mode()
    def predict_action(self, obs: dict[str, Tensor], **kwargs) -> Tensor:
        return self._predict_action_from_chunk_queue(
            obs,
            predict_action_chunk=lambda stacked_obs, **chunk_kwargs: (
                self.predict_action_chunk(
                    stacked_obs,
                    normalize_obs=False,
                    **chunk_kwargs,
                )
            ),
            **kwargs,
        )

    def fsdp_wrap_modules(self) -> tuple[nn.Module, ...]:
        return select_vlm_fsdp_wrap_modules(
            cast(nn.Module, self.vlm),
            use_lora=bool(getattr(getattr(self, "config", None), "use_lora", False)),
        )

    def fsdp_forward_methods(self) -> tuple[str, ...]:
        return ("predict_action", "predict_action_chunk")

    def supports_training_strategy(self, strategy: str) -> bool:
        return strategy in {"ddp", "fsdp"}

    def validate_distributed_strategy(self, strategy: str) -> None:
        if not self.supports_training_strategy(strategy):
            raise ValueError(
                f"{self.__class__.__name__} does not support training strategy {strategy!r}."
            )
        if strategy != "fsdp":
            return
        config = getattr(self, "config", None)
        if not bool(getattr(config, "freeze_action_tokenizer", True)):
            raise ValueError(
                "VLM policy FSDP currently requires freeze_action_tokenizer=true. "
                "Use DDP when jointly training the action tokenizer."
            )
        fsdp_wrap_modules = getattr(self, "fsdp_wrap_modules", None)
        if not callable(fsdp_wrap_modules):
            raise ValueError(
                "VLM policy FSDP requires the policy to expose fsdp_wrap_modules()."
            )
        typed_fsdp_wrap_modules = cast(
            Callable[[], Iterable[nn.Module]],
            fsdp_wrap_modules,
        )
        if not tuple(typed_fsdp_wrap_modules()):
            raise ValueError(
                "VLM policy FSDP requires fsdp_wrap_modules() to select at least "
                "one parameterized VLM block."
            )

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        raise NotImplementedError

    def _freeze_vision_encoder_modules(self) -> None:
        """Family hook for freezing vision modules at construction time."""

    def _ensure_frozen_vision_eval(self) -> None:
        """Family hook for forcing frozen vision modules into eval mode."""

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        raise NotImplementedError

    def _build_vlm_prefix_adapter(self) -> VlmPrefixLogitsAdapter:
        raise NotImplementedError

    def _require_vlm_ce_prefix_adapter(self) -> VlmPrefixLogitsAdapter:
        if self._vlm_ce_prefix_adapter is None:
            self._vlm_ce_prefix_adapter = self._build_vlm_prefix_adapter()
        return self._vlm_ce_prefix_adapter

    def _setup_action_tokens(self, num_action_tokens: int) -> None:
        self_any = cast(Any, self)
        self.action_token_vocabulary = setup_action_token_vocabulary(
            tokenizer=self_any.processor.tokenizer,
            vlm=self_any.vlm,
            num_action_tokens=int(num_action_tokens),
            token_template=self_any.config.action_token_template,
            mask_token=self_any.config.action_mask_token_template,
            init_mode=self_any.config.action_embed_init,
        )
        self._action_mask_token_id = self.action_token_vocabulary.mask_token_id
        self._action_mask_token = self.action_token_vocabulary.mask_token

    def _require_action_token_vocabulary(self) -> ActionTokenVocabulary:
        if self.action_token_vocabulary is None:
            raise RuntimeError("Action token vocabulary has not been initialized.")
        return self.action_token_vocabulary

    @property
    def _supervise_eos(self) -> bool:
        return self.config.decode_stop_mode == "eos"

    def _resolve_eos_token_id(self) -> int | None:
        eos_id = getattr(cast(Any, self).processor.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int) and eos_id >= 0:
            return eos_id
        return None

    def _validate_eos_configuration(self) -> None:
        if self._supervise_eos and self._eos_token_id is None:
            raise ValueError(
                f"{self.__class__.__name__} requires tokenizer.eos_token_id for EOS supervision."
            )

    @staticmethod
    def _is_vocab_matrix_param_name(name: str) -> bool:
        base_name = name[4:] if name.startswith("vlm.") else name
        if not base_name.endswith(".weight"):
            return False
        return ("embed_tokens" in base_name) or ("lm_head" in base_name)

    def _build_trainable_vocab_rows(self, *, vocab_size: int) -> torch.LongTensor:
        vocab = self._require_action_token_vocabulary()
        return build_trainable_vocab_rows(
            vocab=vocab,
            vocab_size=vocab_size,
            owner_name=self.__class__.__name__,
            supervise_eos=self._supervise_eos,
            eos_token_id=cast(int | None, self._eos_token_id),
        )

    def _install_vocab_row_gradient_masks(self) -> None:
        self_any = cast(Any, self)
        install_vocab_row_gradient_masks(
            named_parameters=self_any.vlm.named_parameters(),
            use_lora=self_any.config.use_lora,
            is_vocab_matrix_param_name=self._is_vocab_matrix_param_name,
            build_trainable_rows=lambda vocab_size: self._build_trainable_vocab_rows(
                vocab_size=vocab_size
            ),
            hook_handles=cast(
                list[RemovableHandle], self_any._vocab_row_grad_mask_hooks
            ),
        )

    def _require_prefix_attention(self) -> PrefixAttentionMode:
        prefix_attention = getattr(self, "_PREFIX_ATTENTION", None)
        if prefix_attention not in ("bidirectional", "causal"):
            raise RuntimeError(
                f"{self.__class__.__name__} must define class attr _PREFIX_ATTENTION as "
                "'bidirectional' or 'causal'."
            )
        return cast(PrefixAttentionMode, prefix_attention)

    def _init_token_modeling(self) -> None:
        self.tokenizer_vocab_size = int(self.action_tokenizer.codebook_size)
        if getattr(self.config, "btt_arm_order", "left_then_right") == "right_then_left":
            if (self.action_tokenizer.max_code_seq_len != 16
                    or tuple(self.action_tokenizer.block_schedule) != (8, 8)):
                raise ValueError("right_then_left requires BTT's 16 tokens and (8, 8) schedule.")

    def tokenize_actions_to_policy_tokens(self, actions: Tensor) -> list[list[int]]:
        tokens = tokenize_fixed_actions(self.action_tokenizer, actions)
        if getattr(self.config, "btt_arm_order", "left_then_right") == "right_then_left":
            tokens = swap_btt_arm_token_blocks(tokens)
        return tokens

    def _decode_generated_action_tokens(
        self,
        batch_token_ids: list[list[int]],
        *,
        device: torch.device,
        horizon: int | None = None,
    ) -> Tensor:
        if getattr(self.config, "btt_arm_order", "left_then_right") == "right_then_left":
            if horizon not in (None, 16):
                raise ValueError("right_then_left requires decoding the full 16-token sequence.")
            batch_token_ids = swap_btt_arm_token_blocks(batch_token_ids)
        return detokenize_fixed_action_ids(
            action_tokenizer=self.action_tokenizer,
            batch_token_ids=batch_token_ids,
            device=device,
            horizon=resolve_pad_horizon(
                horizon,
                token_horizon=self._require_max_code_seq_len(),
            ),
            pad_value=0,
        )

    def _post_set_io(self) -> None:
        self.max_code_seq_len = resolve_max_code_seq_len(
            action_tokenizer=self.action_tokenizer,
            owner_name=self.__class__.__name__,
        )
        self._build_state_prefix_projector()
        self.max_decode_tokens = resolve_max_new_tokens(
            self.config.max_new_tokens,
            codec_max_code_seq_len=getattr(
                self.action_tokenizer, "max_code_seq_len", None
            ),
            owner_name=self.__class__.__name__,
            minimum=self.max_code_seq_len,
        )
        self._mask_sentinel = int(self.tokenizer_vocab_size)
        self._validate_decode_stop_configuration()

    def _require_max_code_seq_len(self) -> int:
        if self.max_code_seq_len is None:
            raise RuntimeError(
                "max_code_seq_len must be initialized before block decode"
            )
        return int(self.max_code_seq_len)

    def _validate_decode_stop_configuration(self) -> None:
        validate_decode_keep_k_for_stop_mode(
            decode_stop_mode=self.config.decode_stop_mode,
            decode_keep_k=self.config.decode_keep_k,
        )
        validate_decode_stop_tokenizer_contract(
            decode_stop_mode=self.config.decode_stop_mode,
            action_tokenizer=self.action_tokenizer,
        )

        if self.config.decode_keep_k is not None:
            self._resolve_decode_keep_k()

    def _validate_prefix_decode_budget(self, keep_k: int) -> None:
        validate_prefix_decode_budget(
            action_tokenizer=self.action_tokenizer,
            keep_k=keep_k,
            token_horizon=self._require_max_code_seq_len(),
        )

    def _resolve_decode_keep_k(self, keep_k: int | None = None) -> int:
        raw_keep_k = self.config.decode_keep_k if keep_k is None else keep_k
        if self.config.decode_stop_mode != "fixed_budget":
            if raw_keep_k is not None:
                raise ValueError(
                    "decode_keep_k is only valid when "
                    "policy.decode_stop_mode='fixed_budget'."
                )
            return self._require_max_code_seq_len()

        resolved_keep_k = resolve_decode_keep_k(
            raw_keep_k,
            token_horizon=self._require_max_code_seq_len(),
        )
        self._validate_prefix_decode_budget(resolved_keep_k)
        return resolved_keep_k

    def _resolve_block_schedule(self) -> tuple[int, ...]:
        token_horizon = int(self._require_max_code_seq_len())
        tokenizer_schedule = getattr(self.action_tokenizer, "block_schedule", None)
        return resolve_block_schedule(tokenizer_schedule, token_horizon=token_horizon)

    def _resolve_action_token_block_schedule(
        self, *, token_horizon: int
    ) -> tuple[int, ...]:
        tokenizer_schedule = getattr(self.action_tokenizer, "block_schedule", None)
        return resolve_block_schedule(tokenizer_schedule, token_horizon=token_horizon)

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        obs: dict[str, Tensor],
        *,
        normalize_obs: bool = True,
        **kwargs,
    ) -> tuple[Tensor, dict]:
        result = run_blockwise_decode(
            self,
            obs,
            normalize_obs=normalize_obs,
            decode_kwargs=dict(kwargs),
        )
        if self._supervise_eos:
            resolved_keep_k = self._resolve_decode_keep_k(kwargs.get("decode_keep_k"))
            return result.actions, build_variable_length_decode_info(
                pred_action_token_ids=result.token_ids,
                block_schedule=self._resolve_block_schedule(),
                decode_keep_k=resolved_keep_k,
                stopped_on_eos=result.stopped_on_eos,
            )
        return result.actions, build_fixed_length_decode_info(
            pred_action_token_ids=result.token_ids,
            block_schedule=self._resolve_block_schedule(),
        )

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        return run_action_token_ce_prefix_training_step(
            self,
            batch,
            prefix_adapter=self._require_vlm_ce_prefix_adapter(),
            ignore_index=IGNORE_INDEX,
            gradient_checkpointing=bool(self.config.gradient_checkpointing),
        )

    def set_normalization_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        modes: dict[str, NormMode] | None = None,
    ) -> None:
        self.normalize = None
        self.unnormalize_action = None
        if self.io is None:
            return

        obs_stats, obs_modes = self._collect_observation_norm_stats(
            stats,
            feature_modes=modes,
            features=self.io.state_features,
            use_image_defaults=False,
            missing_stats_warning=(
                "Missing normalization stats for %s; skipping state normalization for this key."
            ),
        )
        self.normalize = Normalize(obs_stats, obs_modes) if obs_modes else None

    def _autocast_context(self):
        device_type = next(self.vlm.parameters()).device.type
        dtype = parse_vlm_dtype(str(getattr(self.config, "vlm_dtype", "bfloat16")))
        if device_type in ("cuda", "cpu") and dtype is not torch.float32:
            return torch.autocast(device_type=device_type, dtype=dtype)
        return contextlib.nullcontext()

    def get_optim_params(self) -> list[dict]:
        is_vocab_matrix_param_name = cast(
            Callable[[str], bool], self._is_vocab_matrix_param_name
        )
        return build_vlm_param_groups(
            named_parameters=self.named_parameters(),
            use_lora=self.config.use_lora,
            is_vocab_matrix_param_name=is_vocab_matrix_param_name,
        )
