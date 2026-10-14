# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Base policy for knowledge-insulated VLM/action-expert co-training."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from praxis.codecs.factory import make_codec
from praxis.policies.core.action_expert import (
    ActionExpert,
    ActionExpertPolicyMixin,
    VlmActionExpertModel,
    sample_flow_training_point,
)
from praxis.policies.core.action_tokens import ActionTokenPolicyBase
from praxis.policies.core.construction import coerce_policy_config
from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.vlm.checkpoint import VlmCheckpointFilterMixin
from praxis.policies.core.vlm.optim import build_vlm_param_groups
from praxis.policies.core.vlm.prefix import VlmInputBatch, VlmPrefixLogitsAdapter
from praxis.policies.core.vlm.schedule import (
    PrefixAttentionMode,
    resolve_block_schedule,
)
from praxis.policies.core.vlm.supervision import IGNORE_INDEX
from praxis.policies.core.vlm.tokenization import (
    require_token_codec,
    tokenize_actions,
)
from praxis.policies.core.vlm.training import (
    build_prepared_action_token_attention_mask,
    compute_prepared_action_token_ce_loss_from_hidden,
    prepare_action_token_ce,
    prepare_action_token_ce_from_token_targets,
    run_action_token_ce_prefix_forward,
)
from praxis.policies.core.vlm.vocab import (
    ActionTokenVocabulary,
    build_trainable_vocab_rows,
    install_vocab_row_gradient_masks,
    resolve_num_action_tokens,
    setup_action_token_vocabulary,
)
from praxis.policies.ki.core.config import PolicyConfig
from praxis.policies.ki.core.one_forward import (
    KiVlmPrefixAdapter,
    require_ki_one_forward_adapter,
)
from praxis.training.core.training_context import TrainingContext


class PolicyBase(
    VlmCheckpointFilterMixin,
    ActionExpertPolicyMixin,
    ActionTokenPolicyBase,
):
    """Shared lifecycle and objective wiring for KI policies."""

    CONFIG_CLASS: ClassVar[type[PolicyConfig] | None] = PolicyConfig
    config: PolicyConfig
    vlm: Any
    processor: Any
    action_tokenizer: Any
    action_expert: ActionExpert | None
    vlm_action_expert: VlmActionExpertModel | None
    _PREFIX_ATTENTION: ClassVar[PrefixAttentionMode]
    _SCHEDULE_STRIP_KEYS: ClassVar[tuple[str, ...]] = ("labels",)

    def __init__(self, config: PolicyConfig | dict[str, Any], **kwargs) -> None:
        config = coerce_policy_config(
            config,
            config_cls=self.CONFIG_CLASS,
            owner_name=self.__class__.__name__,
        )
        super().__init__(**kwargs)
        self.config = cast(PolicyConfig, config)
        self.action_token_vocabulary: ActionTokenVocabulary | None = None
        self._vlm_ce_prefix_adapter: VlmPrefixLogitsAdapter | None = None
        self._initialize_action_expert_members()
        self._vocab_row_grad_mask_hooks: list[RemovableHandle] = []
        self._training_context: TrainingContext = TrainingContext(global_step=0)

        self._load_configured_vlm_runtime()

        self.action_tokenizer = require_token_codec(
            make_codec(self.config.action_tokenizer),
            owner_name=self.__class__.__name__,
        )
        self._validate_frozen_tokenizer_contract()
        if self.config.freeze_action_tokenizer:
            self.action_tokenizer.requires_grad_(False)
            self.action_tokenizer.eval()

        self._setup_action_tokens()
        self._eos_token_id: int | None = self._resolve_eos_token_id()
        self._validate_ce_eos_configuration()
        self._apply_configured_lora(require_vocab_modules=True)
        self._enable_configured_gradient_checkpointing()
        self._install_vocab_row_gradient_masks()

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None = None,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        raise NotImplementedError

    def _build_vlm_prefix_adapter(self) -> KiVlmPrefixAdapter:
        raise NotImplementedError

    def _require_vlm_ce_prefix_adapter(self) -> VlmPrefixLogitsAdapter:
        if self._vlm_ce_prefix_adapter is None:
            self._vlm_ce_prefix_adapter = require_ki_one_forward_adapter(
                self._build_vlm_prefix_adapter()
            )
        return self._vlm_ce_prefix_adapter

    def _require_prefix_attention(self) -> PrefixAttentionMode:
        prefix_attention = getattr(self, "_PREFIX_ATTENTION", None)
        if prefix_attention not in ("bidirectional", "causal"):
            raise RuntimeError(
                f"{self.__class__.__name__} must define _PREFIX_ATTENTION as "
                "'bidirectional' or 'causal'."
            )
        return cast(PrefixAttentionMode, prefix_attention)

    def validate_distributed_strategy(self, strategy: str) -> None:
        super().validate_distributed_strategy(strategy)
        if strategy != "fsdp":
            return
        if not self.config.freeze_action_tokenizer:
            raise ValueError(
                "KI FSDP currently requires freeze_action_tokenizer=true. "
                "Use DDP when jointly training the action tokenizer."
            )

    def train(self, mode: bool = True) -> PolicyBase:
        super().train(mode)
        if self.config.freeze_action_tokenizer:
            self.action_tokenizer.eval()
        return self

    def _post_set_io(self) -> None:
        if self.io is None:
            raise RuntimeError(
                "PolicyIO must be set before building the action expert."
            )
        self._build_action_expert(self.io)

    def set_training_context(self, context: TrainingContext) -> None:
        self._training_context = context

    def _setup_action_tokens(self) -> None:
        num_tokens = resolve_num_action_tokens(
            codebook_size=int(self.action_tokenizer.codebook_size),
            configured=self.config.num_action_tokens,
            owner_name=self.__class__.__name__,
        )
        self.action_token_vocabulary = setup_action_token_vocabulary(
            tokenizer=self.processor.tokenizer,
            vlm=self.vlm,
            num_action_tokens=num_tokens,
            token_template=self.config.action_token_template,
            mask_token=self.config.action_mask_token_template,
            init_mode=self.config.action_embed_init,
        )

    @property
    def _supervise_eos(self) -> bool:
        return bool(self.config.ce_supervise_eos)

    def _resolve_eos_token_id(self) -> int | None:
        eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int) and eos_id >= 0:
            return eos_id
        return None

    def _validate_ce_eos_configuration(self) -> None:
        if self._supervise_eos and self._eos_token_id is None:
            raise ValueError(
                f"{self.__class__.__name__} requires tokenizer.eos_token_id for CE EOS supervision."
            )

    def _require_action_token_vocabulary(self) -> ActionTokenVocabulary:
        if self.action_token_vocabulary is None:
            raise RuntimeError("Action token vocabulary has not been initialized.")
        return self.action_token_vocabulary

    @staticmethod
    def _is_vocab_matrix_param_name(name: str) -> bool:
        base_name = name[4:] if name.startswith("vlm.") else name
        is_embedding_parameter = base_name.endswith((".weight", ".embedding", ".new_embedding"))
        if not is_embedding_parameter:
            return False
        return (
            ("embed_tokens" in base_name)
            or ("wte" in base_name)
            or ("lm_head" in base_name)
        )

    def _build_trainable_vocab_rows(self, *, vocab_size: int) -> torch.LongTensor:
        vocab = self._require_action_token_vocabulary()
        return build_trainable_vocab_rows(
            vocab=vocab,
            vocab_size=vocab_size,
            owner_name=self.__class__.__name__,
            supervise_eos=self._supervise_eos,
            eos_token_id=self._eos_token_id,
        )

    def _build_trainable_vocab_rows_for_param(
        self,
        name: str,
        vocab_size: int,
    ) -> torch.LongTensor:
        """Map global action-token ids onto Molmo2 split-embedding rows."""
        if not name.endswith(".new_embedding") and not name.endswith(".embedding"):
            return self._build_trainable_vocab_rows(vocab_size=vocab_size)
        embedding = self.vlm.get_input_embeddings()
        base_embedding = getattr(embedding, "embedding", None)
        new_embedding = getattr(embedding, "new_embedding", None)
        if base_embedding is None or new_embedding is None:
            return self._build_trainable_vocab_rows(vocab_size=vocab_size)
        base_rows = int(base_embedding.shape[0])
        total_rows = base_rows + int(new_embedding.shape[0])
        rows = self._build_trainable_vocab_rows(vocab_size=total_rows)
        if name.endswith(".new_embedding"):
            rows = rows - base_rows
        else:
            rows = rows[rows < base_rows]
        return rows[(rows >= 0) & (rows < int(vocab_size))]

    def _install_vocab_row_gradient_masks(self) -> None:
        install_vocab_row_gradient_masks(
            named_parameters=cast(nn.Module, self.vlm).named_parameters(),
            use_lora=bool(self.config.use_lora),
            is_vocab_matrix_param_name=self._is_vocab_matrix_param_name,
            build_trainable_rows=lambda vocab_size: self._build_trainable_vocab_rows(
                vocab_size=vocab_size
            ),
            build_trainable_rows_for_param=self._build_trainable_vocab_rows_for_param,
            hook_handles=self._vocab_row_grad_mask_hooks,
        )

    def _resolve_action_token_block_schedule(
        self, *, token_horizon: int
    ) -> tuple[int, ...]:
        tokenizer_schedule = getattr(self.action_tokenizer, "block_schedule", None)
        return resolve_block_schedule(tokenizer_schedule, token_horizon=token_horizon)

    def tokenize_actions_to_policy_tokens(self, actions: Tensor) -> list[list[int]]:
        return tokenize_actions(self.action_tokenizer, actions)

    def _prepare_action_token_ce(
        self,
        batch: dict[str, Tensor],
        *,
        actions: Tensor,
    ):
        target_key = self.config.action_token_target_key
        if target_key is None:
            return prepare_action_token_ce(
                self, batch, actions=actions, ignore_index=IGNORE_INDEX
            )
        if target_key not in batch:
            raise KeyError(
                f"{self.__class__.__name__} requires cached token targets in batch[{target_key!r}]."
            )
        targets = batch[target_key]
        if targets.ndim < 2:
            raise ValueError(
                f"Cached action token targets must include batch and token dimensions, got {tuple(targets.shape)}."
            )
        token_ids_batch = targets.reshape(targets.shape[0], -1).long().tolist()
        lengths = None
        length_key = self.config.action_token_length_key
        if length_key is not None:
            if length_key not in batch:
                raise KeyError(
                    f"Missing cached token lengths in batch[{length_key!r}]."
                )
            lengths = batch[length_key].reshape(-1).long().tolist()
        return prepare_action_token_ce_from_token_targets(
            self,
            batch,
            token_ids_batch=token_ids_batch,
            token_lengths=lengths,
            actions=actions,
            ignore_index=IGNORE_INDEX,
        )

    def _effective_loss_weights(self) -> tuple[float, float]:
        ce_weight = float(self.config.ce_loss_weight)
        flow_weight = float(self.config.flow_loss_weight)
        if int(self._training_context.global_step) < int(
            self.config.ce_warmstart_steps
        ):
            flow_weight = 0.0
        return ce_weight, flow_weight

    def _detach_flow_context(self) -> bool:
        end_step = getattr(self.config, "ki_end_step", None)
        return bool(self.config.action_expert_detach_vlm) and (
            end_step is None or int(self._training_context.global_step) < end_step
        )

    def compute_ce_loss(
        self,
        batch: dict[str, Tensor],
        *,
        actions: Tensor | None = None,
    ) -> Tensor:
        """Run the shared sparse VLM CE branch."""
        raw_batch = dict(batch)
        if actions is None:
            actions = self._actions_from_batch(raw_batch)
        prefix_batch = self._observations_for_state_prefix(raw_batch)
        prepared = self._prepare_action_token_ce(prefix_batch, actions=actions)
        ce_result = run_action_token_ce_prefix_forward(
            self,
            prepared,
            prefix_adapter=self._require_vlm_ce_prefix_adapter(),
            ignore_index=IGNORE_INDEX,
            gradient_checkpointing=bool(self.config.gradient_checkpointing),
        )
        return ce_result.loss

    def _build_vlm_inputs_for_action_expert(
        self,
        obs: dict[str, Tensor],
        *,
        instructions: list[str] | None,
    ) -> VlmInputBatch:
        vlm_inputs = self._build_vlm_inputs(
            obs,
            action_token_strings=None,
            instructions=instructions,
        )
        return {key: value for key, value in vlm_inputs.items() if key != "labels"}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        ce_weight, flow_weight = self._effective_loss_weights()
        raw_batch = dict(batch)
        raw_actions = self._actions_from_batch(raw_batch)
        if ce_weight == 0.0 and flow_weight == 0.0:
            zero = raw_actions.new_zeros(())
            if self.training:
                zero = zero.requires_grad_(True)
            return zero, {"ce_loss": 0.0, "flow_loss": 0.0}

        norm_batch: dict[str, Tensor] | None = None
        norm_actions: Tensor | None = None
        if flow_weight > 0.0:
            norm_batch = self._prepare_normalized_batch(raw_batch)
            norm_actions = self._actions_from_batch(norm_batch).to(
                device=raw_actions.device,
                dtype=raw_actions.dtype,
            )

        if (
            ce_weight == 0.0
            and flow_weight > 0.0
            and self._state_prefix_encoding() == "continuous_projection"
        ):
            raise ValueError(
                "KI flow-only training cannot use "
                "state_prefix_encoding='continuous_projection' because the flow "
                "branch detaches the VLM prefix. Use ce_loss_weight > 0 or "
                "state_prefix_encoding='discrete_text'."
            )

        if ce_weight > 0.0 and flow_weight == 0.0:
            prefix_batch = self._observations_for_state_prefix(raw_batch)
            prepared = self._prepare_action_token_ce(prefix_batch, actions=raw_actions)
            ce_result = run_action_token_ce_prefix_forward(
                self,
                prepared,
                prefix_adapter=self._require_vlm_ce_prefix_adapter(),
                ignore_index=IGNORE_INDEX,
                gradient_checkpointing=bool(self.config.gradient_checkpointing),
            )
            ce_loss = ce_result.loss
            total = ce_weight * ce_loss
            return total, {
                "ce_loss": float(ce_loss.detach().item()),
                "flow_loss": 0.0,
            }

        if flow_weight > 0.0 and ce_weight == 0.0:
            if norm_batch is None or norm_actions is None:
                raise AssertionError("flow-normalized batch was not initialized.")
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
                    detach_prefix=self._detach_flow_context(),
                )
            total = flow_weight * flow_loss
            return total, {
                "ce_loss": 0.0,
                "flow_loss": float(flow_loss.detach().item()),
            }

        if norm_batch is None or norm_actions is None:
            raise AssertionError("flow-normalized batch was not initialized.")
        prefix_batch = self._observations_for_state_prefix(
            raw_batch, norm_batch=norm_batch
        )
        prepared = self._prepare_action_token_ce(prefix_batch, actions=raw_actions)
        sample = sample_flow_training_point(
            norm_actions,
            sampling=self.config.flow_time_sampling,
            beta_alpha=self.config.flow_beta_alpha,
            beta_beta=self.config.flow_beta_beta,
            min_time=self.config.flow_min_time,
        )
        schedule_attention_mask = build_prepared_action_token_attention_mask(
            self,
            prepared,
        )
        expert_context_mask = self._build_flow_expert_context_mask(
            prefix_batch,
            prepared,
        )
        model = self._require_vlm_action_expert()
        one_forward_adapter = require_ki_one_forward_adapter(model.prefix_adapter)
        with self._autocast_context():
            flow_loss, integrated_output = model.loss_with_context(
                vlm_inputs=prepared.vlm_inputs,
                layer_attention_mask=schedule_attention_mask,
                expert_context_mask=expert_context_mask,
                sample=sample,
                action_is_pad=self._action_pad_mask_from_batch(
                    norm_batch,
                    actions=norm_actions,
                ),
                detach_expert_context=self._detach_flow_context(),
                gradient_checkpointing=bool(self.config.gradient_checkpointing),
            )
            ce_loss = compute_prepared_action_token_ce_loss_from_hidden(
                self,
                prepared,
                hidden=integrated_output.prefix_state.hidden,
                project_hidden_to_logits=one_forward_adapter.final_hidden_to_logits,
                ignore_index=IGNORE_INDEX,
            )
        total = ce_weight * ce_loss + flow_weight * flow_loss
        return total, {
            "ce_loss": float(ce_loss.detach().item()),
            "flow_loss": float(flow_loss.detach().item()),
        }

    def get_optim_params(self) -> list[dict]:
        return build_vlm_param_groups(
            named_parameters=self.named_parameters(),
            use_lora=bool(self.config.use_lora),
            is_vocab_matrix_param_name=self._is_vocab_matrix_param_name,
        )

    def _build_flow_expert_context_mask(
        self,
        prefix_batch: dict[str, Tensor],
        prepared: Any,
    ) -> Tensor:
        """Return CE-prefix positions that exactly match the FM no-action prompt."""
        instructions = extract_instructions(
            prefix_batch,
            language_key=None if self.io is None else self.io.language_key,
        )
        with self._without_continuous_state_prefix_embeddings():
            fm_inputs = self._build_vlm_inputs_for_action_expert(
                prefix_batch,
                instructions=instructions,
            )
        # The Molmo2 template emits an assistant generation marker for the
        # no-action FM prompt.  Reuse the exact CE prefix and only remove the
        # trailing action supervision span; this avoids tokenizer-dependent
        # whitespace/EOS drift between separately rendered prompts.
        return _context_mask_matching_fm_prefix(
            prepared_vlm_inputs=prepared.vlm_inputs,
            prepared_prefix_mask=prepared.supervision.prefix_mask,
        fm_vlm_inputs=fm_inputs,
        )


def _context_mask_matching_fm_prefix(
    *,
    prepared_vlm_inputs: VlmInputBatch,
    prepared_prefix_mask: Tensor,
    fm_vlm_inputs: VlmInputBatch,
) -> Tensor:
    """Trim CE prefix context to the token span used by the FM prompt."""
    ki_ids = cast(Tensor, prepared_vlm_inputs["input_ids"])
    fm_ids = cast(Tensor, fm_vlm_inputs["input_ids"]).to(device=ki_ids.device)
    prefix_mask = prepared_prefix_mask.to(device=ki_ids.device, dtype=torch.bool)
    fm_attention = cast(Tensor | None, fm_vlm_inputs.get("attention_mask"))
    if fm_attention is None:
        fm_valid_mask = torch.ones_like(fm_ids, dtype=torch.bool)
    else:
        fm_valid_mask = fm_attention.to(device=ki_ids.device, dtype=torch.bool)

    context_mask = torch.zeros_like(prefix_mask, dtype=torch.bool)
    for row_idx in range(int(ki_ids.shape[0])):
        prefix_positions = prefix_mask[row_idx].nonzero(as_tuple=False).flatten()
        fm_tokens = fm_ids[row_idx][fm_valid_mask[row_idx]]
        # Generation prompts may differ from the CE-rendered sequence by a
        # final assistant boundary token.  Match the longest common prefix so
        # the FM context remains exactly aligned with the KI stream.
        if fm_tokens.numel() and prefix_positions.numel():
            max_len = min(int(prefix_positions.numel()), int(fm_tokens.numel()))
            matching = 0
            while matching < max_len and int(ki_ids[row_idx, prefix_positions[matching]]) == int(fm_tokens[matching]):
                matching += 1
            if matching < int(fm_tokens.numel()) and matching >= int(prefix_positions.numel()) - 4:
                fm_tokens = fm_tokens[:matching]
        keep_len = int(fm_tokens.numel())
        if int(prefix_positions.numel()) < keep_len:
            raise RuntimeError(
                "KI CE prefix is shorter than the FM prompt prefix for row "
                f"{row_idx}: ki={int(prefix_positions.numel())}, fm={keep_len}."
            )
        ki_tokens = ki_ids[row_idx][prefix_positions[:keep_len]]
        if not torch.equal(ki_tokens, fm_tokens):
            mismatch = (ki_tokens != fm_tokens).nonzero(as_tuple=False).flatten()
            mismatch_idx = int(mismatch[0]) if mismatch.numel() else -1
            ki_token = int(ki_tokens[mismatch_idx]) if mismatch_idx >= 0 else -1
            fm_token = int(fm_tokens[mismatch_idx]) if mismatch_idx >= 0 else -1
            raise RuntimeError(
                "KI CE prefix does not start with the FM prompt prefix for row "
                f"{row_idx}: ki={int(prefix_positions.numel())}, fm={keep_len}, "
                f"first_mismatch={mismatch_idx}, ki_token={ki_token}, "
                f"fm_token={fm_token}."
            )
        context_mask[row_idx, prefix_positions[:keep_len]] = True
    return context_mask
