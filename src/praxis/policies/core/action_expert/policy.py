"""Shared continuous-action lane for VLM/action-expert policies."""

from __future__ import annotations

import contextlib
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from praxis.core.checkpoint_artifacts import collect_normalizer_stats
from praxis.core.normalizer import Normalize, Unnormalize
from praxis.core.types import NormMode, PolicyIO
from praxis.policies.core.action_expert.expert import ActionExpert
from praxis.policies.core.action_expert.flow import sample_flow_training_point
from praxis.policies.core.action_expert.integrated import (
    VlmActionExpertModel,
    VlmActionExpertPrefixAdapter,
)
from praxis.policies.core.action_expert.prefix_cache import (
    build_prefix_cache,
)
from praxis.policies.core.action_expert.sampling import sample_actions_euler
from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.state_prefix import (
    STATE_PREFIX_EMBEDDINGS_KEY,
    StatePrefixPolicyMixin,
)
from praxis.policies.core.vlm.dtypes import parse_vlm_dtype
from praxis.policies.core.vlm.fsdp import select_vlm_fsdp_wrap_modules
from praxis.policies.core.vlm.lora import (
    apply_lora_to_vlm,
)
from praxis.policies.core.vlm.prefix import (
    VlmInputBatch,
    suffix_position_offsets_from_vlm_inputs,
)


class ActionExpertPolicyMixin(StatePrefixPolicyMixin):
    """Shared state/action/expert utilities for VLM-conditioned action experts."""

    config: Any
    vlm: Any
    action_expert: ActionExpert | None
    vlm_action_expert: VlmActionExpertModel | None
    io: PolicyIO | None
    normalize: Normalize | None
    unnormalize_action: Unnormalize | None

    def _initialize_action_expert_members(self) -> None:
        self._initialize_state_prefix_members()
        self.action_expert = None
        self.vlm_action_expert = None

    def _load_vlm(self, model_name: str, dtype_str: str) -> None:
        raise NotImplementedError

    def _freeze_vision_encoder_modules(self) -> None:
        """Backbone hook for freezing vision modules."""

    def _ensure_frozen_vision_eval(self) -> None:
        """Backbone hook for keeping frozen vision modules in eval mode."""

    def _load_configured_vlm_runtime(self) -> None:
        self._load_vlm(self.config.vlm_model_name, self.config.vlm_dtype)
        if self.config.freeze_vision_encoder:
            self._freeze_vision_encoder_modules()
        self._setup_state_prefix_token()

    def _apply_configured_lora(self, *, require_vocab_modules: bool) -> None:
        if not self.config.use_lora:
            return
        self.vlm = apply_lora_to_vlm(
            vlm=self.vlm,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            lora_modules_to_save=self.config.lora_modules_to_save,
            require_vocab_modules=require_vocab_modules,
        )

    def _enable_configured_gradient_checkpointing(self) -> None:
        if self.config.gradient_checkpointing:
            cast(Any, self.vlm).gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

    def refresh_after_config_update(self, updated_keys: frozenset[str]) -> None:
        if "n_action_steps" in updated_keys:
            cast(Any, self)._configure_action_queue(
                n_action_steps=self.config.n_action_steps
            )

    def _build_vlm_prefix_adapter(self) -> VlmActionExpertPrefixAdapter:
        raise NotImplementedError

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        instructions: list[str] | None = None,
    ) -> VlmInputBatch:
        raise NotImplementedError

    def _build_action_expert(self, io: PolicyIO) -> None:
        prefix_adapter = self._build_vlm_prefix_adapter()
        self._build_state_prefix_projector(hidden_size=prefix_adapter.hidden_size)
        hidden_size = self._resolve_expert_hidden_size(prefix_adapter)
        self.action_expert = ActionExpert(
            action_dim=int(io.action_dim),
            action_horizon=int(self.config.action_horizon),
            vlm_num_layers=prefix_adapter.num_layers,
            expert_num_layers=self.config.expert_num_layers,
            vlm_attention=prefix_adapter.attention_geometry,
            hidden_size=hidden_size,
            dropout=float(self.config.expert_dropout),
            time_min_period=float(self.config.expert_time_min_period),
            time_max_period=float(self.config.expert_time_max_period),
            self_attn_every_n_layers=int(self.config.expert_self_attn_every_n_layers),
        )
        self.vlm_action_expert = VlmActionExpertModel(
            prefix_adapter=prefix_adapter,
            action_expert=self.action_expert,
        )

    def _resolve_expert_hidden_size(
        self,
        prefix_adapter: VlmActionExpertPrefixAdapter,
    ) -> int:
        configured = getattr(self.config, "expert_hidden_size", None)
        if configured is not None:
            return int(configured)
        multiplier = float(getattr(self.config, "expert_width_multiplier", 0.75))
        return max(1, int(prefix_adapter.hidden_size * multiplier))

    def _require_action_expert(self) -> ActionExpert:
        if self.action_expert is None:
            raise RuntimeError("Action expert is not initialized. Call set_io(...).")
        return self.action_expert

    def _require_vlm_action_expert(self) -> VlmActionExpertModel:
        if self.vlm_action_expert is None:
            raise RuntimeError(
                "Integrated VLM/action-expert model is not initialized. "
                "The policy backbone must implement _build_vlm_prefix_adapter(...)."
            )
        return self.vlm_action_expert

    def _action_key(self) -> str:
        if self.io is not None and self.io.action_feature is not None:
            return self.io.action_feature.key
        return "action"

    def _actions_from_batch(self, batch: dict[str, Tensor]) -> Tensor:
        action_key = self._action_key()
        if action_key not in batch:
            raise KeyError(f"Training batch is missing action key {action_key!r}.")
        actions = batch[action_key]
        if actions.ndim != 3:
            raise ValueError(
                f"Expected action tensor with shape (B, H, D), got {tuple(actions.shape)}."
            )
        return actions

    def _action_pad_mask_from_batch(
        self,
        batch: dict[str, Tensor],
        *,
        actions: Tensor,
    ) -> Tensor | None:
        pad_key = f"{self._action_key()}_is_pad"
        value = batch.get(pad_key)
        if value is None:
            return None
        if value.ndim != 2:
            raise ValueError(
                f"{pad_key} must have shape (B, H), got {tuple(value.shape)}."
            )
        if value.shape != actions.shape[:2]:
            raise ValueError(
                f"{pad_key} shape must match action batch/horizon "
                f"{tuple(actions.shape[:2])}, got {tuple(value.shape)}."
            )
        return value.to(device=actions.device, dtype=torch.bool)

    def _prepare_normalized_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.normalize is None:
            return dict(batch)
        return self.normalize(dict(batch))

    def compute_integrated_flow_loss(
        self,
        batch: dict[str, Tensor],
        *,
        actions: Tensor,
        vlm_inputs: VlmInputBatch,
        detach_prefix: bool,
    ) -> Tensor:
        sample = sample_flow_training_point(
            actions,
            sampling=self.config.flow_time_sampling,
            beta_alpha=self.config.flow_beta_alpha,
            beta_beta=self.config.flow_beta_beta,
            min_time=self.config.flow_min_time,
        )
        loss, _pred = self._require_vlm_action_expert().loss(
            vlm_inputs=vlm_inputs,
            sample=sample,
            action_is_pad=self._action_pad_mask_from_batch(batch, actions=actions),
            detach_prefix=detach_prefix,
            gradient_checkpointing=bool(self.config.gradient_checkpointing),
        )
        return loss

    def fsdp_wrap_modules(self) -> tuple[nn.Module, ...]:
        modules = list(
            select_vlm_fsdp_wrap_modules(
                cast(nn.Module, self.vlm),
                use_lora=bool(self.config.use_lora),
            )
        )
        if self.action_expert is not None:
            modules.append(self.action_expert)
        if self.state_prefix_projector is not None:
            modules.append(self.state_prefix_projector)
        return tuple(modules)

    def fsdp_forward_methods(self) -> tuple[str, ...]:
        return ("predict_action", "predict_action_chunk")

    def supports_training_strategy(self, strategy: str) -> bool:
        return strategy in {"ddp", "fsdp"}

    def validate_distributed_strategy(self, strategy: str) -> None:
        if not self.supports_training_strategy(strategy):
            raise ValueError(
                f"{self.__class__.__name__} does not support training strategy {strategy!r}."
            )
        if strategy == "fsdp" and not self.fsdp_wrap_modules():
            raise ValueError(
                f"{self.__class__.__name__} FSDP requires at least one parameterized "
                "VLM or expert block."
            )

    def train(self, mode: bool = True):
        nn.Module.train(cast(nn.Module, self), mode)
        if self.config.freeze_vision_encoder:
            self._ensure_frozen_vision_eval()
        return self

    def _autocast_context(self):
        device_type = next(cast(nn.Module, self.vlm).parameters()).device.type
        dtype = parse_vlm_dtype(str(getattr(self.config, "vlm_dtype", "bfloat16")))
        if device_type in ("cuda", "cpu") and dtype is not torch.float32:
            return torch.autocast(device_type=device_type, dtype=dtype)
        return contextlib.nullcontext()

    def _build_vlm_inputs_for_action_expert(
        self,
        obs: dict[str, Tensor],
        *,
        instructions: list[str] | None,
    ) -> VlmInputBatch:
        vlm_inputs = self._build_vlm_inputs(
            obs,
            instructions=instructions,
        )
        return {key: value for key, value in vlm_inputs.items() if key != "labels"}

    def _prepare_vlm_prefill_inputs_for_action_expert(
        self,
        vlm_inputs: VlmInputBatch,
    ) -> VlmInputBatch:
        return vlm_inputs

    @torch.inference_mode()
    def predict_action(self, obs: dict[str, Tensor], **kwargs) -> Tensor:
        policy = cast(Any, self)
        return policy._predict_action_from_chunk_queue(
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

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        obs: dict[str, Tensor],
        *,
        normalize_obs: bool = True,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[Tensor, dict[str, Any]]:
        del kwargs
        raw_obs = dict(obs)
        norm_obs = self._prepare_normalized_batch(raw_obs) if normalize_obs else raw_obs
        prefix_obs = self._observations_for_state_prefix(raw_obs, norm_batch=norm_obs)
        vlm_inputs = self._build_vlm_inputs_for_action_expert(
            prefix_obs,
            instructions=extract_instructions(
                raw_obs,
                language_key=None if self.io is None else self.io.language_key,
            ),
        )
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(vlm_inputs["input_ids"], dtype=torch.bool)
        if STATE_PREFIX_EMBEDDINGS_KEY in vlm_inputs:
            cache = self._require_vlm_action_expert().prefill_cache(
                vlm_inputs,
                detach=True,
            )
        else:
            vlm_inputs = self._prepare_vlm_prefill_inputs_for_action_expert(vlm_inputs)
            outputs = self.vlm(**vlm_inputs, use_cache=True, return_dict=True)
            cache = build_prefix_cache(
                past_key_values=getattr(outputs, "past_key_values", None),
                prefix_mask=attention_mask.bool(),
                detach=True,
                suffix_position_offsets=suffix_position_offsets_from_vlm_inputs(
                    vlm_inputs
                ),
            )
        if self.io is None:
            raise RuntimeError("PolicyIO must be set before action prediction.")
        actions = sample_actions_euler(
            self._require_action_expert(),
            cache=cache,
            action_dim=int(
                self.config.action_dim
                if self.config.action_dim is not None
                else self.io.action_dim
            ),
            action_horizon=int(self.config.action_horizon),
            steps=int(self.config.flow_sampling_steps),
            generator=generator,
        )
        if self.unnormalize_action is not None:
            action_key = self._action_key()
            actions = self.unnormalize_action({action_key: actions})[action_key]
        return actions, {"flow_sampling_steps": int(self.config.flow_sampling_steps)}

    def set_normalization_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        modes: dict[str, NormMode] | None = None,
    ) -> None:
        self.normalize = None
        self.unnormalize_action = None
        if self.io is None:
            return

        policy = cast(Any, self)
        relevant_stats, resolved_modes = policy._collect_observation_norm_stats(
            stats,
            feature_modes=modes,
            features=self.io.state_features,
            use_image_defaults=False,
            missing_stats_warning=(
                "Missing normalization stats for %s; skipping state normalization for this key."
            ),
        )
        action_stats, action_mode = policy._collect_action_norm_stats(
            stats=stats,
            feature_modes=modes,
        )
        if (
            self.io.action_feature is not None
            and action_stats is not None
            and action_mode is not None
        ):
            action_key = self.io.action_feature.key
            relevant_stats[action_key] = action_stats
            resolved_modes[action_key] = action_mode
            self.unnormalize_action = Unnormalize(
                {action_key: action_stats},
                {action_key: action_mode},
            )
        self.normalize = (
            Normalize(relevant_stats, resolved_modes) if resolved_modes else None
        )

    def _export_norm_stats_for_checkpoint(
        self,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, str]]:
        stats: dict[str, dict[str, Tensor]] = {}
        modes: dict[str, str] = {}
        collect_normalizer_stats(self.normalize, stats, modes)
        collect_normalizer_stats(self.unnormalize_action, stats, modes)
        return stats, modes
