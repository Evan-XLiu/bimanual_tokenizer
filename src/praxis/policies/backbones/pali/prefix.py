# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""PaliGemma VLM prefix stream adapter."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb

from praxis.policies.core.state_prefix import (
    replace_state_prefix_placeholder_embeddings,
)
from praxis.policies.core.vlm.logits import project_hidden_to_vocab_logits
from praxis.policies.core.vlm.prefix import (
    LayerKV,
    VlmAttentionGeometry,
    VlmInputBatch,
    VlmPrefixState,
    attention_geometry_from_config,
    compact_position_ids_from_mask,
    compact_suffix_position_offsets,
    suffix_position_offsets_from_vlm_inputs,
)


class PaliGemmaPrefixAdapter:
    """Expose PaliGemma's text prefix stream one layer at a time."""

    def __init__(self, vlm: nn.Module) -> None:
        self.vlm = vlm

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.language_model.config.hidden_size)

    @property
    def attention_geometry(self) -> VlmAttentionGeometry:
        return attention_geometry_from_config(self.language_model.config)

    def prepare_prefix(
        self,
        vlm_inputs: VlmInputBatch,
        *,
        layer_attention_mask: Tensor | None = None,
    ) -> VlmPrefixState:
        input_ids = vlm_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("PaliGemma prefix inputs must include input_ids.")
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        pixel_values = vlm_inputs.get("pixel_values")

        input_ids = input_ids.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        inputs_embeds = self._input_embeddings(input_ids)
        if pixel_values is not None:
            inputs_embeds = self._merge_image_features(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values.to(device=self.device),
            )
        inputs_embeds = replace_state_prefix_placeholder_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            vlm_inputs=vlm_inputs,
        )
        token_type_ids = vlm_inputs.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device=self.device)

        seq_len = int(inputs_embeds.shape[1])
        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        expert_position_ids = compact_position_ids_from_mask(
            attention_mask,
            first_position=1,
        )
        causal_mask = layer_attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask.to(device=inputs_embeds.device)
            position_ids = (
                cache_position.unsqueeze(0).expand(input_ids.shape[0], -1) + 1
            )
            suffix_position_offsets = suffix_position_offsets_from_vlm_inputs(
                {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
            )
        else:
            position_ids = expert_position_ids
            suffix_position_offsets = compact_suffix_position_offsets(
                attention_mask,
                first_position=1,
            )
            causal_mask = cast(Any, self.vlm).model._update_causal_mask(
                attention_mask,
                token_type_ids,
                None,
                cache_position,
                inputs_embeds,
                False,
            )

        language_model = self.language_model
        position_embeddings = language_model.rotary_emb(inputs_embeds, position_ids)
        hidden = inputs_embeds * torch.tensor(
            language_model.config.hidden_size**0.5,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        return VlmPrefixState(
            hidden=hidden,
            mask=attention_mask.bool(),
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            suffix_position_offsets=suffix_position_offsets,
            extra={"expert_position_ids": expert_position_ids},
        )

    def forward_layer(
        self, state: VlmPrefixState, *, layer_idx: int
    ) -> tuple[VlmPrefixState, LayerKV]:
        layer_kv = self.project_layer_kv(state, layer_idx=layer_idx)
        return self.forward_prefix_layer(state, layer_idx=layer_idx), layer_kv

    def forward_prefix_layer(
        self, state: VlmPrefixState, *, layer_idx: int
    ) -> VlmPrefixState:
        layer = self.layers[layer_idx]
        layer_output = layer(
            state.hidden,
            attention_mask=self._attention_mask_for_layer(layer, state),
            position_ids=state.position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=state.cache_position,
            position_embeddings=state.position_embeddings,
        )
        hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return VlmPrefixState(
            hidden=hidden,
            mask=state.mask,
            attention_mask=state.attention_mask,
            position_ids=state.position_ids,
            cache_position=state.cache_position,
            position_embeddings=state.position_embeddings,
            suffix_position_offsets=state.suffix_position_offsets,
            extra=state.extra,
        )

    def project_layer_kv(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
        position_ids: Tensor | None = None,
    ) -> LayerKV:
        return self._project_layer_kv(
            self.layers[layer_idx],
            state,
            position_ids=position_ids,
        )

    def final_hidden_to_logits(
        self,
        hidden: Tensor,
        *,
        candidate_token_ids: Tensor | None = None,
    ) -> Tensor:
        hidden = self._final_norm(hidden)
        return project_hidden_to_vocab_logits(
            hidden,
            output_embeddings=self._output_embeddings(),
            input_embeddings=cast(Any, self.vlm).get_input_embeddings(),
            candidate_token_ids=candidate_token_ids,
        )

    @property
    def device(self) -> torch.device:
        return next(cast(nn.Module, self.vlm).parameters()).device

    @property
    def language_model(self) -> Any:
        return cast(Any, self.vlm).language_model

    @property
    def layers(self) -> nn.ModuleList:
        layers = getattr(self.language_model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        model = getattr(self.language_model, "model", None)
        layers = getattr(model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        raise RuntimeError("Could not locate PaliGemma language model layers.")

    def _final_norm(self, hidden: Tensor) -> Tensor:
        norm = getattr(self.language_model, "norm", None)
        if norm is None:
            model = getattr(self.language_model, "model", None)
            norm = getattr(model, "norm", None) if model is not None else None
        if norm is None:
            return hidden
        return cast(Any, norm)(hidden)

    def _output_embeddings(self) -> nn.Module | None:
        for owner in (self.vlm, self.language_model):
            getter = getattr(owner, "get_output_embeddings", None)
            if callable(getter):
                output_embeddings = getter()
                if output_embeddings is not None:
                    return cast(nn.Module, output_embeddings)
            lm_head = getattr(owner, "lm_head", None)
            if isinstance(lm_head, nn.Module):
                return lm_head
        return None

    def _input_embeddings(self, input_ids: Tensor) -> Tensor:
        config = cast(Any, self.vlm).config
        model = cast(Any, self.vlm).model
        image_token_id = int(config.image_token_id)
        vocab_size = int(getattr(model, "vocab_size", config.text_config.vocab_size))
        llm_input_ids = input_ids
        if image_token_id >= vocab_size:
            special_image_mask = input_ids == image_token_id
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        return cast(Any, self.vlm).get_input_embeddings()(llm_input_ids)

    def _merge_image_features(
        self,
        *,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        pixel_values: Tensor,
    ) -> Tensor:
        image_features = cast(Any, self.vlm).get_image_features(pixel_values)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        placeholder_mask = cast(Any, self.vlm).model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        return inputs_embeds.masked_scatter(placeholder_mask, image_features)

    def _project_layer_kv(
        self,
        layer: nn.Module,
        state: VlmPrefixState,
        *,
        position_ids: Tensor | None = None,
    ) -> LayerKV:
        self_attn = cast(Any, layer).self_attn
        normed = cast(Any, layer).input_layernorm(state.hidden)
        input_shape = normed.shape[:-1]
        key = self_attn.k_proj(normed).view(*input_shape, -1, self_attn.head_dim)
        value = self_attn.v_proj(normed).view(*input_shape, -1, self_attn.head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        position_embeddings = state.position_embeddings
        if position_ids is not None:
            position_embeddings = self.language_model.rotary_emb(
                state.hidden,
                position_ids.to(device=state.hidden.device),
            )
        _, key = apply_rotary_pos_emb(
            key,
            key,
            position_embeddings[0],
            position_embeddings[1],
        )
        mask = state.mask.to(device=key.device, dtype=torch.bool)
        return LayerKV(
            key=key.masked_fill(~mask[:, None, :, None], 0),
            value=value.masked_fill(~mask[:, None, :, None], 0),
        )

    @staticmethod
    def _attention_mask_for_layer(
        layer: nn.Module, state: VlmPrefixState
    ) -> Tensor | None:
        attention_mask = state.attention_mask
        if isinstance(attention_mask, dict):
            attention_type = getattr(layer, "attention_type", "full_attention")
            return attention_mask[attention_type]
        return attention_mask
