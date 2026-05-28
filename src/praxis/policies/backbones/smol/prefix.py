"""SmolVLM VLM prefix stream adapter."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

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


class SmolPrefixAdapter:
    """Expose SmolVLM's causal text prefix stream one layer at a time."""

    def __init__(self, vlm: nn.Module) -> None:
        self.vlm = vlm

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def hidden_size(self) -> int:
        return int(self.text_model.config.hidden_size)

    @property
    def attention_geometry(self) -> VlmAttentionGeometry:
        return attention_geometry_from_config(self.text_model.config)

    def prepare_prefix(
        self,
        vlm_inputs: VlmInputBatch,
        *,
        layer_attention_mask: Tensor | None = None,
    ) -> VlmPrefixState:
        input_ids = vlm_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("SmolVLM prefix inputs must include input_ids.")
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        input_ids = input_ids.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        inputs_embeds = self._merge_image_features(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=vlm_inputs.get("pixel_values"),
            pixel_attention_mask=vlm_inputs.get("pixel_attention_mask"),
            image_hidden_states=vlm_inputs.get("image_hidden_states"),
        )
        inputs_embeds = replace_state_prefix_placeholder_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            vlm_inputs=vlm_inputs,
        )

        seq_len = int(inputs_embeds.shape[1])
        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        expert_position_ids = compact_position_ids_from_mask(
            attention_mask,
            first_position=0,
        )
        causal_mask = layer_attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask.to(device=inputs_embeds.device)
            position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
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
                first_position=0,
            )
            causal_mask = create_causal_mask(
                config=self.text_model.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=position_ids,
            )
        position_embeddings = self.text_model.rotary_emb(inputs_embeds, position_ids)
        return VlmPrefixState(
            hidden=inputs_embeds,
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
            attention_mask=state.attention_mask,
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
            input_embeddings=self.text_model.get_input_embeddings(),
            candidate_token_ids=candidate_token_ids,
        )

    @property
    def device(self) -> torch.device:
        return next(cast(nn.Module, self.vlm).parameters()).device

    @property
    def model(self) -> Any:
        return cast(Any, self.vlm).model

    @property
    def text_model(self) -> Any:
        return self.model.text_model

    @property
    def layers(self) -> nn.ModuleList:
        layers = getattr(self.text_model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        raise RuntimeError("Could not locate SmolVLM text model layers.")

    def _final_norm(self, hidden: Tensor) -> Tensor:
        norm = getattr(self.text_model, "norm", None)
        if norm is None:
            return hidden
        return cast(Any, norm)(hidden)

    def _output_embeddings(self) -> nn.Module | None:
        for owner in (self.vlm, self.text_model):
            getter = getattr(owner, "get_output_embeddings", None)
            if callable(getter):
                output_embeddings = getter()
                if output_embeddings is not None:
                    return cast(nn.Module, output_embeddings)
            lm_head = getattr(owner, "lm_head", None)
            if isinstance(lm_head, nn.Module):
                return lm_head
        return None

    def _merge_image_features(
        self,
        *,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        pixel_values: Tensor | None,
        pixel_attention_mask: Tensor | None,
        image_hidden_states: Tensor | None,
    ) -> Tensor:
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError(
                "SmolVLM prefix inputs cannot include both pixel_values and "
                "image_hidden_states."
            )
        if pixel_values is not None:
            image_hidden_states = self.model.get_image_features(
                pixel_values.to(device=self.device),
                None
                if pixel_attention_mask is None
                else pixel_attention_mask.to(device=self.device),
            ).to(inputs_embeds.device)
        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(
                device=inputs_embeds.device,
                dtype=cast(Any, self.vlm).dtype,
            )
        if image_hidden_states is None:
            return inputs_embeds
        return self.model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_hidden_states,
        )

    def _project_layer_kv(
        self,
        layer: nn.Module,
        state: VlmPrefixState,
        *,
        position_ids: Tensor | None = None,
    ) -> LayerKV:
        self_attn = cast(Any, layer).self_attn
        normed = cast(Any, layer).input_layernorm(state.hidden)
        normed = normed.to(dtype=self_attn.k_proj.weight.dtype)
        input_shape = normed.shape[:-1]
        hidden_shape = (*input_shape, -1, self_attn.head_dim)
        key = self_attn.k_proj(normed).view(hidden_shape).transpose(1, 2)
        value = self_attn.v_proj(normed).view(hidden_shape).transpose(1, 2)
        position_embeddings = state.position_embeddings
        if position_ids is not None:
            position_embeddings = self.text_model.rotary_emb(
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
