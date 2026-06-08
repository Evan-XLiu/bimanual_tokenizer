# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL VLM prefix stream adapter."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

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
    suffix_position_offsets_from_vlm_inputs,
)


class QwenPrefixAdapter:
    """Expose Qwen3-VL's causal text prefix stream one layer at a time."""

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
            raise ValueError("Qwen prefix inputs must include input_ids.")
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        input_ids = input_ids.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = (
            self._merge_image_features(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                pixel_values=vlm_inputs.get("pixel_values"),
                image_grid_thw=vlm_inputs.get("image_grid_thw"),
            )
        )
        inputs_embeds = replace_state_prefix_placeholder_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            vlm_inputs=vlm_inputs,
        )

        cache_position = torch.arange(
            int(inputs_embeds.shape[1]),
            device=inputs_embeds.device,
        )
        position_ids = self._position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=vlm_inputs.get("image_grid_thw"),
        )
        text_position_ids = self._text_position_ids(position_ids)
        causal_mask = layer_attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask.to(device=inputs_embeds.device)
        else:
            causal_mask = create_causal_mask(
                config=self.language_model.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=text_position_ids,
            )
        position_embeddings = self.language_model.rotary_emb(
            inputs_embeds,
            position_ids,
        )
        suffix_position_offsets = suffix_position_offsets_from_vlm_inputs(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        )
        return VlmPrefixState(
            hidden=inputs_embeds,
            mask=attention_mask.bool(),
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            suffix_position_offsets=suffix_position_offsets,
            extra={
                "expert_position_ids": position_ids,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
            },
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
        hidden = self._apply_deepstack(hidden, state=state, layer_idx=layer_idx)
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
            input_embeddings=self.model.get_input_embeddings(),
            candidate_token_ids=candidate_token_ids,
        )

    @property
    def device(self) -> torch.device:
        return next(cast(nn.Module, self.vlm).parameters()).device

    @property
    def model(self) -> Any:
        return cast(Any, self.vlm).model

    @property
    def language_model(self) -> Any:
        return self.model.language_model

    @property
    def layers(self) -> nn.ModuleList:
        layers = getattr(self.language_model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        raise RuntimeError("Could not locate Qwen3-VL text model layers.")

    def _final_norm(self, hidden: Tensor) -> Tensor:
        norm = getattr(self.language_model, "norm", None)
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

    def _merge_image_features(
        self,
        *,
        input_ids: Tensor,
        inputs_embeds: Tensor,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, list[Tensor] | None]:
        if pixel_values is None:
            return inputs_embeds, None, None
        if image_grid_thw is None:
            raise ValueError("Qwen image prefix inputs must include image_grid_thw.")

        image_embeds, deepstack_image_embeds = self.model.get_image_features(
            pixel_values.to(device=self.device),
            image_grid_thw.to(device=self.device),
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device,
            inputs_embeds.dtype,
        )
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds, image_mask[..., 0], deepstack_image_embeds

    def _position_ids(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        image_grid_thw: Tensor | None,
    ) -> Tensor:
        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=None
            if image_grid_thw is None
            else image_grid_thw.to(device=input_ids.device),
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        self.model.rope_deltas = rope_deltas
        return position_ids

    @staticmethod
    def _text_position_ids(position_ids: Tensor) -> Tensor:
        if position_ids.ndim == 3 and int(position_ids.shape[0]) == 4:
            return position_ids[0]
        if position_ids.ndim == 3:
            return position_ids[0]
        if position_ids.ndim == 2:
            return position_ids
        raise ValueError(
            f"Expected Qwen position_ids rank 2 or 3, got {tuple(position_ids.shape)}."
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
        key = self_attn.k_norm(self_attn.k_proj(normed).view(hidden_shape))
        value = self_attn.v_proj(normed).view(hidden_shape)
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

    def _apply_deepstack(
        self,
        hidden: Tensor,
        *,
        state: VlmPrefixState,
        layer_idx: int,
    ) -> Tensor:
        extra = state.extra or {}
        visual_pos_masks = extra.get("visual_pos_masks")
        deepstack_visual_embeds = extra.get("deepstack_visual_embeds")
        if (
            visual_pos_masks is None
            or deepstack_visual_embeds is None
            or layer_idx >= len(deepstack_visual_embeds)
        ):
            return hidden
        return self.language_model._deepstack_process(
            hidden,
            cast(Tensor, visual_pos_masks),
            cast(list[Tensor], deepstack_visual_embeds)[layer_idx],
        )


def with_qwen_position_ids(
    vlm: nn.Module,
    vlm_inputs: VlmInputBatch,
) -> VlmInputBatch:
    """Add native Qwen3-VL M-RoPE position ids for cache-prefill inputs."""
    input_ids = cast(Tensor | None, vlm_inputs.get("input_ids"))
    if input_ids is None:
        raise ValueError("Qwen prefill inputs must include input_ids.")
    attention_mask = cast(Tensor | None, vlm_inputs.get("attention_mask"))
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    qwen_model = cast(Any, vlm).model
    position_ids, rope_deltas = qwen_model.get_rope_index(
        input_ids=input_ids,
        image_grid_thw=vlm_inputs.get("image_grid_thw"),
        video_grid_thw=vlm_inputs.get("video_grid_thw"),
        attention_mask=attention_mask,
    )
    qwen_model.rope_deltas = rope_deltas
    out = dict(vlm_inputs)
    out["position_ids"] = position_ids
    return out
