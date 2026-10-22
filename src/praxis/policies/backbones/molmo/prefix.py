# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 VLM prefix stream adapter.

Molmo2's text stack is a Qwen3-sized decoder, but its remote implementation
uses fused QKV projections and ``transformer.blocks`` rather than the usual
``layers``/``self_attn.k_proj`` layout.  This adapter keeps the shared Praxis
prefix contract while matching those native modules exactly.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor
from transformers.masking_utils import create_causal_mask

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
from praxis.policies.core.vlm.schedule import overlay_bidirectional_token_mask


def _apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply Molmo2's (Llama-compatible) rotary embedding convention."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = q.shape[-1] // 2

    def rotate(x: Tensor) -> Tensor:
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    return (q * cos) + (rotate(q) * sin), (k * cos) + (rotate(k) * sin)


def _image_token_mask_function(token_type_ids: Tensor):
    """Return Molmo2's bidirectional image-token mask overlay."""

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> Tensor:
        del head_idx
        safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        kv_is_image = token_type_ids[batch_idx, safe_idx]
        kv_is_image = torch.where(
            kv_idx < token_type_ids.shape[1], kv_is_image, torch.zeros_like(kv_is_image)
        )
        q_is_image = token_type_ids[batch_idx, q_idx]
        return q_is_image & kv_is_image

    return inner_mask


class MolmoPrefixAdapter:
    """Expose Molmo2's causal text prefix stream one layer at a time."""

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
            raise ValueError("Molmo2 prefix inputs must include input_ids.")
        attention_mask = vlm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        input_ids = input_ids.to(device=self.device)
        attention_mask = attention_mask.to(device=self.device)
        inputs_embeds = self._build_input_embeddings(input_ids, vlm_inputs)
        inputs_embeds = replace_state_prefix_placeholder_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            vlm_inputs=vlm_inputs,
        )

        seq_len = int(inputs_embeds.shape[1])
        cache_position = torch.arange(seq_len, device=inputs_embeds.device)
        supplied_position_ids = vlm_inputs.get("position_ids")
        if supplied_position_ids is not None:
            position_ids = supplied_position_ids.to(
                device=inputs_embeds.device, dtype=torch.long
            )
        elif layer_attention_mask is None:
            position_ids = compact_position_ids_from_mask(attention_mask)
        else:
            position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)

        causal_mask = layer_attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask.to(device=inputs_embeds.device)
            causal_mask = overlay_bidirectional_token_mask(
                causal_mask,
                vlm_inputs.get("token_type_ids"),
            )
            suffix_position_offsets = suffix_position_offsets_from_vlm_inputs(
                {"attention_mask": attention_mask, "position_ids": position_ids}
            )
        else:
            token_type_ids = vlm_inputs.get("token_type_ids")
            mask_kwargs: dict[str, Any] = {
                "config": self.text_model.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            if token_type_ids is not None:
                mask_kwargs["or_mask_function"] = _image_token_mask_function(
                    token_type_ids.to(device=inputs_embeds.device, dtype=torch.bool)
                )
            causal_mask = create_causal_mask(**mask_kwargs)
            suffix_position_offsets = compact_suffix_position_offsets(attention_mask)

        position_embeddings = self._position_embeddings(
            inputs_embeds,
            position_ids,
            layer_idx=0,
        )
        return VlmPrefixState(
            hidden=inputs_embeds,
            mask=attention_mask.bool(),
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            suffix_position_offsets=suffix_position_offsets,
            extra={"token_type_ids": vlm_inputs.get("token_type_ids")},
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
        position_embeddings = self._position_embeddings(
            state.hidden,
            state.position_ids,
            layer_idx=layer_idx,
        )
        layer_output = layer(
            state.hidden,
            attention_mask=state.attention_mask,
            position_ids=state.position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=state.cache_position,
            position_embeddings=position_embeddings,
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
            layer_idx=layer_idx,
        )

    def final_hidden_to_logits(
        self,
        hidden: Tensor,
        *,
        candidate_token_ids: Tensor | None = None,
    ) -> Tensor:
        hidden = self.text_model.ln_f(hidden)
        return project_hidden_to_vocab_logits(
            hidden,
            output_embeddings=self.vlm.lm_head,
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
        return self.model.transformer

    @property
    def layers(self) -> nn.ModuleList:
        layers = getattr(self.text_model, "blocks", None)
        if isinstance(layers, nn.ModuleList):
            return layers
        raise RuntimeError("Could not locate Molmo2 text model blocks.")

    def _build_input_embeddings(
        self, input_ids: Tensor, vlm_inputs: VlmInputBatch
    ) -> Tensor:
        model = self.model
        merge_visual_inputs = getattr(model, "merge_visual_inputs", None)
        build_input_embeddings = getattr(model, "build_input_embeddings", None)
        pixel_values = vlm_inputs.get("pixel_values")
        if callable(merge_visual_inputs) and callable(build_input_embeddings):
            images, token_pooling = merge_visual_inputs(
                input_ids=input_ids,
                pixel_values=None
                if pixel_values is None
                else pixel_values.to(device=self.device),
                image_token_pooling=self._get(vlm_inputs, "image_token_pooling"),
                image_grids=self._get(vlm_inputs, "image_grids"),
                image_num_crops=self._get(vlm_inputs, "image_num_crops"),
                pixel_values_videos=self._get(vlm_inputs, "pixel_values_videos"),
                video_token_pooling=self._get(vlm_inputs, "video_token_pooling"),
                video_grids=self._get(vlm_inputs, "video_grids"),
            )
            hidden, _ = build_input_embeddings(input_ids, images, token_pooling)
            return hidden
        # Useful for small test doubles and future Transformers-native Molmo2.
        safe_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        return self.vlm.get_input_embeddings()(safe_ids)

    @staticmethod
    def _get(inputs: VlmInputBatch, key: str) -> Tensor | None:
        value = inputs.get(key)
        return None if value is None else value

    def _position_embeddings(
        self,
        hidden: Tensor,
        position_ids: Tensor,
        *,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor]:
        rotary_embeddings = getattr(self.text_model, "rotary_embs", None)
        rope_layers = getattr(self.text_model.config, "rope_scaling_layers", None)
        if rotary_embeddings is not None and rope_layers is not None:
            name = "scaling" if layer_idx in rope_layers else "default"
            return rotary_embeddings[name](hidden, position_ids)
        return self.text_model.rotary_emb(hidden, position_ids)

    def _project_layer_kv(
        self,
        layer: nn.Module,
        state: VlmPrefixState,
        *,
        position_ids: Tensor | None = None,
        layer_idx: int = 0,
    ) -> LayerKV:
        self_attn = cast(Any, layer).self_attn
        normed = state.hidden
        if not bool(getattr(self.text_model.config, "norm_after", False)):
            normed = cast(Any, layer).attn_norm(normed)
        att_proj = self_attn.att_proj
        normed = normed.to(dtype=att_proj.weight.dtype)
        q_dim, k_dim, _ = tuple(int(dim) for dim in self_attn.fused_dims)
        _, key_flat, value_flat = att_proj(normed).split(
            (q_dim, k_dim, k_dim), dim=-1
        )
        if self_attn.k_norm is not None and self_attn.qk_norm_type != "qwen3":
            key_flat = self_attn.k_norm(key_flat)
        key = key_flat.view(*normed.shape[:-1], -1, self_attn.head_dim)
        if self_attn.k_norm is not None and self_attn.qk_norm_type == "qwen3":
            key = self_attn.k_norm(key)
        key = key.transpose(1, 2)
        value = value_flat.view(*normed.shape[:-1], -1, self_attn.head_dim)
        value = value.transpose(1, 2)
        projected_position_ids = state.position_ids
        if position_ids is not None:
            projected_position_ids = position_ids.to(device=state.hidden.device)
        position_embeddings = self._position_embeddings(
            state.hidden,
            projected_position_ids,
            layer_idx=layer_idx,
        )
        _, key = _apply_rotary_pos_emb(
            key, key, position_embeddings[0], position_embeddings[1]
        )
        mask = state.mask.to(device=key.device, dtype=torch.bool)
        return LayerKV(
            key=key.masked_fill(~mask[:, None, :, None], 0),
            value=value.masked_fill(~mask[:, None, :, None], 0),
        )
