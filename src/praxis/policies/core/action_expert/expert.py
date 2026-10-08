# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""SmolVLA-style flow-matching action expert conditioned on VLM K/V."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.nn.blocks.mlp import GatedMlp
from praxis.nn.embeddings.positional import RotaryPosEmbed
from praxis.nn.embeddings.time import sinusoidal_scalar_features
from praxis.nn.layers.norm import Fp32RMSNorm
from praxis.policies.core.action_expert.flow import FlowSample
from praxis.policies.core.action_expert.prefix_cache import (
    PrefixCache,
    cache_to_device,
    flatten_cache_tensor,
    validate_cache_matches_mask,
)
from praxis.policies.core.vlm.prefix import LayerKV, VlmAttentionGeometry

ExpertAttentionKind = Literal["self_attn", "cross_attn"]
SWIGLU_FFN_MULTIPLIER = 4


class ActionExpert(nn.Module):
    """SmolVLA-style suffix stream conditioned on VLM prefix K/V."""

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        vlm_num_layers: int,
        expert_num_layers: int | None,
        vlm_attention: VlmAttentionGeometry,
        hidden_size: int,
        dropout: float,
        time_min_period: float,
        time_max_period: float,
        self_attn_every_n_layers: int,
        architecture: str = "smolvla",
        num_attention_heads: int | None = None,
        mlp_ratio: float = 4.0,
        ffn_multiple_of: int = 256,
        qk_norm: bool = True,
        causal_attn: bool = False,
        context_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if time_min_period <= 0.0 or time_max_period <= 0.0:
            raise ValueError(
                "time_min_period and time_max_period must be positive, got "
                f"{time_min_period} and {time_max_period}."
            )
        if time_min_period > time_max_period:
            raise ValueError(
                "time_min_period must be <= time_max_period, got "
                f"{time_min_period} and {time_max_period}."
            )
        self.action_dim = int(action_dim)
        self.architecture = str(architecture)
        if self.architecture not in {"smolvla", "molmoact2"}:
            raise ValueError(f"Unsupported action-expert architecture {architecture!r}.")
        self.action_horizon = int(action_horizon)
        self.hidden_size = int(hidden_size)
        self.vlm_num_layers = int(vlm_num_layers)
        if self.vlm_num_layers <= 0:
            raise ValueError(f"vlm_num_layers must be positive, got {vlm_num_layers}.")
        self.expert_num_layers = int(
            self.vlm_num_layers if expert_num_layers is None else expert_num_layers
        )
        if self.expert_num_layers <= 0:
            raise ValueError(
                f"expert_num_layers must be positive, got {expert_num_layers}."
            )
        if self.expert_num_layers > self.vlm_num_layers:
            raise ValueError(
                "expert_num_layers must be <= vlm_num_layers, got "
                f"{self.expert_num_layers} and {self.vlm_num_layers}."
            )
        if self.vlm_num_layers % self.expert_num_layers != 0:
            raise ValueError(
                "vlm_num_layers must be divisible by expert_num_layers, got "
                f"{self.vlm_num_layers} and {self.expert_num_layers}."
            )
        self.layer_stride = self.vlm_num_layers // self.expert_num_layers
        self.num_layers = self.expert_num_layers
        self.self_attn_every_n_layers = int(self_attn_every_n_layers)
        if self.self_attn_every_n_layers == 0 or self.self_attn_every_n_layers < -1:
            raise ValueError(
                "self_attn_every_n_layers must be -1 or a positive integer, got "
                f"{self.self_attn_every_n_layers}."
            )
        self.num_attention_heads = int(
            num_attention_heads
            if num_attention_heads is not None
            else vlm_attention.num_attention_heads
        )
        self.num_key_value_heads = (
            self.num_attention_heads if self.architecture == "molmoact2"
            else int(vlm_attention.num_key_value_heads)
        )
        self.head_dim = (
            self.hidden_size // self.num_attention_heads
            if self.architecture == "molmoact2"
            else int(vlm_attention.head_dim)
        )
        if self.architecture == "molmoact2" and self.hidden_size % self.num_attention_heads:
            raise ValueError("MolmoAct2 hidden_size must be divisible by num_attention_heads.")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError(
                "VLM attention head counts must be positive, got "
                f"{self.num_attention_heads} and {self.num_key_value_heads}."
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "VLM num_attention_heads must be divisible by num_key_value_heads, "
                f"got {self.num_attention_heads} and {self.num_key_value_heads}."
            )
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError(
                f"VLM attention head_dim must be positive and even, got {self.head_dim}."
            )
        self.attention_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.vlm_kv_dim = int(vlm_attention.num_key_value_heads * vlm_attention.head_dim)
        self.time_min_period = float(time_min_period)
        self.time_max_period = float(time_max_period)

        self.action_proj = nn.Linear(self.action_dim, self.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.action_time_mlp_out = nn.Linear(self.hidden_size, self.hidden_size)
        layer_cls = MolmoAct2ActionExpertLayer if self.architecture == "molmoact2" else ActionExpertLayer
        self.layers = nn.ModuleList(
            [
                layer_cls(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    attention_kind=self._attention_kind(expert_idx * self.layer_stride),
                    dropout=float(dropout),
                    vlm_kv_dim=self.vlm_kv_dim,
                    mlp_ratio=float(mlp_ratio),
                    ffn_multiple_of=int(ffn_multiple_of),
                    qk_norm=bool(qk_norm),
                    causal_attn=bool(causal_attn),
                    context_layer_norm=bool(context_layer_norm),
                )
                for expert_idx in range(self.expert_num_layers)
            ]
        )
        self.norm = Fp32RMSNorm(self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.action_dim)

    def _action_time_tokens(self, noisy_actions: Tensor, timesteps: Tensor) -> Tensor:
        action_tokens = self.action_proj(noisy_actions)
        time_tokens = sinusoidal_scalar_features(
            timesteps,
            self.hidden_size,
            min_frequency=1.0 / self.time_max_period,
            max_frequency=1.0 / self.time_min_period,
            angular_scale=2.0 * math.pi,
            endpoint=True,
            descending=True,
            channel_order="sin_cos",
        )
        time_tokens = time_tokens[:, None, :].expand_as(action_tokens)
        action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
        return self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(action_time_tokens))
        )

    def forward(
        self,
        *,
        cache: PrefixCache,
        noisy_actions: Tensor,
        timesteps: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> Tensor:
        if noisy_actions.ndim != 3:
            raise ValueError(
                "noisy_actions must have shape (B, H, D), got "
                f"{tuple(noisy_actions.shape)}."
            )
        batch_size, horizon, action_dim = noisy_actions.shape
        if horizon != self.action_horizon or action_dim != self.action_dim:
            raise ValueError(
                "noisy_actions shape mismatch: expected "
                f"(B, {self.action_horizon}, {self.action_dim}), got "
                f"{tuple(noisy_actions.shape)}."
            )
        if timesteps.shape != (batch_size,):
            raise ValueError(
                f"timesteps must have shape ({batch_size},), got {tuple(timesteps.shape)}."
            )

        validate_cache_matches_mask(cache)
        device = noisy_actions.device
        cache = cache_to_device(cache, device=device)
        hidden, expert_pad_mask, expert_attention_mask = self.prepare_suffix_inputs(
            noisy_actions=noisy_actions,
            timesteps=timesteps,
            action_is_pad=action_is_pad,
        )

        if len(cache.layers) != self.vlm_num_layers:
            raise ValueError(
                "Prefix cache layer count must match VLM layer count, "
                f"got cache={len(cache.layers)} vlm={self.vlm_num_layers}."
            )
        for vlm_layer_idx, layer_kv in enumerate(cache.layers):
            hidden = self.forward_suffix_layer(
                hidden=hidden,
                vlm_layer_idx=vlm_layer_idx,
                layer_kv=layer_kv,
                prefix_mask=cache.mask,
                prefix_position_offsets=cache.suffix_position_offsets,
                expert_pad_mask=expert_pad_mask,
                expert_attention_mask=expert_attention_mask,
            )
        return self.velocity_from_suffix_hidden(hidden)

    def prepare_suffix_inputs(
        self,
        *,
        noisy_actions: Tensor,
        timesteps: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Build expert suffix tokens plus the masks shared by all expert layers."""
        if noisy_actions.ndim != 3:
            raise ValueError(
                "noisy_actions must have shape (B, H, D), got "
                f"{tuple(noisy_actions.shape)}."
            )
        batch_size, horizon, action_dim = noisy_actions.shape
        if horizon != self.action_horizon or action_dim != self.action_dim:
            raise ValueError(
                "noisy_actions shape mismatch: expected "
                f"(B, {self.action_horizon}, {self.action_dim}), got "
                f"{tuple(noisy_actions.shape)}."
            )
        if timesteps.shape != (batch_size,):
            raise ValueError(
                f"timesteps must have shape ({batch_size},), got {tuple(timesteps.shape)}."
            )

        device = noisy_actions.device
        expert_pad_mask = self._expert_pad_mask(
            action_is_pad=action_is_pad,
            batch_size=batch_size,
            horizon=horizon,
            device=device,
        )
        expert_attention_mask = self._expert_attention_mask(
            batch_size=batch_size,
            seq_len=int(expert_pad_mask.shape[1]),
            device=device,
        )

        param_dtype = self.action_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=param_dtype)
        hidden = self._action_time_tokens(
            noisy_actions,
            timesteps.to(device=device, dtype=param_dtype),
        )
        return hidden, expert_pad_mask, expert_attention_mask

    def forward_suffix_layer(
        self,
        *,
        hidden: Tensor,
        vlm_layer_idx: int,
        layer_kv: LayerKV,
        prefix_mask: Tensor,
        prefix_position_offsets: Tensor | None,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
    ) -> Tensor:
        """Advance the expert stream at the SmolVLA layer paired with a VLM layer."""
        expert_layer_idx = self._expert_layer_index(vlm_layer_idx)
        if expert_layer_idx is None:
            return hidden
        attention_kind = self._attention_kind(vlm_layer_idx)
        expert_positions = self._expert_positions(
            expert_pad_mask=expert_pad_mask,
            prefix_mask=prefix_mask,
            prefix_position_offsets=prefix_position_offsets,
            offset_after_prefix=attention_kind == "self_attn",
        )
        layer = self.layers[expert_layer_idx]
        if self.architecture == "molmoact2":
            return layer(
                hidden=hidden,
                layer_kv=layer_kv,
                prefix_mask=prefix_mask,
                expert_pad_mask=expert_pad_mask,
                expert_attention_mask=expert_attention_mask,
                expert_positions=expert_positions,
            )
        return layer(
            hidden=hidden,
            layer_kv=layer_kv,
            prefix_mask=prefix_mask,
            expert_pad_mask=expert_pad_mask,
            expert_attention_mask=expert_attention_mask,
            expert_positions=expert_positions,
        )

    def velocity_from_suffix_hidden(self, hidden: Tensor) -> Tensor:
        hidden = self.norm(hidden)
        action_hidden = hidden[:, -self.action_horizon :]
        return self.out_proj(action_hidden)

    def _expert_layer_index(self, vlm_layer_idx: int) -> int | None:
        if (
            self.layer_stride > 1
            and vlm_layer_idx > 0
            and vlm_layer_idx % self.layer_stride != 0
        ):
            return None
        expert_layer_idx = vlm_layer_idx // self.layer_stride
        if expert_layer_idx >= self.expert_num_layers:
            return None
        return expert_layer_idx

    def _attention_kind(self, vlm_layer_idx: int) -> ExpertAttentionKind:
        if (
            self.self_attn_every_n_layers > 0
            and vlm_layer_idx % self.self_attn_every_n_layers == 0
        ):
            return "self_attn"
        return "cross_attn"

    @staticmethod
    def _expert_positions(
        *,
        expert_pad_mask: Tensor,
        prefix_mask: Tensor,
        prefix_position_offsets: Tensor | None = None,
        offset_after_prefix: bool,
    ) -> Tensor:
        positions = torch.cumsum(expert_pad_mask.to(dtype=torch.long), dim=1) - 1
        positions = positions.clamp_min(0)
        if offset_after_prefix:
            if prefix_position_offsets is None:
                prefix_offsets = prefix_mask.to(device=positions.device).sum(
                    dim=1,
                    keepdim=True,
                )
            else:
                prefix_offsets = prefix_position_offsets.to(
                    device=positions.device,
                    dtype=positions.dtype,
                )
                if prefix_offsets.ndim == 1:
                    prefix_offsets = prefix_offsets[:, None]
                if prefix_offsets.shape != (positions.shape[0], 1):
                    raise ValueError(
                        "prefix_position_offsets must have shape (B,) or (B, 1), got "
                        f"{tuple(prefix_position_offsets.shape)}."
                    )
            positions = positions + prefix_offsets.to(dtype=positions.dtype)
        else:
            positions = positions - positions.min(dim=1, keepdim=True).values
        return positions

    def _expert_pad_mask(
        self,
        *,
        action_is_pad: Tensor | None,
        batch_size: int,
        horizon: int,
        device: torch.device,
    ) -> Tensor:
        """Return the token-stream mask for action-expert inputs.

        ``action_is_pad`` is a loss mask, not a token-stream mask. Diffusion
        context still renders the full noised action horizon so positional and
        attention geometry stays identical across samples in the batch.
        """
        if action_is_pad is not None and action_is_pad.shape != (
            int(batch_size),
            int(horizon),
        ):
            raise ValueError(
                "action_is_pad must have shape (B, H), got "
                f"{tuple(action_is_pad.shape)} for B={batch_size}, H={horizon}."
            )
        action_mask = torch.ones(
            (int(batch_size), int(horizon)),
            dtype=torch.bool,
            device=device,
        )
        return action_mask

    def _expert_attention_mask(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        attention_mask = torch.ones(
            (int(batch_size), int(seq_len)),
            dtype=torch.bool,
            device=device,
        )
        return attention_mask

    def loss(
        self,
        *,
        cache: PrefixCache,
        sample: FlowSample,
        action_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pred_velocity = self(
            cache=cache,
            noisy_actions=sample.noisy_actions,
            timesteps=sample.timesteps,
            action_is_pad=action_is_pad,
        )
        loss = masked_action_mse_loss(
            pred_velocity,
            sample.target_velocity,
            action_is_pad=action_is_pad,
        )
        return loss, pred_velocity


class ActionExpertLayer(nn.Module):
    """One SmolVLA expert block using either self-attn or cross-attn."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_kind: ExpertAttentionKind,
        dropout: float,
        vlm_kv_dim: int | None = None,
        mlp_ratio: float = 4.0,
        ffn_multiple_of: int = 256,
        qk_norm: bool = False,
        causal_attn: bool = True,
        context_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        del vlm_kv_dim, mlp_ratio, ffn_multiple_of, qk_norm, causal_attn, context_layer_norm
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.attention_kind: ExpertAttentionKind = attention_kind
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim

        self.input_layernorm = Fp32RMSNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.attention_dim)
        if attention_kind == "self_attn":
            self.k_proj = nn.Linear(self.hidden_size, self.kv_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.kv_dim)
        elif attention_kind == "cross_attn":
            self.k_proj = nn.Linear(self.kv_dim, self.kv_dim)
            self.v_proj = nn.Linear(self.kv_dim, self.kv_dim)
        else:
            raise ValueError(f"Unsupported expert attention kind {attention_kind!r}.")
        self.o_proj = nn.Linear(self.attention_dim, self.hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.post_attention_layernorm = Fp32RMSNorm(self.hidden_size)
        self.mlp = GatedMlp(
            in_features=self.hidden_size,
            hidden_features=SWIGLU_FFN_MULTIPLIER * self.hidden_size,
            act_layer=nn.SiLU,
            drop=float(dropout),
        )
        self.rope = RotaryPosEmbed(self.head_dim, pairing="split")

    def forward(
        self,
        *,
        hidden: Tensor,
        layer_kv: LayerKV,
        prefix_mask: Tensor,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
        expert_positions: Tensor,
    ) -> Tensor:
        if expert_pad_mask.shape != hidden.shape[:2]:
            raise ValueError(
                "expert_pad_mask must have shape (B, S) matching hidden tokens, "
                f"got {tuple(expert_pad_mask.shape)} and hidden={tuple(hidden.shape)}."
            )
        if expert_attention_mask.shape != hidden.shape[:2]:
            raise ValueError(
                "expert_attention_mask must have shape (B, S) matching hidden tokens, "
                f"got {tuple(expert_attention_mask.shape)} and hidden={tuple(hidden.shape)}."
            )
        if expert_positions.shape != hidden.shape[:2]:
            raise ValueError(
                "expert_positions must have shape (B, S) matching hidden tokens, "
                f"got {tuple(expert_positions.shape)} and hidden={tuple(hidden.shape)}."
            )

        residual = hidden
        normed = self.input_layernorm(hidden).to(dtype=self.q_proj.weight.dtype)
        query = self.rope(
            self._to_query_heads(self.q_proj(normed)),
            positions=expert_positions,
        )

        if self.attention_kind == "self_attn":
            key = torch.cat(
                [
                    self._prefix_cache_to_kv_heads(layer_kv.key, device=hidden.device),
                    self.rope(
                        self._to_key_value_heads(self.k_proj(normed)),
                        positions=expert_positions,
                    ),
                ],
                dim=2,
            )
            value = torch.cat(
                [
                    self._prefix_cache_to_kv_heads(
                        layer_kv.value,
                        device=hidden.device,
                    ),
                    self._to_key_value_heads(self.v_proj(normed)),
                ],
                dim=2,
            )
            attn_mask = self._self_attention_mask(
                prefix_mask=prefix_mask,
                expert_pad_mask=expert_pad_mask,
                expert_attention_mask=expert_attention_mask,
            )
        else:
            prefix_key = flatten_cache_tensor(layer_kv.key).to(
                device=hidden.device,
                dtype=self.k_proj.weight.dtype,
            )
            prefix_value = flatten_cache_tensor(layer_kv.value).to(
                device=hidden.device,
                dtype=self.v_proj.weight.dtype,
            )
            self._validate_prefix_flat_dim(prefix_key)
            self._validate_prefix_flat_dim(prefix_value)
            key = self._to_key_value_heads(self.k_proj(prefix_key))
            value = self._to_key_value_heads(self.v_proj(prefix_value))
            attn_mask = self._cross_attention_mask(
                prefix_mask=prefix_mask,
                query_len=int(hidden.shape[1]),
            )

        key = self._expand_key_value_heads(key)
        value = self._expand_key_value_heads(value)
        attended = F.scaled_dot_product_attention(
            query,
            key.to(dtype=query.dtype),
            value.to(dtype=query.dtype),
            attn_mask=attn_mask,
            dropout_p=0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            hidden.shape[0], hidden.shape[1], self.attention_dim
        )
        hidden = residual + self.dropout(
            self.o_proj(attended.to(dtype=self.o_proj.weight.dtype))
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class MolmoAct2ActionExpertLayer(nn.Module):
    """MolmoAct2 block: full non-causal self-attention followed by VLM cross-attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_kind: ExpertAttentionKind,
        dropout: float,
        vlm_kv_dim: int,
        mlp_ratio: float,
        ffn_multiple_of: int,
        qk_norm: bool,
        causal_attn: bool,
        context_layer_norm: bool,
    ) -> None:
        super().__init__()
        del attention_kind
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.attention_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim
        self.vlm_kv_dim = int(vlm_kv_dim)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.causal_attn = bool(causal_attn)
        self.context_layer_norm_enabled = bool(context_layer_norm)

        self.input_layernorm = Fp32RMSNorm(self.hidden_size)
        self.self_q_proj = nn.Linear(self.hidden_size, self.attention_dim)
        self.self_k_proj = nn.Linear(self.hidden_size, self.kv_dim)
        self.self_v_proj = nn.Linear(self.hidden_size, self.kv_dim)
        self.self_o_proj = nn.Linear(self.attention_dim, self.hidden_size)
        self.context_layer_norm = Fp32RMSNorm(self.vlm_kv_dim) if context_layer_norm else nn.Identity()
        self.cross_q_proj = nn.Linear(self.hidden_size, self.attention_dim)
        self.cross_k_proj = nn.Linear(self.vlm_kv_dim, self.kv_dim)
        self.cross_v_proj = nn.Linear(self.vlm_kv_dim, self.kv_dim)
        self.cross_o_proj = nn.Linear(self.attention_dim, self.hidden_size)
        self.q_norm = Fp32RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = Fp32RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.rope = RotaryPosEmbed(self.head_dim, pairing="split")
        inner = int(math.ceil(float(self.hidden_size) * float(mlp_ratio) / int(ffn_multiple_of)) * int(ffn_multiple_of))
        self.post_attention_layernorm = Fp32RMSNorm(self.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, inner),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner, self.hidden_size),
            nn.Dropout(float(dropout)),
        )
        self.dropout = nn.Dropout(float(dropout))

    def _heads(self, x: Tensor, heads: int) -> Tensor:
        b, s, _ = x.shape
        return x.view(b, s, heads, self.head_dim).transpose(1, 2)

    def _expand(self, x: Tensor) -> Tensor:
        return x if self.num_key_value_groups == 1 else x.repeat_interleave(self.num_key_value_groups, dim=1)

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None) -> Tensor:
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask, dropout_p=0.0, is_causal=self.causal_attn and mask is None)
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], self.attention_dim)

    def forward(
        self,
        *,
        hidden: Tensor,
        layer_kv: LayerKV,
        prefix_mask: Tensor,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
        expert_positions: Tensor,
    ) -> Tensor:
        del expert_attention_mask
        normed = self.input_layernorm(hidden).to(dtype=self.self_q_proj.weight.dtype)
        q = self.rope(self.q_norm(self._heads(self.self_q_proj(normed), self.num_attention_heads)), positions=expert_positions)
        k = self.rope(self.k_norm(self._heads(self.self_k_proj(normed), self.num_key_value_heads)), positions=expert_positions)
        v = self._heads(self.self_v_proj(normed), self.num_key_value_heads)
        suffix_mask = expert_pad_mask[:, None, None, :].expand(hidden.shape[0], 1, hidden.shape[1], hidden.shape[1])
        hidden = hidden + self.dropout(self.self_o_proj(self._attend(q, self._expand(k), self._expand(v), suffix_mask)))

        normed = self.input_layernorm(hidden).to(dtype=self.cross_q_proj.weight.dtype)
        q = self.rope(self.q_norm(self._heads(self.cross_q_proj(normed), self.num_attention_heads)), positions=expert_positions)
        prefix_k = flatten_cache_tensor(layer_kv.key).to(device=hidden.device, dtype=self.cross_k_proj.weight.dtype)
        prefix_v = flatten_cache_tensor(layer_kv.value).to(device=hidden.device, dtype=self.cross_v_proj.weight.dtype)
        if prefix_k.shape[-1] != self.vlm_kv_dim or prefix_v.shape[-1] != self.vlm_kv_dim:
            raise ValueError(f"MolmoAct2 VLM KV dim must be {self.vlm_kv_dim}, got {prefix_k.shape[-1]} and {prefix_v.shape[-1]}.")
        prefix_k = self.context_layer_norm(prefix_k)
        prefix_v = self.context_layer_norm(prefix_v)
        k = self.k_norm(self._heads(self.cross_k_proj(prefix_k), self.num_key_value_heads))
        v = self._heads(self.cross_v_proj(prefix_v), self.num_key_value_heads)
        cross_mask = prefix_mask.to(device=hidden.device, dtype=torch.bool)[:, None, None, :]
        hidden = hidden + self.dropout(self.cross_o_proj(self._attend(q, self._expand(k), self._expand(v), cross_mask)))
        return hidden + self.mlp(self.post_attention_layernorm(hidden))

    def _to_query_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _to_key_value_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _prefix_cache_to_kv_heads(
        self, tensor: Tensor, *, device: torch.device
    ) -> Tensor:
        flat = flatten_cache_tensor(tensor).to(
            device=device, dtype=self.q_proj.weight.dtype
        )
        self._validate_prefix_flat_dim(flat)
        return self._to_key_value_heads(flat)

    def _validate_prefix_flat_dim(self, tensor: Tensor) -> None:
        if int(tensor.shape[-1]) != self.kv_dim:
            raise ValueError(
                "Flattened VLM cache dim must match VLM K/V geometry, got "
                f"{int(tensor.shape[-1])} and expected {self.kv_dim}."
            )

    def _expand_key_value_heads(self, tensor: Tensor) -> Tensor:
        if self.num_key_value_groups == 1:
            return tensor
        return tensor.repeat_interleave(self.num_key_value_groups, dim=1)

    def _self_attention_mask(
        self,
        *,
        prefix_mask: Tensor,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
    ) -> Tensor:
        batch_size, query_len = expert_pad_mask.shape
        prefix = prefix_mask.to(device=expert_pad_mask.device, dtype=torch.bool)[
            :, None, :
        ].expand(batch_size, query_len, -1)
        suffix = _make_att_2d_masks(expert_pad_mask, expert_attention_mask)
        return torch.cat([prefix, suffix], dim=2)[:, None, :, :]

    @staticmethod
    def _cross_attention_mask(*, prefix_mask: Tensor, query_len: int) -> Tensor:
        batch_size = int(prefix_mask.shape[0])
        return prefix_mask.to(dtype=torch.bool)[:, None, None, :].expand(
            batch_size,
            1,
            int(query_len),
            int(prefix_mask.shape[1]),
        )


def _make_att_2d_masks(pad_mask: Tensor, attention_mask: Tensor) -> Tensor:
    if pad_mask.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError(
            "pad_mask and attention_mask must both have shape (B, S), got "
            f"{tuple(pad_mask.shape)} and {tuple(attention_mask.shape)}."
        )
    block_ids = torch.cumsum(attention_mask.to(dtype=torch.long), dim=1)
    att_2d = block_ids[:, None, :] <= block_ids[:, :, None]
    pad_2d = pad_mask[:, None, :].bool() & pad_mask[:, :, None].bool()
    return att_2d & pad_2d


def masked_action_mse_loss(
    pred: Tensor,
    target: Tensor,
    *,
    action_is_pad: Tensor | None,
) -> Tensor:
    """Reduce velocity MSE over non-padded action steps only."""
    per_value = F.mse_loss(pred, target, reduction="none")
    if action_is_pad is None:
        return per_value.mean()
    if action_is_pad.shape != pred.shape[:2]:
        raise ValueError(
            "action_is_pad must have shape (B, H) matching predicted actions, "
            f"got {tuple(action_is_pad.shape)} for pred={tuple(pred.shape)}."
        )
    valid = ~action_is_pad.to(device=pred.device, dtype=torch.bool)
    valid_values = valid[:, :, None].expand_as(per_value)
    denom = valid_values.sum()
    if int(denom.item()) == 0:
        return per_value.sum() * 0.0
    return per_value.masked_select(valid_values).sum() / denom.to(dtype=per_value.dtype)
