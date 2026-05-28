# --------------------------------------------------------------------
# Copyright (C) 2024-2025 EPFL & Apple Inc.
# Licensed under the EPFL-Apple Sample Code License (Non-Commercial)
#
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

from collections.abc import Callable
from functools import partial

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.nn.layers.norm import Fp32RMSNorm

__all__ = ["Attention", "SelfAttention", "CrossAttention"]

_DEFAULT_NORM_LAYER = partial(Fp32RMSNorm, elementwise_affine=False)


class Attention(nn.Module):
    """Shared multi-head attention core."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        proj_bias: bool = False,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        q = self.q_norm(q)
        k = self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)
        x = einops.rearrange(x, "b h n d -> b n (h d)")
        return self.proj_drop(self.proj(x))


class SelfAttention(Attention):
    """Self-attention with fused QKV projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x, block_mask=None, **kwargs):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self._attend(q, k, v, block_mask=block_mask)


class CrossAttention(Attention):
    """Cross-attention with fused KV projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        proj_drop: float = 0.0,
        qk_norm: bool = True,
        norm_layer: Callable[[int], nn.Module] = _DEFAULT_NORM_LAYER,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            proj_bias=proj_bias,
            proj_drop=proj_drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x, context, block_mask=None, **kwargs):
        q = self.q_proj(x)
        k, v = self.kv_proj(context).chunk(2, dim=-1)
        return self._attend(q, k, v, block_mask=block_mask)
