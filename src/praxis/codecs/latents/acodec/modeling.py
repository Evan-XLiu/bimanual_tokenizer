"""ActionCodec-family Perceiver encoder/decoder blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from praxis.nn.embeddings.positional import SinusoidPosEmbed

__all__ = ["ActionCodecDecoder", "ActionCodecEncoder"]


class _L2Norm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_val: float = 1e-2) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), float(init_val)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _PerceiverAttention(nn.Module):
    """Official ActionCodec-style attention for Perceiver cross/self blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_norm: str = "ln",
        bias: bool = False,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads

        self.q_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.out_proj = nn.Linear(self.dim, self.dim, bias=bias)

        if qk_norm == "ln":
            self.q_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
            self.k_norm = nn.LayerNorm(self.head_dim, elementwise_affine=False)
        elif qk_norm == "l2":
            self.q_norm = _L2Norm()
            self.k_norm = _L2Norm()
        elif qk_norm == "none":
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported qk_norm={qk_norm!r}.")

        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch, length, self.dim)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        context = x if context is None else context
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(context))
        value = self._split_heads(self.v_proj(context))

        query = self.q_norm(query)
        key = self.k_norm(key)

        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        return self.out_proj(self._merge_heads(out))


class _FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        mlp_ratio: float = 4.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(float(mlp_ratio) * dim)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.up_proj(x)))


class _PerceiverBlock(nn.Module):
    """Perceiver block matching the official ActionCodec block shape."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        qk_norm: str = "ln",
        layer_scale: bool = True,
        add_self_attn: bool = False,
        add_causal_mask: bool = False,
    ) -> None:
        super().__init__()
        self.add_causal_mask = bool(add_causal_mask)
        self.norm1 = nn.LayerNorm(dim, eps=1e-2)
        self.cross_attn = _PerceiverAttention(
            dim=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            bias=False,
        )

        self.self_attn: _PerceiverAttention | None = None
        if add_self_attn:
            self.norm_self_attn = nn.LayerNorm(dim, eps=1e-2)
            self.self_attn = _PerceiverAttention(
                dim=dim,
                num_heads=num_heads,
                qk_norm=qk_norm,
                bias=False,
            )

        self.norm2 = nn.LayerNorm(dim, eps=1e-2)
        self.mlp = _FeedForward(dim=dim, mlp_ratio=mlp_ratio, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_scale = _LayerScale(dim) if layer_scale else nn.Identity()
        self.mlp_scale = _LayerScale(dim) if layer_scale else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.cross_attn(x=x, context=context, mask=context_mask)
        x = residual + self.attn_scale(self.dropout(x))

        if self.self_attn is not None:
            residual = x
            x = self.norm_self_attn(x)
            x = self.self_attn(x=x, is_causal=self.add_causal_mask)
            x = residual + self.attn_scale(self.dropout(x))

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        return residual + self.mlp_scale(self.dropout(x))


class _PerceiverStack(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        qk_norm: str,
        layer_scale: bool,
        add_self_attn: bool,
        add_causal_mask: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _PerceiverBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    layer_scale=layer_scale,
                    add_self_attn=add_self_attn,
                    add_causal_mask=add_causal_mask,
                )
                for _ in range(depth)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, context=context, context_mask=context_mask)
        return x


def _num_heads_from_head_dim(emb_dim: int, head_dim: int) -> int:
    if emb_dim % head_dim != 0:
        raise ValueError(f"emb_dim={emb_dim} must be divisible by head_dim={head_dim}.")
    return emb_dim // head_dim


def _init_trunc_normal_linear(linear: nn.Linear) -> None:
    nn.init.trunc_normal_(linear.weight, std=0.02)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _scaled_pos(
    pos_embed: SinusoidPosEmbed,
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale: float,
) -> torch.Tensor:
    return pos_embed([length], device=device, dtype=dtype) * scale


class ActionCodecEncoder(nn.Module):
    """Perceiver encoder for ActionCodec continuous latents."""

    def __init__(
        self,
        sample_dim: int,
        sample_horizon: int,
        latent_horizon: int,
        emb_dim: int,
        latent_dim: int,
        head_dim: int,
        depth: int,
        dropout: float,
        *,
        mlp_ratio: float = 4.0,
        qk_norm: str = "ln",
        layer_scale: bool = True,
        pos_scale: float = 0.1,
        add_self_attn: bool = False,
        add_causal_mask: bool = False,
    ) -> None:
        super().__init__()
        self.sample_horizon = int(sample_horizon)
        self.latent_horizon = int(latent_horizon)
        self.pos_scale = float(pos_scale)

        self.input_proj = nn.Linear(sample_dim, emb_dim)
        self.sample_pos = SinusoidPosEmbed(emb_dim, max_shape=[self.sample_horizon])

        self.query_tokens = nn.Parameter(torch.empty(self.latent_horizon, emb_dim))
        self.query_pos = SinusoidPosEmbed(emb_dim, max_shape=[self.latent_horizon])

        self.transformer = _PerceiverStack(
            dim=emb_dim,
            depth=depth,
            num_heads=_num_heads_from_head_dim(emb_dim, head_dim),
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            qk_norm=qk_norm,
            layer_scale=layer_scale,
            add_self_attn=add_self_attn,
            add_causal_mask=add_causal_mask,
        )
        self.output_proj = nn.Linear(emb_dim, latent_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_trunc_normal_linear(self.input_proj)
        _init_trunc_normal_linear(self.output_proj)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(
                f"actions must be rank-3 [B, T, D], got shape={tuple(actions.shape)}."
            )
        if int(actions.shape[1]) != self.sample_horizon:
            raise ValueError(
                f"action horizon mismatch: expected {self.sample_horizon}, got {int(actions.shape[1])}."
            )

        batch = int(actions.shape[0])
        context = self.input_proj(actions)
        context = context + _scaled_pos(
            self.sample_pos,
            self.sample_horizon,
            device=context.device,
            dtype=context.dtype,
            scale=self.pos_scale,
        )

        queries = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        queries = queries + _scaled_pos(
            self.query_pos,
            self.latent_horizon,
            device=queries.device,
            dtype=queries.dtype,
            scale=self.pos_scale,
        )

        hidden = self.transformer(x=queries, context=context)
        return self.output_proj(hidden)


class ActionCodecDecoder(nn.Module):
    """Perceiver decoder for ActionCodec continuous latents."""

    def __init__(
        self,
        sample_dim: int,
        sample_horizon: int,
        latent_horizon: int,
        emb_dim: int,
        latent_dim: int,
        head_dim: int,
        depth: int,
        dropout: float,
        *,
        mlp_ratio: float = 4.0,
        qk_norm: str = "ln",
        layer_scale: bool = True,
        pos_scale: float = 0.1,
        add_self_attn: bool = False,
        add_causal_mask: bool = False,
    ) -> None:
        super().__init__()
        self.sample_horizon = int(sample_horizon)
        self.latent_horizon = int(latent_horizon)
        self.pos_scale = float(pos_scale)

        self.memory_proj = nn.Linear(latent_dim, emb_dim)
        self.memory_pos = SinusoidPosEmbed(emb_dim, max_shape=[self.latent_horizon])

        self.query_tokens = nn.Parameter(torch.empty(1, emb_dim))
        self.query_pos = SinusoidPosEmbed(emb_dim, max_shape=[self.sample_horizon])

        self.transformer = _PerceiverStack(
            dim=emb_dim,
            depth=depth,
            num_heads=_num_heads_from_head_dim(emb_dim, head_dim),
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            qk_norm=qk_norm,
            layer_scale=layer_scale,
            add_self_attn=add_self_attn,
            add_causal_mask=add_causal_mask,
        )
        self.output_proj = nn.Linear(emb_dim, sample_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_trunc_normal_linear(self.memory_proj)
        _init_trunc_normal_linear(self.output_proj)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError(
                f"latents must be rank-3 [B, L, D], got shape={tuple(latents.shape)}."
            )
        if int(latents.shape[1]) != self.latent_horizon:
            raise ValueError(
                f"latent horizon mismatch: expected {self.latent_horizon}, got {int(latents.shape[1])}."
            )

        batch = int(latents.shape[0])
        memory = self.memory_proj(latents)
        memory = memory + _scaled_pos(
            self.memory_pos,
            self.latent_horizon,
            device=memory.device,
            dtype=memory.dtype,
            scale=self.pos_scale,
        )

        queries = self.query_tokens.unsqueeze(0).expand(batch, self.sample_horizon, -1)
        queries = queries + _scaled_pos(
            self.query_pos,
            self.sample_horizon,
            device=queries.device,
            dtype=queries.dtype,
            scale=self.pos_scale,
        )

        hidden = self.transformer(x=queries, context=memory)
        return self.output_proj(hidden)
