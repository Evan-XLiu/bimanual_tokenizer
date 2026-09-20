# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared-arm encoder, causal cross-attention, and decoder for BQuest."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from praxis.codecs.latents.quest.modeling import ResidualTemporalBlock
from praxis.nn.embeddings.positional import SinusoidPosEmbed


class SharedArmTemporalEncoder(nn.Module):
    """QueST-style causal encoder shared by both arms."""

    def __init__(
        self,
        *,
        arm_dim: int,
        action_horizon: int,
        latent_horizon: int,
        encoder_dim: int,
        encoder_nhead: int,
        encoder_nlayer: int,
        dropout: float,
        use_causal: bool,
    ) -> None:
        super().__init__()
        if action_horizon // latent_horizon != 4:
            raise ValueError(
                "BQuest shared encoder requires action_horizon / latent_horizon = 4."
            )
        self.action_proj = nn.Linear(arm_dim, encoder_dim)
        self.conv_block = ResidualTemporalBlock(
            encoder_dim,
            encoder_dim,
            kernel_size=[5, 3, 3],
            stride=[2, 2, 1],
            causal=use_causal,
        )
        self.pos = SinusoidPosEmbed(encoder_dim, max_shape=[latent_horizon])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_dim,
                nhead=encoder_nhead,
                dim_feedforward=4 * encoder_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=encoder_nlayer,
            enable_nested_tensor=False,
        )
        self.use_causal = bool(use_causal)
        self.latent_horizon = int(latent_horizon)

    def forward(self, arm_actions: Tensor) -> Tensor:
        if arm_actions.ndim != 3 or int(arm_actions.shape[-1]) != self.action_proj.in_features:
            raise ValueError(
                "arm actions must be shaped [B, T, "
                f"{self.action_proj.in_features}], got {tuple(arm_actions.shape)}."
            )
        x = self.action_proj(arm_actions)
        x = self.conv_block(x)
        if int(x.shape[1]) != self.latent_horizon:
            raise RuntimeError(
                "BQuest temporal encoder produced an unexpected horizon: "
                f"expected {self.latent_horizon}, got {int(x.shape[1])}."
            )
        x = x + self.pos([x.shape[1]], device=x.device, dtype=x.dtype)
        if self.use_causal:
            mask = _causal_mask(x.shape[1], x.device)
            return self.encoder(x, mask=mask)
        return self.encoder(x)


class BidirectionalCausalCrossAttention(nn.Module):
    """Compute left-to-right and right-to-left attention from original inputs."""

    def __init__(self, dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.left_queries_right = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True
        )
        self.right_queries_left = nn.MultiheadAttention(
            dim, nhead, dropout=dropout, batch_first=True
        )
        self.left_norm = nn.LayerNorm(dim)
        self.right_norm = nn.LayerNorm(dim)

    def forward(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        if left.shape != right.shape:
            raise ValueError(
                "left and right arm features must have the same shape, got "
                f"{tuple(left.shape)} and {tuple(right.shape)}."
            )
        mask = _causal_mask(left.shape[1], left.device)
        # Keep both calls anchored to the original H_L/H_R tensors.
        left_update, _ = self.left_queries_right(
            left, right, right, attn_mask=mask, need_weights=False
        )
        right_update, _ = self.right_queries_left(
            right, left, left, attn_mask=mask, need_weights=False
        )
        return self.left_norm(left + left_update), self.right_norm(right + right_update)


class BQuestDecoder(nn.Module):
    """One shared causal decoder for coarse and refined latent trajectories."""

    def __init__(
        self,
        *,
        action_dim: int,
        action_horizon: int,
        latent_horizon: int,
        latent_dim: int,
        decoder_dim: int,
        decoder_nhead: int,
        decoder_nlayer: int,
        dropout: float,
        use_causal: bool,
    ) -> None:
        super().__init__()
        self.post_quant = nn.Linear(latent_dim, decoder_dim)
        self.pos = SinusoidPosEmbed(decoder_dim, max_shape=[action_horizon])
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=decoder_dim,
                nhead=decoder_nhead,
                dim_feedforward=4 * decoder_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=decoder_nlayer,
        )
        self.action_head = nn.Linear(decoder_dim, action_dim)
        self.action_horizon = int(action_horizon)
        self.latent_horizon = int(latent_horizon)
        self.use_causal = bool(use_causal)

    def forward(self, latents: Tensor) -> Tensor:
        if latents.ndim != 3 or int(latents.shape[1]) != self.latent_horizon:
            raise ValueError(
                "BQuest decoder expects latents shaped "
                f"[B, {self.latent_horizon}, D], got {tuple(latents.shape)}."
            )
        memory = self.post_quant(latents)
        query = torch.zeros(
            (memory.shape[0], self.action_horizon, memory.shape[-1]),
            dtype=memory.dtype,
            device=memory.device,
        )
        query = query + self.pos(
            [self.action_horizon], device=query.device, dtype=query.dtype
        )
        mask = (
            _causal_mask(self.action_horizon, query.device)
            if self.use_causal
            else None
        )
        decoded = self.decoder(query, memory, tgt_mask=mask)
        return self.action_head(decoded)


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones((int(length), int(length)), dtype=torch.bool, device=device),
        diagonal=1,
    )


def init_bquest_weights(module: nn.Module) -> None:
    """Use the same simple Xavier initialization family as QueST."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.xavier_uniform_(child.weight)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)
