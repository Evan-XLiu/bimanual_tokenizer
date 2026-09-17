# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Query-based decoder for timestep-paired BTT arm latents."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from praxis.nn.embeddings.positional import SinusoidPosEmbed


class TimestepPairedBTTDecoder(nn.Module):
    """Decode aligned left/right FSQ embeddings into joint bimanual actions."""

    def __init__(
        self,
        *,
        latent_dim: int,
        latent_horizon: int,
        model_dim: int,
        action_dim: int,
        action_horizon: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        use_causal: bool,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.latent_horizon = int(latent_horizon)
        self.model_dim = int(model_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.use_causal = bool(use_causal)

        # Each memory slot represents concat(L_t, R_t), never a merged token ID.
        self.post_quant = nn.Linear(2 * self.latent_dim, self.model_dim)
        self.decoder_pos = SinusoidPosEmbed(
            self.model_dim, max_shape=[self.action_horizon]
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.model_dim,
                nhead=int(num_heads),
                dim_feedforward=4 * self.model_dim,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(num_layers),
        )
        self.action_head = nn.Linear(self.model_dim, self.action_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Match the Linear/LayerNorm initialization used by QueST."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def pair_latents(self, left_latents: Tensor, right_latents: Tensor) -> Tensor:
        """Return `[L_t, R_t]` feature pairs at each matching timestep."""

        expected = (self.latent_horizon, self.latent_dim)
        for name, latents in (
            ("left_latents", left_latents),
            ("right_latents", right_latents),
        ):
            if latents.ndim != 3 or tuple(latents.shape[1:]) != expected:
                raise ValueError(
                    f"{name} must have shape [B, {expected[0]}, {expected[1]}], "
                    f"got {tuple(latents.shape)}."
                )
        if left_latents.shape[0] != right_latents.shape[0]:
            raise ValueError("left and right latent batch sizes must match.")
        return torch.cat((left_latents, right_latents), dim=-1)

    def build_memory(self, left_latents: Tensor, right_latents: Tensor) -> Tensor:
        """Project eight aligned bimanual pairs into decoder memory."""

        return self.post_quant(self.pair_latents(left_latents, right_latents))

    def decode_features(self, left_latents: Tensor, right_latents: Tensor) -> Tensor:
        """Return the QueST-style decoder features before the action head."""

        memory = self.build_memory(left_latents, right_latents)
        query = torch.zeros(
            (memory.shape[0], self.action_horizon, self.model_dim),
            dtype=memory.dtype,
            device=memory.device,
        )
        query = query + self.decoder_pos(
            [query.shape[1]], device=query.device, dtype=query.dtype
        )
        if self.use_causal:
            mask = nn.Transformer.generate_square_subsequent_mask(
                query.shape[1], device=query.device
            )
            # Intentionally omit memory_mask: every query can read all paired slots.
            return self.decoder(query, memory, tgt_mask=mask, tgt_is_causal=True)
        return self.decoder(query, memory)

    def forward(self, left_latents: Tensor, right_latents: Tensor) -> Tensor:
        decoded = self.decode_features(left_latents, right_latents)
        return self.action_head(decoded)
