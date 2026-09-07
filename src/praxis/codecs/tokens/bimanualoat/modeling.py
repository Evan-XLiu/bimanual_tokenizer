# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Independent common/relative modeling blocks for bimanual OAT."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from praxis.codecs.latents.oat.modeling import (
    LinearHead,
    LinearLayer,
    MaskedNestedDropout,
    SampleEmbedder,
)
from praxis.codecs.latents.oat.schedule import create_register_schedule_mask
from praxis.nn.architectures.transformer import TransformerDecoder
from praxis.nn.embeddings.positional import SinusoidPosEmbed


class PairedRegisterEncoder(nn.Module):
    """Encode common and relative modes with separate register streams."""

    def __init__(
        self,
        *,
        arm_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        common_latent_dim: int,
        relative_latent_dim: int,
        register_schedule: Sequence[int],
    ) -> None:
        super().__init__()
        schedule = tuple(int(value) for value in register_schedule)
        if not schedule or any(value <= 0 for value in schedule):
            raise ValueError(f"register_schedule must be positive, got {schedule}.")

        self.arm_dim = int(arm_dim)
        self.sample_dim = 2 * self.arm_dim
        self.register_schedule = schedule
        self.latent_horizon = sum(schedule)
        self.common_latent_dim = int(common_latent_dim)
        self.relative_latent_dim = int(relative_latent_dim)

        self.shared_arm_emb = SampleEmbedder(self.arm_dim, emb_dim)
        self.temporal_pos_emb = SinusoidPosEmbed(emb_dim, max_shape=[sample_horizon])
        self.branch_embeddings = nn.Parameter(torch.randn(2, emb_dim))
        self.common_registers = nn.Parameter(torch.randn(self.latent_horizon, emb_dim))
        self.relative_registers = nn.Parameter(
            torch.randn(self.latent_horizon, emb_dim)
        )
        self.register_pos_emb = SinusoidPosEmbed(
            emb_dim, max_shape=[self.latent_horizon]
        )
        self.shared_transformer = TransformerDecoder(
            dim=emb_dim,
            depth=depth,
            head_dim=head_dim,
            drop=pdropout,
        )
        self.common_head = LinearHead(emb_dim, self.common_latent_dim)
        self.relative_head = LinearHead(emb_dim, self.relative_latent_dim)

    def encode_arm_features(self, sample: Tensor) -> tuple[Tensor, Tensor]:
        if sample.ndim != 3 or sample.shape[-1] != self.sample_dim:
            raise ValueError(
                "expected canonical bimanual actions shaped "
                f"[B, T, {self.sample_dim}], got {tuple(sample.shape)}."
            )
        left, right = sample.split(self.arm_dim, dim=-1)
        left_feature = self.shared_arm_emb(left)
        right_feature = self.shared_arm_emb(right)
        common = (left_feature + right_feature) / 2
        relative = (left_feature - right_feature) / 2
        position = self.temporal_pos_emb(
            [sample.shape[1]], device=sample.device, dtype=common.dtype
        )
        return common + position, relative + position

    def _encode_stream(
        self,
        feature: Tensor,
        registers: Tensor,
        branch_index: int,
        head: LinearHead,
    ) -> Tensor:
        branch = self.branch_embeddings[branch_index].to(dtype=feature.dtype)
        feature = feature + branch.view(1, 1, -1)
        register_batch = registers.unsqueeze(0).expand(feature.shape[0], -1, -1)
        register_batch = register_batch + self.register_pos_emb(
            [self.latent_horizon],
            device=feature.device,
            dtype=register_batch.dtype,
        )
        register_batch = register_batch + branch.view(1, 1, -1)
        mask = create_register_schedule_mask(self.register_schedule, feature.device)
        encoded = self.shared_transformer(
            x=register_batch,
            context=feature,
            block_mask=mask,
            context_block_mask=None,
        )
        return head(encoded)

    def forward(self, sample: Tensor) -> tuple[Tensor, Tensor]:
        common_feature, relative_feature = self.encode_arm_features(sample)
        common = self._encode_stream(
            common_feature, self.common_registers, 0, self.common_head
        )
        relative = self._encode_stream(
            relative_feature, self.relative_registers, 1, self.relative_head
        )
        return common, relative


class PairedSinglePassDecoder(nn.Module):
    """Decode common and relative latent streams independently."""

    def __init__(
        self,
        *,
        arm_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        latent_dropout_mode: str,
        common_latent_dim: int,
        relative_latent_dim: int,
        latent_horizon: int,
    ) -> None:
        super().__init__()
        self.arm_dim = int(arm_dim)
        self.sample_dim = 2 * self.arm_dim
        self.sample_horizon = int(sample_horizon)
        self.latent_horizon = int(latent_horizon)
        self.latent_dropout_mode = latent_dropout_mode

        self.sample_pos_emb = SinusoidPosEmbed(emb_dim, max_shape=[self.sample_horizon])
        self.latent_pos_emb = SinusoidPosEmbed(emb_dim, max_shape=[self.latent_horizon])
        self.common_latent_proj = LinearLayer(common_latent_dim, emb_dim)
        self.relative_latent_proj = LinearLayer(relative_latent_dim, emb_dim)
        self.branch_embeddings = nn.Parameter(torch.randn(2, emb_dim))
        self.common_nested_dropout = MaskedNestedDropout(
            emb_dim, size_sampling_mode=latent_dropout_mode
        )
        self.relative_nested_dropout = MaskedNestedDropout(
            emb_dim, size_sampling_mode=latent_dropout_mode
        )
        self.shared_decoder = TransformerDecoder(
            dim=emb_dim,
            depth=depth,
            head_dim=head_dim,
            drop=pdropout,
            enable_self_attn=False,
        )
        self.shared_action_head = LinearHead(emb_dim, self.arm_dim)

    def _synchronized_nested_dropout(
        self,
        common: Tensor,
        relative: Tensor,
        decode_keep_k: list[int] | None,
    ) -> tuple[Tensor, Tensor]:
        if self.latent_dropout_mode == "disable":
            return common, relative
        if common.shape[:2] != relative.shape[:2]:
            raise ValueError("common and relative latent shapes must match.")

        keep_ks: Tensor | None = None
        if decode_keep_k is not None:
            keep_ks = torch.as_tensor(decode_keep_k, device=common.device)
        elif self.training:
            keep_ks = self.common_nested_dropout.sample_keep_k(
                common.shape[0], common.shape[1], common.device
            )
        if keep_ks is None:
            return common, relative

        mask = keep_ks.unsqueeze(1) <= torch.arange(
            common.shape[1], device=common.device
        ).unsqueeze(0)
        common_mask = self.common_nested_dropout.dropout_mask_token.to(
            device=common.device, dtype=common.dtype
        )
        relative_mask = self.relative_nested_dropout.dropout_mask_token.to(
            device=relative.device, dtype=relative.dtype
        )
        return (
            torch.where(mask.unsqueeze(-1), common_mask.view(1, 1, -1), common),
            torch.where(mask.unsqueeze(-1), relative_mask.view(1, 1, -1), relative),
        )

    def _decode_stream(self, memory: Tensor, branch_index: int) -> Tensor:
        branch = self.branch_embeddings[branch_index].to(dtype=memory.dtype)
        query = self.sample_pos_emb(
            [self.sample_horizon], device=memory.device, dtype=memory.dtype
        ).expand(memory.shape[0], -1, -1)
        branch = branch.view(1, 1, -1)
        return self.shared_decoder(query + branch, memory + branch)

    def decode_modes(
        self,
        common_latents: Tensor,
        relative_latents: Tensor,
        decode_keep_k: list[int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        common = self.common_latent_proj(common_latents)
        relative = self.relative_latent_proj(relative_latents)
        common, relative = self._synchronized_nested_dropout(
            common, relative, decode_keep_k
        )
        position = self.latent_pos_emb(
            [self.latent_horizon], device=common.device, dtype=common.dtype
        )
        common = self._decode_stream(common + position, 0)
        relative = self._decode_stream(relative + position, 1)
        return common, relative

    def forward(
        self,
        common_latents: Tensor,
        relative_latents: Tensor,
        decode_keep_k: list[int] | None = None,
    ) -> Tensor:
        common, relative = self.decode_modes(
            common_latents, relative_latents, decode_keep_k
        )
        left = self.shared_action_head(common + relative)
        canonical_right = self.shared_action_head(common - relative)
        return torch.cat((left, canonical_right), dim=-1)
