# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Bimanual QueST tokenizer with two-level residual FSQ tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from praxis.codecs.core import BaseCodec
from praxis.codecs.latents.core import Autoencoder, AutoencoderEncoding
from praxis.codecs.latents.core.training import (
    masked_autoencoder_loss,
    split_decay_param_groups,
)
from praxis.codecs.tokens.bquest.config import BQuestTokenizerConfig
from praxis.codecs.tokens.bquest.modeling import (
    BidirectionalCausalCrossAttention,
    BQuestDecoder,
    SharedArmTemporalEncoder,
    init_bquest_weights,
)
from praxis.codecs.tokens.core import AutoencoderTokenization, AutoencoderTokenizer
from praxis.core.types import NormMode, PolicyIO
from praxis.nn.quantization.fsq import FiniteScalarQuantization

_NORMALIZATION_KIND = "bquest_joint_pair_v1"


@dataclass(frozen=True, kw_only=True)
class BQuestTokenization(AutoencoderTokenization):
    """Both the coarse latent and the final residual latent path."""

    coarse_latents: Tensor


class BQuestTokenizer(AutoencoderTokenizer, Autoencoder):
    """Shared-arm temporal/cross-attention tokenizer for 14D bimanual actions."""

    config: BQuestTokenizerConfig

    def __init__(
        self,
        config: dict[str, Any] | BQuestTokenizerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = BQuestTokenizerConfig()
        elif isinstance(config, BQuestTokenizerConfig):
            cfg = config
        else:
            cfg = BQuestTokenizerConfig.from_dict(config)
        if cfg.action_dim is None:
            raise ValueError("BQuestTokenizer requires action_dim=14.")
        if cfg.action_horizon != 32 or cfg.downsample_factor != 4:
            raise ValueError(
                "BQuestTokenizer currently requires action_horizon=32 and "
                "downsample_factor=4."
            )

        self.config = cfg
        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self.latent_horizon = self.action_horizon // int(cfg.downsample_factor)
        self.latent_dim = int(cfg.latent_dim)
        self.arm_dim = self.action_dim // 2

        self.register_buffer(
            "right_arm_sign",
            torch.tensor(cfg.right_arm_sign, dtype=torch.float32),
            persistent=True,
        )
        self.arm_encoder = SharedArmTemporalEncoder(
            arm_dim=self.arm_dim,
            action_horizon=self.action_horizon,
            latent_horizon=self.latent_horizon,
            encoder_dim=cfg.encoder_dim,
            encoder_nhead=cfg.encoder_nhead,
            encoder_nlayer=cfg.encoder_nlayer,
            dropout=cfg.dropout,
            use_causal=cfg.use_causal_encoder,
        )
        self.cross_attention = BidirectionalCausalCrossAttention(
            cfg.encoder_dim, cfg.encoder_nhead, cfg.dropout
        )
        self.fusion = torch.nn.Linear(2 * cfg.encoder_dim, cfg.encoder_dim)
        self.pre_quant = torch.nn.Linear(cfg.encoder_dim, self.latent_dim)
        self.decoder = BQuestDecoder(
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            latent_horizon=self.latent_horizon,
            latent_dim=self.latent_dim,
            decoder_dim=cfg.decoder_dim,
            decoder_nhead=cfg.decoder_nhead,
            decoder_nlayer=cfg.decoder_nlayer,
            dropout=cfg.dropout,
            use_causal=cfg.use_causal_decoder,
        )
        self.coarse_quantizer = FiniteScalarQuantization(levels=list(cfg.fsq_levels))
        self.refine_quantizer = FiniteScalarQuantization(levels=list(cfg.fsq_levels))
        init_bquest_weights(self)

    @property
    def block_schedule(self) -> tuple[int, int]:
        return (self.latent_horizon, self.latent_horizon)

    @property
    def codebook_size(self) -> int:
        return int(self.coarse_quantizer.codebook_size)

    @property
    def max_code_seq_len(self) -> int:
        return 2 * self.latent_horizon

    @property
    def supports_prefix_decode(self) -> bool:
        return False

    def canonicalize(self, actions: Tensor) -> Tensor:
        self._validate_action_shape(actions)
        if not self.config.canonicalize_right_arm:
            return actions
        left, right = actions.split(self.arm_dim, dim=-1)
        sign = self.right_arm_sign.to(device=actions.device, dtype=actions.dtype)
        return torch.cat((left, right * sign), dim=-1)

    def decanonicalize(self, actions: Tensor) -> Tensor:
        return self.canonicalize(actions)

    def _normalize_action_tensor(self, actions: Tensor) -> Tensor:
        return BaseCodec._normalize_action_tensor(self, self.canonicalize(actions))

    def _unnormalize_action_tensor(self, actions: Tensor) -> Tensor:
        canonical = BaseCodec._unnormalize_action_tensor(self, actions)
        return self.decanonicalize(canonical)

    def set_normalization_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        modes: dict[str, NormMode] | None = None,
    ) -> None:
        source_key = self.action_key if self.action_key in stats else None
        if source_key is None and len(stats) == 1:
            source_key = next(iter(stats))
        if source_key is None:
            BaseCodec.set_normalization_stats(self, stats, modes=modes)
            return
        transformed = self._canonicalize_stats(stats[source_key])
        mode = modes.get(source_key) if modes is not None else None
        BaseCodec.set_normalization_stats(
            self,
            {self.action_key: transformed},
            modes={self.action_key: mode} if mode is not None else None,
        )

    def set_io(self, io: PolicyIO) -> None:
        """Update the action key without re-canonicalizing already transformed stats."""
        old_key = self.action_key
        old_stats = self.normalize_action.stats if self.normalize_action else None
        old_modes = self.normalize_action.modes if self.normalize_action else None
        self.io = io
        if io.action_feature is not None:
            self.action_key = io.action_feature.key
        if old_key == self.action_key or old_stats is None or old_modes is None:
            return
        source_key = old_key if old_key in old_stats else next(iter(old_stats), None)
        if source_key is not None:
            BaseCodec.set_normalization_stats(
                self,
                {self.action_key: old_stats[source_key]},
                modes={self.action_key: old_modes[source_key]},
            )

    def _canonicalize_stats(self, stats: dict[str, Tensor]) -> dict[str, Tensor]:
        tensors = {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in stats.items()
        }
        for name, value in tensors.items():
            if name in {"min", "max", "mean", "std"} and (
                value.ndim != 1 or value.numel() != self.action_dim
            ):
                raise ValueError(
                    f"action statistic {name!r} must have shape [{self.action_dim}], "
                    f"got {tuple(value.shape)}."
                )
        sign = (
            self.right_arm_sign.cpu()
            if self.config.canonicalize_right_arm
            else torch.ones(self.arm_dim, dtype=torch.float32)
        )
        result = {
            name: value
            for name, value in tensors.items()
            if name not in {"min", "max", "mean", "std"}
        }
        if "min" in tensors and "max" in tensors:
            left_min, right_min = tensors["min"].split(self.arm_dim)
            left_max, right_max = tensors["max"].split(self.arm_dim)
            canonical_right_min = torch.where(sign > 0, right_min, -right_max)
            canonical_right_max = torch.where(sign > 0, right_max, -right_min)
            pair_min = torch.minimum(left_min, canonical_right_min)
            pair_max = torch.maximum(left_max, canonical_right_max)
            result["min"] = pair_min.repeat(2)
            result["max"] = pair_max.repeat(2)
        if "mean" in tensors and "std" in tensors:
            left, right = tensors["mean"].split(self.arm_dim)
            left_std, right_std = tensors["std"].split(self.arm_dim)
            canonical_right_mean = right * sign
            pair_mean = (left + canonical_right_mean) / 2
            pair_var = (
                left_std.square()
                + right_std.square()
                + (left - pair_mean).square()
                + (canonical_right_mean - pair_mean).square()
            ) / 2
            result["mean"] = pair_mean.repeat(2)
            result["std"] = pair_var.clamp_min(0.0).sqrt().repeat(2)
        elif "mean" in tensors:
            left, right = tensors["mean"].split(self.arm_dim)
            result["mean"] = ((left + right * sign) / 2).repeat(2)
        elif "std" in tensors:
            left, right = tensors["std"].split(self.arm_dim)
            result["std"] = torch.sqrt((left.square() + right.square()) / 2).repeat(2)
        return result

    def _encode_fused_norm_actions(self, norm_actions: Tensor) -> Tensor:
        self._validate_action_shape(norm_actions)
        left, right = norm_actions.split(self.arm_dim, dim=-1)
        left_features = self.arm_encoder(left)
        right_features = self.arm_encoder(right)
        left_context, right_context = self.cross_attention(left_features, right_features)
        return self.fusion(torch.cat((left_context, right_context), dim=-1))

    def _encode_norm_actions_to_latents(
        self, norm_actions: Tensor
    ) -> AutoencoderEncoding:
        """Encode normalized actions to the continuous pre-quantized latent path."""
        fused = self._encode_fused_norm_actions(norm_actions)
        return AutoencoderEncoding(latents=self.pre_quant(fused))

    def _normalization_checkpoint_payload(self) -> dict[str, Any]:
        payload = super()._normalization_checkpoint_payload()
        payload["normalization_kind"] = _NORMALIZATION_KIND
        return payload

    def _load_normalization_checkpoint_payload(self, data: dict[str, Any]) -> None:
        kind = data.get("normalization_kind")
        if kind != _NORMALIZATION_KIND:
            raise ValueError(
                "BQuestTokenizer requires canonical normalization payload "
                f"{_NORMALIZATION_KIND!r}, got {kind!r}."
            )
        stats = data.get("stats", {})
        modes = {key: NormMode(value) for key, value in data.get("modes", {}).items()}
        BaseCodec.set_normalization_stats(self, stats, modes=modes)

    def _tokenize_norm_actions(self, norm_actions: Tensor) -> BQuestTokenization:
        fused = self._encode_fused_norm_actions(norm_actions)
        raw_latent = self.pre_quant(fused)
        # FSQ.forward() returns normalized codebook values, while raw_latent is
        # unbounded.  Compute the residual after mapping raw_latent into that
        # same normalized code space.
        latent_code = self._latent_to_fsq_code_space(raw_latent)
        coarse, coarse_tokens = self.coarse_quantizer(raw_latent)
        residual = latent_code - coarse.detach()
        residual_scale = self._fsq_residual_scale(residual)
        refine_code, refine_tokens = self._quantize_fsq_code_space(
            residual / residual_scale, self.refine_quantizer
        )
        refine = refine_code * residual_scale
        return BQuestTokenization(
            latents=coarse + refine,
            coarse_latents=coarse,
            token_ids=torch.cat((coarse_tokens, refine_tokens), dim=1),
        )

    def _latent_to_fsq_code_space(self, latent: Tensor) -> Tensor:
        """Map raw encoder latents to the normalized FSQ code space."""
        levels = torch.as_tensor(
            self.config.fsq_levels, device=latent.device, dtype=torch.float32
        )
        half_width = levels // 2
        code_min = -torch.ones_like(half_width)
        code_max = (levels - 1.0 - half_width) / half_width
        bounded = self.coarse_quantizer.bound(latent.float()) / half_width
        return torch.maximum(torch.minimum(bounded, code_max), code_min)

    def _quantize_fsq_code_space(
        self, codes: Tensor, quantizer: FiniteScalarQuantization
    ) -> tuple[Tensor, Tensor]:
        """Quantize already-normalized FSQ codes without rebounding them."""
        levels = torch.as_tensor(
            self.config.fsq_levels, device=codes.device, dtype=torch.float32
        )
        half_width = levels // 2
        code_min = -torch.ones_like(half_width)
        code_max = (levels - 1.0 - half_width) / half_width
        bounded = torch.maximum(torch.minimum(codes.float(), code_max), code_min)
        token_ids = quantizer.codes_to_indices(bounded).long()
        quantized_grid = quantizer.indices_to_embedding(token_ids)
        quantized = bounded + (quantized_grid - bounded).detach()
        return quantized, token_ids

    def _fsq_residual_scale(self, values: Tensor) -> Tensor:
        """Return half a coarse FSQ grid step per latent channel.

        The first-level residual is smaller than one coarse grid step. Scaling
        it by this amount gives the second FSQ its full normalized input range;
        scaling the quantized code back keeps ``coarse + refine`` in the same
        code space as the first-level latent.
        """
        levels = torch.as_tensor(
            self.config.fsq_levels, device=values.device, dtype=torch.float32
        )
        half_width = levels // 2
        return 1.0 / (2.0 * half_width)

    def _tokens_to_latents(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or int(tokens.shape[1]) != self.max_code_seq_len:
            raise ValueError(
                "BQuest token shape mismatch: expected "
                f"[B, {self.max_code_seq_len}], got {tuple(tokens.shape)}."
            )
        coarse_tokens, refine_tokens = tokens.long().split(self.latent_horizon, dim=1)
        coarse = self.coarse_quantizer.indices_to_embedding(coarse_tokens)
        refine = self.refine_quantizer.indices_to_embedding(refine_tokens)
        refine = refine * self._fsq_residual_scale(refine)
        return coarse + refine

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs: Any) -> Tensor:
        del kwargs
        return self.decoder(latents)

    def decode_coarse(self, latents: Tensor) -> Tensor:
        """Decode only the level-1 coarse latent stream to physical actions."""
        return self._unnormalize_action_tensor(self.decoder(latents))

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        actions = batch[self.action_key]
        norm_actions = self._normalize_action_tensor(actions)
        encoded = self._tokenize_norm_actions(norm_actions)
        recon_8 = self.decoder(encoded.coarse_latents)
        recon_16 = self.decoder(encoded.latents)
        pad_mask = batch.get(f"{self.action_key}_is_pad")
        coarse_loss = masked_autoencoder_loss(
            F.mse_loss(recon_8, norm_actions, reduction="none"), pad_mask
        )
        refine_loss = masked_autoencoder_loss(
            F.mse_loss(recon_16, norm_actions, reduction="none"), pad_mask
        )
        loss = coarse_loss + refine_loss
        return loss, {
            "coarse_recon_loss": float(coarse_loss.detach().item()),
            "refine_recon_loss": float(refine_loss.detach().item()),
        }

    def get_optim_params(self) -> list[dict[str, Any]]:
        return split_decay_param_groups(self, decay_weight_decay=0.0, no_decay_weight_decay=0.0)

    def _validate_action_shape(self, actions: Tensor) -> None:
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"BQuest expects actions shaped [B, {self.action_horizon}, {self.action_dim}], "
                f"got {tuple(actions.shape)}."
            )
