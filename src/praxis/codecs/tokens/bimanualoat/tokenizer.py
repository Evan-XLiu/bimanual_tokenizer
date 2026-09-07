# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Paired bimanual OAT tokenizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from praxis.codecs.core import BaseCodec
from praxis.codecs.latents.core import AutoencoderEncoding
from praxis.codecs.latents.oat import OATAutoencoder
from praxis.codecs.tokens.bimanualoat.config import BimanualOATTokenizerConfig
from praxis.codecs.tokens.bimanualoat.convention import apply_bimanual_convention
from praxis.codecs.tokens.bimanualoat.modeling import (
    PairedRegisterEncoder,
    PairedSinglePassDecoder,
)
from praxis.codecs.tokens.bimanualoat.quantizer import (
    PairedFiniteScalarQuantization,
)
from praxis.codecs.tokens.core import AutoencoderTokenization, AutoencoderTokenizer
from praxis.codecs.tokens.oat.prefix import prefix_token_ids_to_tensor
from praxis.core.types import NormMode, PolicyIO

_NORMALIZATION_KIND = "bimanualoat_pooled_v1"


class BimanualOATTokenizer(AutoencoderTokenizer, OATAutoencoder):
    """OAT tokenizer with paired common and relative bimanual streams."""

    config: BimanualOATTokenizerConfig
    encoder: PairedRegisterEncoder
    decoder: PairedSinglePassDecoder

    def __init__(
        self,
        config: dict[str, Any] | BimanualOATTokenizerConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            cfg = BimanualOATTokenizerConfig()
        elif isinstance(config, BimanualOATTokenizerConfig):
            cfg = config
        else:
            cfg = BimanualOATTokenizerConfig.from_dict(config)
        super().__init__(config=cfg, **kwargs)
        sign = torch.tensor(cfg.right_arm_sign, dtype=torch.float32)
        self.register_buffer("right_arm_sign", sign, persistent=True)
        self.common_latent_dim = len(cfg.common_quantizer_levels)
        self.relative_latent_dim = len(cfg.relative_quantizer_levels)
        self.quantizer = PairedFiniteScalarQuantization(
            common_levels=cfg.common_quantizer_levels,
            relative_levels=cfg.relative_quantizer_levels,
            drop_quant_p=cfg.quantizer_drop_quant_p,
            corrupt_tokens_p=cfg.quantizer_corrupt_tokens_p,
            min_corrupt_tokens_p=cfg.quantizer_min_corrupt_tokens_p,
            apply_corrupt_tokens_p=cfg.quantizer_apply_corrupt_tokens_p,
        )

    def _build_encoder(
        self, cfg: BimanualOATTokenizerConfig, latent_dim: int
    ) -> PairedRegisterEncoder:
        del latent_dim
        if cfg.action_dim is None:
            raise ValueError("BimanualOATTokenizer requires action_dim.")
        return PairedRegisterEncoder(
            arm_dim=int(cfg.action_dim) // 2,
            sample_horizon=cfg.action_horizon,
            emb_dim=cfg.emb_dim,
            head_dim=cfg.head_dim,
            depth=cfg.encoder_depth,
            pdropout=cfg.dropout,
            common_latent_dim=len(cfg.common_quantizer_levels),
            relative_latent_dim=len(cfg.relative_quantizer_levels),
            register_schedule=cfg.register_schedule,
        )

    def _build_decoder(
        self,
        cfg: BimanualOATTokenizerConfig,
        latent_dim: int,
        latent_horizon: int,
    ) -> PairedSinglePassDecoder:
        del latent_dim
        if cfg.action_dim is None:
            raise ValueError("BimanualOATTokenizer requires action_dim.")
        return PairedSinglePassDecoder(
            arm_dim=int(cfg.action_dim) // 2,
            sample_horizon=cfg.action_horizon,
            emb_dim=cfg.emb_dim,
            head_dim=cfg.head_dim,
            depth=cfg.decoder_depth,
            pdropout=cfg.dropout,
            latent_dropout_mode=cfg.latent_dropout_mode,
            common_latent_dim=len(cfg.common_quantizer_levels),
            relative_latent_dim=len(cfg.relative_quantizer_levels),
            latent_horizon=latent_horizon,
        )

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return self.config.block_schedule

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    @property
    def max_code_seq_len(self) -> int:
        return self.latent_horizon

    def canonicalize(self, actions: Tensor) -> Tensor:
        return apply_bimanual_convention(actions, self.right_arm_sign)

    def decanonicalize(self, actions: Tensor) -> Tensor:
        return apply_bimanual_convention(actions, self.right_arm_sign)

    def _normalize_action_tensor(self, actions: Tensor) -> Tensor:
        return BaseCodec._normalize_action_tensor(self, self.canonicalize(actions))

    def _unnormalize_action_tensor(self, actions: Tensor) -> Tensor:
        canonical = BaseCodec._unnormalize_action_tensor(self, actions)
        return self.decanonicalize(canonical)

    def _encode_norm_actions_to_latents(
        self, norm_actions: Tensor
    ) -> AutoencoderEncoding:
        common, relative = self.encoder(norm_actions)
        return AutoencoderEncoding(latents=torch.cat((common, relative), dim=-1))

    def _tokenize_norm_actions(self, norm_actions: Tensor) -> AutoencoderTokenization:
        common, relative = self.encoder(norm_actions)
        latents, tokens = self.quantizer(common, relative)
        return AutoencoderTokenization(latents=latents, token_ids=tokens)

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs: Any) -> Tensor:
        latents, decode_keep_k = self._prepare_decode_latents(
            latents, decode_keep_k=kwargs.get("decode_keep_k")
        )
        common, relative = latents.split(
            (self.common_latent_dim, self.relative_latent_dim), dim=-1
        )
        return self.decoder(common, relative, decode_keep_k=decode_keep_k)

    def _tokens_to_latents(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or int(tokens.shape[1]) != self.latent_horizon:
            raise ValueError(
                "token shape mismatch: expected "
                f"[B, {self.latent_horizon}], got {tuple(tokens.shape)}."
            )
        return self.quantizer.indices_to_embedding(tokens.long())

    def detokenize(
        self, tokens: Tensor | Sequence[Sequence[int]], **kwargs: Any
    ) -> Tensor:
        if "decode_keep_k" in kwargs:
            raise ValueError(
                "detokenize derives decode_keep_k from token prefix lengths."
            )
        ref_param = next(self.parameters(), None)
        device = ref_param.device if ref_param is not None else None
        token_tensor, token_lens = prefix_token_ids_to_tensor(
            tokens, token_horizon=self.latent_horizon, device=device
        )
        latents = self._tokens_to_latents(token_tensor)
        return self.decode(latents, decode_keep_k=token_lens, **kwargs)

    def set_io(self, io: PolicyIO) -> None:
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
        pooled = self._pool_bimanual_stats(stats[source_key])
        mode = modes.get(source_key) if modes is not None else None
        BaseCodec.set_normalization_stats(
            self,
            {self.action_key: pooled},
            modes={self.action_key: mode} if mode is not None else None,
        )

    def _pool_bimanual_stats(self, stats: dict[str, Tensor]) -> dict[str, Tensor]:
        arm_dim = self.action_dim // 2
        tensors = {
            name: torch.as_tensor(value, dtype=torch.float32)
            for name, value in stats.items()
            if name in {"min", "max", "mean", "std"}
        }
        for name, value in tensors.items():
            if value.ndim != 1 or value.numel() != self.action_dim:
                raise ValueError(
                    f"action statistic {name!r} must have shape "
                    f"[{self.action_dim}], got {tuple(value.shape)}."
                )
        sign = self.right_arm_sign.cpu()
        result: dict[str, Tensor] = {}
        if "min" in tensors and "max" in tensors:
            left_min, right_min = tensors["min"].split(arm_dim)
            left_max, right_max = tensors["max"].split(arm_dim)
            canonical_right_min = torch.where(sign > 0, right_min, -right_max)
            canonical_right_max = torch.where(sign > 0, right_max, -right_min)
            pooled_min = torch.minimum(left_min, canonical_right_min)
            pooled_max = torch.maximum(left_max, canonical_right_max)
            result["min"] = pooled_min.repeat(2)
            result["max"] = pooled_max.repeat(2)
        if "mean" in tensors and "std" in tensors:
            left_mean, right_mean = tensors["mean"].split(arm_dim)
            left_std, right_std = tensors["std"].split(arm_dim)
            right_mean = right_mean * sign
            pooled_mean = (left_mean + right_mean) / 2
            pooled_var = (
                left_std.square()
                + right_std.square()
                + (left_mean - pooled_mean).square()
                + (right_mean - pooled_mean).square()
            ) / 2
            result["mean"] = pooled_mean.repeat(2)
            result["std"] = pooled_var.sqrt().repeat(2)
        return result

    def _normalization_checkpoint_payload(self) -> dict[str, Any]:
        payload = super()._normalization_checkpoint_payload()
        payload["normalization_kind"] = _NORMALIZATION_KIND
        return payload

    def _load_normalization_checkpoint_payload(self, data: dict[str, Any]) -> None:
        kind = data.get("normalization_kind")
        if kind != _NORMALIZATION_KIND:
            raise ValueError(
                "BimanualOATTokenizer requires pooled normalization payload "
                f"{_NORMALIZATION_KIND!r}, got {kind!r}."
            )
        stats = data.get("stats", {})
        modes = {key: NormMode(value) for key, value in data.get("modes", {}).items()}
        BaseCodec.set_normalization_stats(self, stats, modes=modes)
