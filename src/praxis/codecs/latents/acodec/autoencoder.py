# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""ActionCodec continuous action-latent autoencoder."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from praxis.codecs.latents.acodec.config import (
    ActionCodecAutoencoderConfig,
    ActionCodecModelConfig,
)
from praxis.codecs.latents.acodec.modeling import (
    ActionCodecDecoder,
    ActionCodecEncoder,
)
from praxis.codecs.latents.core import Autoencoder, AutoencoderEncoding


class ActionCodecAutoencoder(Autoencoder):
    """Perceiver encoder/decoder without vector quantization."""

    config: ActionCodecModelConfig

    def __init__(
        self,
        config: dict[str, Any] | ActionCodecModelConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = ActionCodecAutoencoderConfig()
        elif isinstance(config, ActionCodecModelConfig):
            cfg = config
        else:
            cfg = ActionCodecAutoencoderConfig.from_dict(config)

        self.config = cfg
        if cfg.action_dim is None:
            raise ValueError(
                f"{self.__class__.__name__} requires action_dim at construction time. "
                "Set codec.action_dim explicitly or build via dataset/PolicyIO-aware path."
            )

        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self.latent_horizon = int(cfg.latent_horizon)
        self.latent_dim = int(cfg.latent_dim)

        self.encoder = ActionCodecEncoder(
            sample_dim=self.action_dim,
            sample_horizon=self.action_horizon,
            latent_horizon=self.latent_horizon,
            emb_dim=cfg.emb_dim,
            latent_dim=self.latent_dim,
            head_dim=cfg.head_dim,
            depth=cfg.encoder_depth,
            dropout=cfg.dropout,
            mlp_ratio=cfg.mlp_ratio,
            qk_norm=cfg.qk_norm,
            layer_scale=cfg.layer_scale,
            pos_scale=cfg.pos_scale,
            add_self_attn=cfg.encoder_add_self_attn,
            add_causal_mask=cfg.encoder_add_causal_mask,
        )
        self.decoder = ActionCodecDecoder(
            sample_dim=self.action_dim,
            sample_horizon=self.action_horizon,
            latent_horizon=self.latent_horizon,
            emb_dim=cfg.emb_dim,
            latent_dim=self.latent_dim,
            head_dim=cfg.head_dim,
            depth=cfg.decoder_depth,
            dropout=cfg.dropout,
            mlp_ratio=cfg.mlp_ratio,
            qk_norm=cfg.qk_norm,
            layer_scale=cfg.layer_scale,
            pos_scale=cfg.pos_scale,
            add_self_attn=cfg.decoder_add_self_attn,
            add_causal_mask=cfg.decoder_add_causal_mask,
        )

    def _encode_norm_actions_to_latents(
        self, norm_actions: Tensor
    ) -> AutoencoderEncoding:
        return AutoencoderEncoding(latents=self.encoder(norm_actions))

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs: Any) -> Tensor:
        del kwargs
        return self.decoder(latents)
