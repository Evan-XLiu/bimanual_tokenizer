"""QueST continuous action-latent autoencoder."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from praxis.codecs.latents.core import Autoencoder, AutoencoderEncoding
from praxis.codecs.latents.quest.config import QueSTAutoencoderConfig, QueSTModelConfig
from praxis.codecs.latents.quest.modeling import ResidualTemporalBlock
from praxis.nn.embeddings.positional import SinusoidPosEmbed


class QueSTAutoencoder(Autoencoder):
    """QueST temporal-conv/transformer autoencoder without FSQ."""

    config: QueSTModelConfig

    def __init__(
        self, config: dict[str, Any] | QueSTModelConfig | None = None, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = QueSTAutoencoderConfig()
        elif isinstance(config, QueSTModelConfig):
            cfg = config
        else:
            cfg = QueSTAutoencoderConfig.from_dict(config)

        if not _is_power_of_two(cfg.downsample_factor):
            raise ValueError(
                f"downsample_factor must be a power of two, got {cfg.downsample_factor}."
            )
        if cfg.action_horizon % cfg.downsample_factor != 0:
            raise ValueError(
                f"action_horizon={cfg.action_horizon} must be divisible by "
                f"downsample_factor={cfg.downsample_factor}."
            )

        self.config = cfg
        if cfg.action_dim is None:
            raise ValueError(
                f"{self.__class__.__name__} requires action_dim at construction time. "
                "Set codec.action_dim explicitly or build via dataset/PolicyIO-aware path."
            )
        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self.latent_horizon = self.action_horizon // int(cfg.downsample_factor)
        self.latent_dim = int(cfg.latent_dim)

        kernel_sizes, strides = _make_conv_schedule(int(cfg.downsample_factor))

        self.action_proj = nn.Linear(self.action_dim, cfg.encoder_dim)
        self.conv_block = ResidualTemporalBlock(
            cfg.encoder_dim,
            cfg.encoder_dim,
            kernel_size=kernel_sizes,
            stride=strides,
            causal=cfg.use_causal_encoder,
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.encoder_dim,
                nhead=cfg.encoder_nhead,
                dim_feedforward=4 * cfg.encoder_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=cfg.encoder_nlayer,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=cfg.decoder_dim,
                nhead=cfg.decoder_nhead,
                dim_feedforward=4 * cfg.decoder_dim,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=cfg.decoder_nlayer,
        )

        self.encoder_pos = SinusoidPosEmbed(
            cfg.encoder_dim, max_shape=[cfg.action_horizon]
        )
        self.decoder_pos = SinusoidPosEmbed(
            cfg.decoder_dim, max_shape=[cfg.action_horizon]
        )

        self.pre_quant = nn.Linear(cfg.encoder_dim, self.latent_dim)
        self.post_quant = nn.Linear(self.latent_dim, cfg.decoder_dim)
        self.action_head = nn.Linear(cfg.decoder_dim, self.action_dim)

        self.use_causal_encoder = bool(cfg.use_causal_encoder)
        self.use_causal_decoder = bool(cfg.use_causal_decoder)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _encode_norm_actions_to_latents(
        self, norm_actions: Tensor
    ) -> AutoencoderEncoding:
        x = self.action_proj(norm_actions)
        x = self.conv_block(x)
        x = x + self.encoder_pos([x.shape[1]], device=x.device, dtype=x.dtype)

        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                x.shape[1], device=x.device
            )
            x = self.encoder(x, mask=mask)
        else:
            x = self.encoder(x)

        return AutoencoderEncoding(latents=self.pre_quant(x))

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs: Any) -> Tensor:
        del kwargs
        memory = self.post_quant(latents)
        query = torch.zeros(
            (memory.shape[0], self.action_horizon, self.config.decoder_dim),
            dtype=memory.dtype,
            device=memory.device,
        )
        query = query + self.decoder_pos(
            [query.shape[1]], device=query.device, dtype=query.dtype
        )
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                query.shape[1], device=query.device
            )
            decoded = self.decoder(query, memory, tgt_mask=mask, tgt_is_causal=True)
        else:
            decoded = self.decoder(query, memory)
        return self.action_head(decoded)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _make_conv_schedule(downsample_factor: int) -> tuple[list[int], list[int]]:
    n_halves = int(math.log2(downsample_factor))
    if n_halves == 0:
        return [3, 2], [1, 1]
    kernel_sizes = [5] + [3] * n_halves
    strides = [2] * n_halves + [1]
    return kernel_sizes, strides
