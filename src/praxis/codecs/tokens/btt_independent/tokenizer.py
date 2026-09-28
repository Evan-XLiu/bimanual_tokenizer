# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""BTT ablation with independent arm parameters and the original joint decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import torch
import torch.nn.functional as F
from torch import Tensor

from praxis.codecs.latents.core.training import masked_autoencoder_loss
from praxis.codecs.tokens.btt.decoder import TimestepPairedBTTDecoder
from praxis.codecs.tokens.btt_independent.config import BTTIndependentConfig
from praxis.codecs.tokens.btt_independent.encoder import BTTIndependentEncoder
from praxis.codecs.tokens.core import BaseTokenizer
from praxis.nn.quantization.fsq import FiniteScalarQuantization


@dataclass(frozen=True, kw_only=True)
class BTTIndependentTokenization:
    """Independent quantized representations and token IDs for both arms."""

    left_latents: Tensor
    right_latents: Tensor
    left_tokens: Tensor
    right_tokens: Tensor

    @property
    def token_ids(self) -> Tensor:
        """Serialize 8 left then 8 right IDs without merging any token pair."""

        return torch.cat((self.left_tokens, self.right_tokens), dim=1)


class BTTIndependentTokenizer(BaseTokenizer):
    """Bimanual temporal tokenizer with paired post-dequantization memory."""

    config: BTTIndependentConfig

    def __init__(
        self, config: dict[str, Any] | BTTIndependentConfig | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = BTTIndependentConfig()
        elif isinstance(config, BTTIndependentConfig):
            cfg = config
        else:
            cfg = BTTIndependentConfig.from_dict(config)
        self.config = cfg

        self.encoder = BTTIndependentEncoder(cfg)
        # Independent learned projections; FSQ retains the original fixed grid.
        self.left_pre_quant = torch.nn.Linear(cfg.model_dim, cfg.latent_dim)
        self.right_pre_quant = torch.nn.Linear(cfg.model_dim, cfg.latent_dim)
        self.quantizer = FiniteScalarQuantization(levels=list(cfg.fsq_levels))
        self.decoder = TimestepPairedBTTDecoder(
            latent_dim=cfg.latent_dim,
            latent_horizon=cfg.latent_horizon_per_arm,
            model_dim=cfg.model_dim,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            num_heads=cfg.decoder_nhead,
            num_layers=cfg.decoder_nlayer,
            dropout=cfg.dropout,
            use_causal=cfg.use_causal_decoder,
        )
        nn_init = torch.nn.init
        for projection in (self.left_pre_quant, self.right_pre_quant):
            nn_init.xavier_uniform_(projection.weight)
            if projection.bias is not None:
                nn_init.zeros_(projection.bias)

    @classmethod
    def _from_pretrained(cls, **kwargs: Any):
        # Missing/extra keys must not silently initialize a different architecture.
        if kwargs.get("strict", True) is not True:
            raise ValueError("btt_independent requires strict checkpoint loading.")
        kwargs["strict"] = True
        return super()._from_pretrained(**kwargs)

    @staticmethod
    def validate_pretrained_runtime_overrides(
        *,
        build_cfg: dict[str, Any],
        source_cfg: dict[str, Any],
        load_id: str,
    ) -> None:
        from praxis.core.config import config_values_equal

        for key, value in build_cfg.items():
            if key not in source_cfg or not config_values_equal(value, source_cfg[key]):
                raise ValueError(
                    "Cannot override btt_independent checkpoint field "
                    f"{key!r} at {load_id}."
                )

    @property
    def codebook_size(self) -> int:
        return int(self.quantizer.codebook_size)

    @property
    def max_code_seq_len(self) -> int:
        """Total serialized token count: eight left plus eight right."""

        return 2 * self.config.latent_horizon_per_arm

    @property
    def block_schedule(self) -> tuple[int, int]:
        return (
            self.config.latent_horizon_per_arm,
            self.config.latent_horizon_per_arm,
        )

    def _normalize_action_tensor(self, actions: Tensor) -> Tensor:
        left, right = self.encoder.split_canonicalize_and_normalize(actions)
        return torch.cat((left, right), dim=-1)

    def _unnormalize_action_tensor(self, actions: Tensor) -> Tensor:
        self.encoder._validate_action_shape(actions)
        left, right = actions.split(self.config.arm_dim, dim=-1)
        return self.encoder.unnormalize_and_decanonicalize(left, right)

    def _encode_normalized_arms(self, norm_actions: Tensor) -> tuple[Tensor, Tensor]:
        self.encoder._validate_action_shape(norm_actions)
        left, right = norm_actions.split(self.config.arm_dim, dim=-1)
        left_features = self.encoder.encode_single_arm_dynamics(left, arm="left")
        right_features = self.encoder.encode_single_arm_dynamics(right, arm="right")
        return self.encoder.cross_attention(left_features, right_features)

    def _tokenize_norm_actions(
        self, norm_actions: Tensor
    ) -> BTTIndependentTokenization:
        left_features, right_features = self._encode_normalized_arms(norm_actions)
        left_latents, left_tokens = self.quantizer(self.left_pre_quant(left_features))
        right_latents, right_tokens = self.quantizer(
            self.right_pre_quant(right_features)
        )
        return BTTIndependentTokenization(
            left_latents=left_latents,
            right_latents=right_latents,
            left_tokens=left_tokens,
            right_tokens=right_tokens,
        )

    def tokenize_with_latents(self, actions: Tensor) -> BTTIndependentTokenization:
        return self._tokenize_norm_actions(self._normalize_action_tensor(actions))

    def encode(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Return distinct `(left_tokens, right_tokens)`, each shaped `[B, 8]`."""

        encoded = self.tokenize_with_latents(actions)
        return encoded.left_tokens, encoded.right_tokens

    def tokenize(self, actions: Tensor) -> Tensor:
        """Project API serialization: `[left_0..7, right_0..7]`, shape `[B,16]`."""

        return self.tokenize_with_latents(actions).token_ids

    def _tokens_to_latents(
        self, left_tokens: Tensor, right_tokens: Tensor
    ) -> tuple[Tensor, Tensor]:
        expected = self.config.latent_horizon_per_arm
        for name, tokens in (
            ("left_tokens", left_tokens),
            ("right_tokens", right_tokens),
        ):
            if tokens.ndim != 2 or tokens.shape[1] != expected:
                raise ValueError(
                    f"{name} must have shape [B, {expected}], got {tuple(tokens.shape)}."
                )
        if left_tokens.shape[0] != right_tokens.shape[0]:
            raise ValueError("left and right token batch sizes must match.")
        return (
            self.quantizer.indices_to_embedding(left_tokens.long()),
            self.quantizer.indices_to_embedding(right_tokens.long()),
        )

    def decode(self, left_tokens: Tensor, right_tokens: Tensor) -> Tensor:
        left_latents, right_latents = self._tokens_to_latents(left_tokens, right_tokens)
        norm_actions = self.decoder(left_latents, right_latents)
        return self._unnormalize_action_tensor(norm_actions)

    def detokenize(self, tokens: Tensor, **kwargs: Any) -> Tensor:
        if kwargs:
            raise TypeError(
                "BTTIndependent detokenize received unsupported kwargs: "
                f"{sorted(kwargs)}"
            )
        expected = self.max_code_seq_len
        if tokens.ndim != 2 or tokens.shape[1] != expected:
            raise ValueError(
                f"tokens must have shape [B, {expected}], got {tuple(tokens.shape)}."
            )
        left_tokens, right_tokens = tokens.split(
            self.config.latent_horizon_per_arm, dim=1
        )
        return self.decode(left_tokens, right_tokens)

    def decode_latents(self, left_latents: Tensor, right_latents: Tensor) -> Tensor:
        norm_actions = self.decoder(left_latents, right_latents)
        return self._unnormalize_action_tensor(norm_actions)

    def autoencode(self, actions: Tensor) -> Tensor:
        encoded = self.tokenize_with_latents(actions)
        return self.decode_latents(encoded.left_latents, encoded.right_latents)

    @overload
    def forward(self, batch: Tensor) -> Tensor: ...

    @overload
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]: ...

    def forward(
        self, batch: Tensor | dict[str, Tensor]
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        """Autoencode tensors; use normalized MSE for the project training API."""

        if isinstance(batch, Tensor):
            return self.autoencode(batch)
        actions = batch[self.action_key]
        norm_actions = self._normalize_action_tensor(actions)
        encoded = self._tokenize_norm_actions(norm_actions)
        recon = self.decoder(encoded.left_latents, encoded.right_latents)
        raw_loss = F.mse_loss(recon, norm_actions, reduction="none")
        loss = masked_autoencoder_loss(raw_loss, batch.get(f"{self.action_key}_is_pad"))
        return loss, {}
