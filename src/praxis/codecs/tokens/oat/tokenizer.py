# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""OAT tokenizer implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from praxis.codecs.latents.oat import OATAutoencoder
from praxis.codecs.tokens.core import (
    AutoencoderTokenization,
    AutoencoderTokenizer,
)
from praxis.codecs.tokens.oat.config import OATTokenizerConfig
from praxis.codecs.tokens.oat.prefix import prefix_token_ids_to_tensor
from praxis.nn.quantization.fsq import FiniteScalarQuantization


class OATTokenizer(AutoencoderTokenizer, OATAutoencoder):
    """OAT latent autoencoder with finite-scalar quantization."""

    config: OATTokenizerConfig

    def __init__(
        self, config: dict[str, Any] | OATTokenizerConfig | None = None, **kwargs
    ) -> None:
        if config is None:
            cfg = OATTokenizerConfig()
        elif isinstance(config, OATTokenizerConfig):
            cfg = config
        else:
            cfg = OATTokenizerConfig.from_dict(config)
        super().__init__(config=cfg, **kwargs)
        if self.latent_dim != len(self.config.quantizer_levels):
            raise ValueError(
                "OATTokenizer latent_dim must match quantizer_levels length; "
                f"got latent_dim={self.latent_dim}, "
                f"levels={len(self.config.quantizer_levels)}."
            )
        self.quantizer = FiniteScalarQuantization(
            levels=list(self.config.quantizer_levels),
            drop_quant_p=self.config.quantizer_drop_quant_p,
            corrupt_tokens_p=self.config.quantizer_corrupt_tokens_p,
            min_corrupt_tokens_p=self.config.quantizer_min_corrupt_tokens_p,
            apply_corrupt_tokens_p=self.config.quantizer_apply_corrupt_tokens_p,
        )

    @property
    def block_schedule(self) -> tuple[int, ...]:
        return self.config.block_schedule

    @property
    def codebook_size(self) -> int:
        return int(self.quantizer.codebook_size)

    @property
    def max_code_seq_len(self) -> int:
        return int(self.latent_horizon)

    def _tokenize_norm_actions(self, norm_actions: Tensor) -> AutoencoderTokenization:
        encoded = OATAutoencoder._encode_norm_actions_to_latents(self, norm_actions)
        latents, tokens = self.quantizer(encoded.latents)
        return AutoencoderTokenization(latents=latents, token_ids=tokens)

    def _tokens_to_latents(self, tokens: Tensor) -> Tensor:
        if int(tokens.shape[1]) != self.latent_horizon:
            raise ValueError(
                f"token length mismatch: expected {self.latent_horizon}, got {int(tokens.shape[1])}."
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
            tokens,
            token_horizon=self.latent_horizon,
            device=device,
        )
        latents = self._tokens_to_latents(token_tensor)
        return self.decode(latents, decode_keep_k=token_lens, **kwargs)
