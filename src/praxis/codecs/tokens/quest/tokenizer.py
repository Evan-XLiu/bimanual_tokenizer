"""QueST tokenizer implementation (finite-scalar quantization adaptation)."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from praxis.codecs.latents.quest import QueSTAutoencoder
from praxis.codecs.tokens.core import (
    AutoencoderTokenization,
    AutoencoderTokenizer,
)
from praxis.codecs.tokens.quest.config import QueSTTokenizerConfig
from praxis.nn.quantization.fsq import FiniteScalarQuantization


class QueSTTokenizer(AutoencoderTokenizer, QueSTAutoencoder):
    """QueST latent autoencoder with finite-scalar quantization."""

    config: QueSTTokenizerConfig

    def __init__(
        self, config: dict[str, Any] | QueSTTokenizerConfig | None = None, **kwargs
    ) -> None:
        if config is None:
            cfg = QueSTTokenizerConfig()
        elif isinstance(config, QueSTTokenizerConfig):
            cfg = config
        else:
            cfg = QueSTTokenizerConfig.from_dict(config)
        super().__init__(config=cfg, **kwargs)
        if self.latent_dim != len(self.config.fsq_levels):
            raise ValueError(
                "QueSTTokenizer latent_dim must match fsq_levels length; "
                f"got latent_dim={self.latent_dim}, "
                f"levels={len(self.config.fsq_levels)}."
            )
        self.quantizer = FiniteScalarQuantization(levels=list(self.config.fsq_levels))

    @property
    def codebook_size(self) -> int:
        return int(self.quantizer.codebook_size)

    @property
    def max_code_seq_len(self) -> int:
        return int(self.latent_horizon)

    def _tokenize_norm_actions(self, norm_actions: Tensor) -> AutoencoderTokenization:
        encoded = QueSTAutoencoder._encode_norm_actions_to_latents(self, norm_actions)
        quant, token_ids = self.quantizer(encoded.latents)
        return AutoencoderTokenization(latents=quant, token_ids=token_ids)

    def _tokens_to_latents(self, tokens: Tensor) -> Tensor:
        if tokens.shape[1] != self.latent_horizon:
            raise ValueError(
                f"token length mismatch: expected {self.latent_horizon}, got {tokens.shape[1]}."
            )
        return self.quantizer.indices_to_embedding(tokens)
