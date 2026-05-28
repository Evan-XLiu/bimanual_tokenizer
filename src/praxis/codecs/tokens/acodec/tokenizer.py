"""Standalone ActionCodec-style tokenizer (VQ-first)."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from praxis.codecs.latents.acodec import ActionCodecAutoencoder
from praxis.codecs.tokens.acodec.config import ActionCodecTokenizerConfig
from praxis.codecs.tokens.core import (
    AutoencoderTokenization,
    AutoencoderTokenizer,
)
from praxis.nn.quantization.vq import VectorQuantization


class ActionCodecTokenizer(AutoencoderTokenizer, ActionCodecAutoencoder):
    """ActionCodec latent autoencoder with vector quantization."""

    config: ActionCodecTokenizerConfig

    def __init__(
        self,
        config: dict[str, Any] | ActionCodecTokenizerConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            cfg = ActionCodecTokenizerConfig()
        elif isinstance(config, ActionCodecTokenizerConfig):
            cfg = config
        else:
            cfg = ActionCodecTokenizerConfig.from_dict(config)
        super().__init__(config=cfg, **kwargs)
        self.quantizer = VectorQuantization(
            input_dim=self.latent_dim,
            codebook_size=self.config.vq_codebook_size,
            codebook_dim=self.config.vq_codebook_dim,
            commitment=self.config.vq_commitment_weight,
            decay=self.config.vq_decay,
            epsilon=self.config.vq_epsilon,
            threshold_ema_dead=self.config.vq_threshold_ema_dead_code,
            kmeans_init=self.config.vq_kmeans_init,
            kmeans_iters=self.config.vq_kmeans_iters,
        )

    @property
    def codebook_size(self) -> int:
        return int(self.quantizer.codebook_size)

    @property
    def max_code_seq_len(self) -> int:
        return int(self.latent_horizon)

    @property
    def block_schedule(self) -> tuple[int, ...]:
        # ActionCodec latents are decoded in one parallel block.
        return (int(self.latent_horizon),)

    def _tokenize_norm_actions(self, norm_actions: Tensor) -> AutoencoderTokenization:
        encoded = ActionCodecAutoencoder._encode_norm_actions_to_latents(
            self, norm_actions
        )
        quantized, token_ids, commitment_loss, codebook_loss = self.quantizer(
            encoded.latents
        )
        return AutoencoderTokenization(
            latents=quantized,
            token_ids=token_ids,
            loss_terms=(commitment_loss, codebook_loss),
            metrics={
                "vq_commitment_loss": float(commitment_loss.detach().item()),
                "vq_codebook_loss": float(codebook_loss.detach().item()),
            },
        )

    def _tokens_to_latents(self, tokens: Tensor) -> Tensor:
        if int(tokens.shape[1]) != self.latent_horizon:
            raise ValueError(
                f"token length mismatch: expected {self.latent_horizon}, got {int(tokens.shape[1])}."
            )
        return self.quantizer.indices_to_quantized(tokens)
