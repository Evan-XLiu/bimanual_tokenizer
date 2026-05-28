"""Tests for the shared autoencoder tokenizer lifecycle."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from praxis.codecs.latents.core import (
    Autoencoder,
    AutoencoderEncoding,
    BaseLatentCodec,
)
from praxis.codecs.tokens.core import AutoencoderTokenization, AutoencoderTokenizer


class _DummyLatentAutoencoder(Autoencoder):
    def __init__(self) -> None:
        super().__init__()
        self.action_dim = 2
        self.latent_horizon = 3
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def _encode_norm_actions_to_latents(self, norm_actions: Tensor):
        return AutoencoderEncoding(
            latents=norm_actions[:, : self.latent_horizon] + self.bias,
        )

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs) -> Tensor:
        del kwargs
        return latents - self.bias


class _DummyAutoencoderTokenizer(AutoencoderTokenizer, _DummyLatentAutoencoder):
    def _tokenize_norm_actions(self, norm_actions: Tensor) -> AutoencoderTokenization:
        encoded = _DummyLatentAutoencoder._encode_norm_actions_to_latents(
            self, norm_actions
        )
        return AutoencoderTokenization(
            latents=encoded.latents + 10.0,
            token_ids=torch.zeros(
                (norm_actions.shape[0], self.latent_horizon),
                dtype=torch.long,
                device=norm_actions.device,
            ),
        )

    @property
    def codebook_size(self) -> int:
        return 8

    @property
    def max_code_seq_len(self) -> int:
        return self.latent_horizon

    def _tokens_to_latents(self, tokens: Tensor) -> Tensor:
        if int(tokens.shape[1]) != self.latent_horizon:
            raise ValueError(
                f"token length mismatch: expected {self.latent_horizon}, got {int(tokens.shape[1])}."
            )
        return tokens.unsqueeze(-1).expand(-1, -1, self.action_dim).float()


def test_autoencoder_tokenizer_owns_common_lifecycle():
    tok = _DummyAutoencoderTokenizer()
    assert isinstance(tok, Autoencoder)
    assert isinstance(tok, BaseLatentCodec)
    actions = torch.randn(2, 3, 2)

    loss, metrics = tok({"action": actions})
    assert loss.ndim == 0
    assert metrics == {}

    encoded = tok.tokenize_with_latents(actions)
    latents = encoded.latents
    tokens = encoded.token_ids
    assert latents.shape == (2, 3, 2)
    assert tokens.shape == (2, 3)
    assert torch.allclose(tok.encode(actions), actions[:, :3])
    assert not torch.allclose(latents, tok.encode(actions))
    assert tok.tokenize(actions).shape == (2, 3)
    assert tok.autoencode(actions).shape == actions.shape


def test_autoencoder_tokenizer_keeps_prefix_decode_opt_in():
    tok = _DummyAutoencoderTokenizer()
    assert tok.supports_prefix_decode is False

    exact = torch.zeros((2, 3), dtype=torch.long)
    assert tok.detokenize(exact).shape == (2, 3, 2)

    prefix = torch.zeros((2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="token length mismatch"):
        tok.detokenize(prefix)
