"""Tests for the shared continuous action-latent autoencoder lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from praxis.codecs.core import BaseCodec
from praxis.codecs.latents.core import (
    Autoencoder,
    AutoencoderEncoding,
    BaseLatentCodec,
    LatentRegularizationConfig,
    LatentSpaceConfig,
)


class _DummyAutoencoder(Autoencoder):
    def __init__(self) -> None:
        super().__init__()
        self.latent_horizon = 3
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def _encode_norm_actions_to_latents(
        self, norm_actions: Tensor
    ) -> AutoencoderEncoding:
        return AutoencoderEncoding(
            latents=norm_actions + self.bias,
            metrics={"encoded": 1.0},
        )

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs) -> Tensor:
        del kwargs
        return latents - self.bias


def test_autoencoder_is_a_base_codec() -> None:
    codec = _DummyAutoencoder()

    assert isinstance(codec, BaseCodec)
    assert isinstance(codec, BaseLatentCodec)
    assert codec.variable_length_codes is False
    assert codec.max_code_seq_len == 3
    assert codec.supports_prefix_decode is False
    assert not hasattr(codec, "block_schedule")


def test_autoencoder_owns_common_lifecycle() -> None:
    codec = _DummyAutoencoder()
    actions = torch.randn(2, 3, 2)

    loss, metrics = codec({"action": actions})
    assert loss.ndim == 0
    assert metrics == {"encoded": 1.0}

    latents = codec.encode(actions)
    assert latents.shape == actions.shape
    assert codec.autoencode(actions).shape == actions.shape


def test_autoencoder_adds_standalone_latent_regularization() -> None:
    codec = _DummyAutoencoder()
    codec.config = SimpleNamespace(
        latent_space=LatentSpaceConfig(
            regularization=LatentRegularizationConfig(
                mean_weight=1.0,
                variance_weight=0.0,
            )
        )
    )
    actions = torch.ones(2, 3, 2)

    loss, metrics = codec({"action": actions})

    assert loss.item() > 0.0
    assert "latent_mean_loss" in metrics
    assert "latent_variance_loss" not in metrics
    assert "latent_mean_abs" in metrics


def test_autoencoder_decoder_noise_is_training_only(monkeypatch) -> None:
    codec = _DummyAutoencoder()
    codec.config = SimpleNamespace(
        latent_space=LatentSpaceConfig(
            regularization=LatentRegularizationConfig(enabled=False),
            decoder_noise_std=0.25,
        )
    )
    actions = torch.zeros(2, 3, 2)
    monkeypatch.setattr(torch, "randn_like", lambda x: torch.ones_like(x))

    codec.train()
    train_loss, train_metrics = codec({"action": actions})
    assert train_loss.item() == pytest.approx(0.25**2)
    assert train_metrics["latent_decoder_noise_std"] == 0.25

    codec.eval()
    eval_loss, eval_metrics = codec({"action": actions})
    assert eval_loss.item() == pytest.approx(0.0)
    assert "latent_decoder_noise_std" not in eval_metrics
