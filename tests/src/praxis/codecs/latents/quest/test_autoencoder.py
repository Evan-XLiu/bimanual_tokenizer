"""Smoke tests for QueSTAutoencoder."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.quest import QueSTAutoencoder, QueSTAutoencoderConfig


def _tiny_config(**overrides) -> QueSTAutoencoderConfig:
    base = {
        "action_dim": 3,
        "action_horizon": 8,
        "encoder_dim": 32,
        "encoder_nhead": 4,
        "encoder_nlayer": 1,
        "decoder_dim": 32,
        "decoder_nhead": 4,
        "decoder_nlayer": 1,
        "dropout": 0.0,
        "downsample_factor": 4,
        "use_causal_encoder": True,
        "use_causal_decoder": True,
        "latent_dim": 2,
    }
    base.update(overrides)
    return QueSTAutoencoderConfig(**base)


def test_quest_autoencoder_forward_encode_decode() -> None:
    codec = QueSTAutoencoder(config=_tiny_config())
    actions = torch.randn(2, 8, 3)

    loss, metrics = codec({"action": actions})
    assert loss.ndim == 0
    assert "latent_mean_loss" in metrics
    assert "latent_variance_loss" in metrics

    latents = codec.encode(actions)
    assert codec.max_code_seq_len == codec.latent_horizon
    assert latents.shape == (2, codec.latent_horizon, 2)
    assert codec.decode(latents).shape == actions.shape


def test_quest_autoencoder_requires_action_dim() -> None:
    with pytest.raises(ValueError, match="requires action_dim"):
        QueSTAutoencoder(config=_tiny_config(action_dim=None))
