# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for ActionCodecAutoencoder."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.acodec import (
    ActionCodecAutoencoder,
    ActionCodecAutoencoderConfig,
)


def _tiny_config(**overrides) -> ActionCodecAutoencoderConfig:
    base = {
        "action_dim": 3,
        "action_horizon": 8,
        "latent_horizon": 4,
        "emb_dim": 32,
        "latent_dim": 16,
        "head_dim": 8,
        "encoder_depth": 1,
        "decoder_depth": 1,
        "dropout": 0.0,
    }
    base.update(overrides)
    return ActionCodecAutoencoderConfig(**base)


def test_action_codec_autoencoder_forward_encode_decode() -> None:
    codec = ActionCodecAutoencoder(config=_tiny_config())
    actions = torch.randn(2, 8, 3)

    loss, metrics = codec({"action": actions})
    assert loss.ndim == 0
    assert "latent_mean_loss" in metrics
    assert "latent_variance_loss" in metrics

    latents = codec.encode(actions)
    assert latents.shape == (2, 4, 16)
    assert codec.decode(latents).shape == actions.shape


def test_action_codec_uses_official_style_perceiver_defaults() -> None:
    codec = ActionCodecAutoencoder(config=_tiny_config())

    assert codec.encoder.query_tokens.shape == (4, 32)
    assert codec.decoder.query_tokens.shape == (1, 32)
    assert codec.encoder.pos_scale == pytest.approx(0.1)
    assert codec.decoder.pos_scale == pytest.approx(0.1)

    block = codec.encoder.transformer.layers[0]
    assert isinstance(block.norm1, torch.nn.LayerNorm)
    assert block.norm1.eps == pytest.approx(1e-2)
    assert isinstance(block.norm2, torch.nn.LayerNorm)
    assert block.norm2.eps == pytest.approx(1e-2)
    assert isinstance(block.mlp.act, torch.nn.GELU)
    assert hasattr(block.attn_scale, "scale")
    assert hasattr(block.mlp_scale, "scale")


def test_action_codec_rejects_shape_mismatches() -> None:
    codec = ActionCodecAutoencoder(config=_tiny_config())

    with pytest.raises(ValueError, match="action horizon mismatch"):
        codec.encode(torch.randn(2, 7, 3))
    with pytest.raises(ValueError, match="latent horizon mismatch"):
        codec.decode(torch.randn(2, 3, 16))


def test_action_codec_rejects_non_divisible_head_dim() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        ActionCodecAutoencoder(config=_tiny_config(emb_dim=30, head_dim=8))


def test_action_codec_autoencoder_requires_action_dim() -> None:
    with pytest.raises(ValueError, match="requires action_dim"):
        ActionCodecAutoencoder(config=_tiny_config(action_dim=None))
