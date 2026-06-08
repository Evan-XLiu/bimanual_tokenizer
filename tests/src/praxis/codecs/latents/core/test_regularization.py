# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared latent regularization helpers."""

from __future__ import annotations

import pytest
import torch

from praxis.codecs.latents.core import (
    LatentRegularizationConfig,
    LatentSpaceConfig,
    latent_regularization_terms,
)


def test_latent_regularization_terms_report_weighted_losses() -> None:
    latents = torch.randn(4, 3, 2)
    cfg = LatentRegularizationConfig(
        mean_weight=0.5,
        variance_weight=0.25,
        covariance_weight=0.125,
        norm_weight=0.0625,
    )

    losses, metrics = latent_regularization_terms(latents, cfg)

    assert len(losses) == 4
    assert all(loss.ndim == 0 for loss in losses)
    assert "latent_mean_loss" in metrics
    assert "latent_mean_loss_weighted" in metrics
    assert "latent_variance_loss" in metrics
    assert "latent_covariance_loss" in metrics
    assert "latent_norm_loss" in metrics


def test_latent_regularization_can_be_disabled() -> None:
    losses, metrics = latent_regularization_terms(
        torch.randn(2, 3, 4),
        LatentRegularizationConfig(enabled=False),
    )

    assert losses == ()
    assert metrics == {}


def test_latent_regularization_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="mean_weight"):
        LatentRegularizationConfig(mean_weight=-1.0)


def test_latent_space_config_builds_nested_regularization() -> None:
    cfg = LatentSpaceConfig.from_dict(
        {
            "regularization": {"mean_weight": 0.5},
            "decoder_noise_std": 0.1,
        }
    )

    assert cfg.regularization.mean_weight == 0.5
    assert cfg.decoder_noise_std == 0.1


def test_latent_space_config_rejects_negative_decoder_noise() -> None:
    with pytest.raises(ValueError, match="decoder_noise_std"):
        LatentSpaceConfig(decoder_noise_std=-0.1)
