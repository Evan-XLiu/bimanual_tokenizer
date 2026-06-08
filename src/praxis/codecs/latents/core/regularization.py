# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Latent-prior regularization for standalone continuous codecs."""

from __future__ import annotations

import torch
from torch import Tensor

from praxis.codecs.latents.core.config import LatentRegularizationConfig


def latent_regularization_terms(
    latents: Tensor,
    config: LatentRegularizationConfig | None,
) -> tuple[tuple[Tensor, ...], dict[str, float]]:
    """Return weighted latent regularization losses and scalar diagnostics."""
    if config is None or not config.enabled:
        return (), {}
    if latents.ndim < 2:
        raise ValueError(
            f"latents must have at least batch and feature dims, got shape={tuple(latents.shape)}."
        )

    flat = latents.reshape(-1, latents.shape[-1])
    mean = flat.mean(dim=0)
    variance = flat.var(dim=0, unbiased=False)

    mean_loss = mean.square().mean()
    variance_loss = (variance - 1.0).square().mean()
    norm_loss = (flat.square().mean(dim=-1) - 1.0).square().mean()

    losses: list[Tensor] = []
    metrics = {
        "latent_mean_abs": float(mean.abs().mean().detach().item()),
        "latent_variance_mean": float(variance.mean().detach().item()),
    }
    _append_weighted_loss(
        losses,
        metrics,
        name="latent_mean_loss",
        loss=mean_loss,
        weight=config.mean_weight,
    )
    _append_weighted_loss(
        losses,
        metrics,
        name="latent_variance_loss",
        loss=variance_loss,
        weight=config.variance_weight,
    )
    _append_weighted_loss(
        losses,
        metrics,
        name="latent_norm_loss",
        loss=norm_loss,
        weight=config.norm_weight,
    )
    if config.covariance_weight > 0.0:
        covariance_loss = _offdiag_covariance_loss(flat, mean)
        _append_weighted_loss(
            losses,
            metrics,
            name="latent_covariance_loss",
            loss=covariance_loss,
            weight=config.covariance_weight,
        )

    return tuple(losses), metrics


def _append_weighted_loss(
    losses: list[Tensor],
    metrics: dict[str, float],
    *,
    name: str,
    loss: Tensor,
    weight: float,
) -> None:
    if weight <= 0.0:
        return
    weighted = loss * float(weight)
    losses.append(weighted)
    metrics[name] = float(loss.detach().item())
    metrics[f"{name}_weighted"] = float(weighted.detach().item())


def _offdiag_covariance_loss(flat: Tensor, mean: Tensor) -> Tensor:
    if flat.shape[0] <= 1:
        return flat.new_zeros(())
    centered = flat - mean
    covariance = centered.T @ centered / float(flat.shape[0])
    offdiag = covariance - torch.diag_embed(torch.diagonal(covariance))
    return offdiag.square().mean()
