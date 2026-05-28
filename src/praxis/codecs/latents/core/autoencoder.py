"""Shared lifecycle for continuous action-latent autoencoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from praxis.codecs.latents.core.base import BaseLatentCodec
from praxis.codecs.latents.core.config import (
    LatentRegularizationConfig,
    LatentSpaceConfig,
)
from praxis.codecs.latents.core.regularization import latent_regularization_terms
from praxis.codecs.latents.core.training import (
    masked_autoencoder_loss,
    split_decay_param_groups,
)


@dataclass(frozen=True, kw_only=True)
class AutoencoderEncoding:
    """Latents and optional auxiliary losses from an encoder path."""

    latents: Tensor
    loss_terms: tuple[Tensor, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)


class Autoencoder(BaseLatentCodec):
    """Base class for continuous action-latent autoencoders.

    Subclasses own their encoder and decoder implementations. This base owns
    the shared normalized-action training and encode/decode lifecycle.
    """

    def _encode_norm_actions_to_latents(
        self, norm_actions: Tensor
    ) -> AutoencoderEncoding:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _encode_norm_actions_to_latents()."
        )

    def _decode_to_norm_actions(self, latents: Tensor, **kwargs: Any) -> Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _decode_to_norm_actions()."
        )

    def _decay_weight_decay(self) -> float | None:
        return None

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        actions = batch[self.action_key]
        norm_actions = self._normalize_action_tensor(actions)
        encoded = self._encode_norm_actions_to_latents(norm_actions)
        latent_space = getattr(getattr(self, "config", None), "latent_space", None)
        if latent_space is None:
            latent_space = LatentSpaceConfig(
                regularization=LatentRegularizationConfig(enabled=False)
            )
        if not isinstance(latent_space, LatentSpaceConfig):
            raise TypeError(
                f"{self.__class__.__name__}.config.latent_space must be a LatentSpaceConfig."
            )

        noise_std = latent_space.decoder_noise_std
        apply_decoder_noise = self.training and noise_std > 0.0
        decode_latents = (
            encoded.latents + torch.randn_like(encoded.latents) * noise_std
            if apply_decoder_noise
            else encoded.latents
        )
        recon = self._decode_to_norm_actions(decode_latents)
        regularization_terms, regularization_metrics = latent_regularization_terms(
            encoded.latents,
            latent_space.regularization,
        )
        raw_loss = F.mse_loss(recon, norm_actions, reduction="none")
        primary_loss = masked_autoencoder_loss(
            raw_loss,
            batch.get(f"{self.action_key}_is_pad"),
        )
        loss = primary_loss + sum(encoded.loss_terms) + sum(regularization_terms)
        metrics = dict(encoded.metrics)
        metrics.update(regularization_metrics)
        if apply_decoder_noise:
            metrics["latent_decoder_noise_std"] = noise_std
        return loss, metrics

    def encode(self, actions: Tensor) -> Tensor:
        norm_actions = self._normalize_action_tensor(actions)
        return self._encode_norm_actions_to_latents(norm_actions).latents

    def decode(self, latents: Tensor, **kwargs: Any) -> Tensor:
        norm_actions = self._decode_to_norm_actions(latents, **kwargs)
        return self._unnormalize_action_tensor(norm_actions)

    def autoencode(self, actions: Tensor, **kwargs: Any) -> Tensor:
        latents = self.encode(actions)
        return self.decode(latents, **kwargs)

    def get_optim_params(self) -> list[dict]:
        return split_decay_param_groups(
            self,
            decay_weight_decay=self._decay_weight_decay(),
            no_decay_weight_decay=0.0,
        )
