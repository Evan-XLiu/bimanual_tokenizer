# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared continuous action-latent codec interfaces."""

from praxis.codecs.latents.core.autoencoder import Autoencoder, AutoencoderEncoding
from praxis.codecs.latents.core.base import BaseLatentCodec
from praxis.codecs.latents.core.config import (
    LatentRegularizationConfig,
    LatentSpaceConfig,
    build_latent_autoencoder_config,
)
from praxis.codecs.latents.core.regularization import latent_regularization_terms
from praxis.codecs.latents.core.training import (
    masked_autoencoder_loss,
    split_decay_param_groups,
)

__all__ = [
    "Autoencoder",
    "AutoencoderEncoding",
    "BaseLatentCodec",
    "LatentSpaceConfig",
    "LatentRegularizationConfig",
    "build_latent_autoencoder_config",
    "latent_regularization_terms",
    "masked_autoencoder_loss",
    "split_decay_param_groups",
]
