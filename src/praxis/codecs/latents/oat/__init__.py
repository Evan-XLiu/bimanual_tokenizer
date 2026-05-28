"""OAT-family continuous action-latent autoencoder components."""

from praxis.codecs.latents.oat.autoencoder import OATAutoencoder
from praxis.codecs.latents.oat.config import OATAutoencoderConfig, OATModelConfig
from praxis.codecs.latents.oat.modeling import RegisterQueryEncoder, SinglePassDecoder
from praxis.codecs.latents.oat.schedule import (
    create_register_schedule_mask,
    is_power_of_two,
    pow2_register_schedule,
    powers_of_two,
    register_schedule_prefix_boundaries,
    unit_register_schedule,
)

__all__ = [
    "OATAutoencoder",
    "OATAutoencoderConfig",
    "OATModelConfig",
    "RegisterQueryEncoder",
    "SinglePassDecoder",
    "create_register_schedule_mask",
    "is_power_of_two",
    "pow2_register_schedule",
    "powers_of_two",
    "register_schedule_prefix_boundaries",
    "unit_register_schedule",
]
