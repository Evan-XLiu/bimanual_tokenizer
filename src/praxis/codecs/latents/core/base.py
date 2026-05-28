"""Base class for continuous action-latent codecs."""

from __future__ import annotations

from abc import ABC
from typing import Any

from torch import Tensor

from praxis.codecs.core import BaseCodec


class BaseLatentCodec(BaseCodec, ABC):
    """Base class for codecs whose code representation is continuous."""

    @property
    def max_code_seq_len(self) -> int | None:
        latent_horizon = getattr(self, "latent_horizon", None)
        if latent_horizon is None:
            return super().max_code_seq_len
        return int(latent_horizon)

    def encode(self, actions: Tensor) -> Tensor:
        """Encode continuous actions to continuous latent representations."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement encode()."
        )

    def decode(self, latents: Tensor, **kwargs: Any) -> Tensor:
        """Decode continuous latent representations back to actions."""
        del kwargs
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement decode()."
        )
