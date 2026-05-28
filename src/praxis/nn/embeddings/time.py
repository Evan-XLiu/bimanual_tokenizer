"""Shared embeddings for scalar coordinates such as time or phase."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

__all__ = [
    "SinusoidalScalarEmbedding",
    "geometric_frequency_grid",
    "sinusoidal_scalar_features",
]

ChannelOrder = Literal["sin_cos", "cos_sin"]


def _validate_scalar_values(values: torch.Tensor) -> None:
    if values.ndim != 1:
        raise ValueError(f"values must be rank-1 [B], got shape={tuple(values.shape)}.")


def _validate_frequency_range(min_frequency: float, max_frequency: float) -> None:
    if min_frequency <= 0:
        raise ValueError(f"min_frequency must be positive, got {min_frequency}.")
    if max_frequency <= 0:
        raise ValueError(f"max_frequency must be positive, got {max_frequency}.")
    if min_frequency > max_frequency:
        raise ValueError(
            "min_frequency must be <= max_frequency, got "
            f"{min_frequency} and {max_frequency}."
        )


def geometric_frequency_grid(
    num_frequencies: int,
    *,
    min_frequency: float,
    max_frequency: float,
    endpoint: bool = True,
    descending: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return geometrically spaced positive frequencies."""
    if num_frequencies < 0:
        raise ValueError(
            f"num_frequencies must be non-negative, got {num_frequencies}."
        )
    if num_frequencies == 0:
        return torch.empty(0, device=device, dtype=dtype or torch.float32)
    _validate_frequency_range(min_frequency, max_frequency)

    target_dtype = dtype or torch.float32
    start = float(max_frequency if descending else min_frequency)
    end = float(min_frequency if descending else max_frequency)
    if num_frequencies == 1:
        return torch.full((1,), start, device=device, dtype=target_dtype)

    denominator = num_frequencies - 1 if endpoint else num_frequencies
    fractions = torch.arange(num_frequencies, device=device, dtype=target_dtype)
    fractions = fractions / float(denominator)
    log_start = math.log(start)
    log_end = math.log(end)
    return torch.exp(log_start + fractions * (log_end - log_start))


def sinusoidal_scalar_features(
    values: torch.Tensor,
    dim: int,
    *,
    min_frequency: float,
    max_frequency: float,
    angular_scale: float = 1.0,
    endpoint: bool = True,
    descending: bool = False,
    channel_order: ChannelOrder = "sin_cos",
    pad_odd_dim: bool = False,
) -> torch.Tensor:
    """Encode rank-1 scalar values with sin/cos features over a frequency grid."""
    _validate_scalar_values(values)
    if dim < 0:
        raise ValueError(f"dim must be non-negative, got {dim}.")
    if dim % 2 != 0 and not pad_odd_dim:
        raise ValueError(f"dim must be even when pad_odd_dim=False, got {dim}.")
    if channel_order not in ("sin_cos", "cos_sin"):
        raise ValueError(f"Unsupported channel_order {channel_order!r}.")

    half_dim = dim // 2
    frequencies = geometric_frequency_grid(
        half_dim,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        endpoint=endpoint,
        descending=descending,
        device=values.device,
        dtype=values.dtype if values.is_floating_point() else torch.float32,
    )
    angles = (
        values.to(dtype=frequencies.dtype).unsqueeze(-1)
        * frequencies.unsqueeze(0)
        * angular_scale
    )
    first = torch.sin(angles)
    second = torch.cos(angles)
    if channel_order == "cos_sin":
        first, second = second, first
    features = torch.cat([first, second], dim=-1)
    if dim % 2 != 0:
        features = torch.cat([features, torch.zeros_like(features[:, :1])], dim=-1)
    return features


class SinusoidalScalarEmbedding(nn.Module):
    """Project sinusoidal features for a rank-1 scalar coordinate."""

    def __init__(
        self,
        dim: int,
        feature_dim: int = 256,
        hidden_dim: int | None = None,
        *,
        min_frequency: float,
        max_frequency: float,
        angular_scale: float = 1.0,
        endpoint: bool = True,
        descending: bool = False,
        channel_order: ChannelOrder = "sin_cos",
        pad_odd_dim: bool = False,
        activation: Literal["silu", "mish"] = "silu",
        weight_init_style: Literal["xavier", "trunc_normal"] = "xavier",
        as_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim or dim)
        self.min_frequency = float(min_frequency)
        self.max_frequency = float(max_frequency)
        self.angular_scale = float(angular_scale)
        self.endpoint = bool(endpoint)
        self.descending = bool(descending)
        self.channel_order: ChannelOrder = channel_order
        self.pad_odd_dim = bool(pad_odd_dim)
        self.weight_init_style = weight_init_style
        self.as_tokens = bool(as_tokens)

        if self.feature_dim % 2 != 0 and not self.pad_odd_dim:
            raise ValueError(
                "feature_dim must be even when pad_odd_dim=False, got "
                f"{self.feature_dim}."
            )

        act: nn.Module
        if activation == "silu":
            act = nn.SiLU()
        elif activation == "mish":
            act = nn.Mish()
        else:
            raise ValueError(f"Unsupported activation {activation!r}.")

        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim, bias=True),
            act,
            nn.Linear(self.hidden_dim, self.dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(module.weight, std=0.02)
                else:
                    raise ValueError(
                        f"Unsupported weight init: {self.weight_init_style}"
                    )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def features(self, values: torch.Tensor) -> torch.Tensor:
        return sinusoidal_scalar_features(
            values,
            self.feature_dim,
            min_frequency=self.min_frequency,
            max_frequency=self.max_frequency,
            angular_scale=self.angular_scale,
            endpoint=self.endpoint,
            descending=self.descending,
            channel_order=self.channel_order,
            pad_odd_dim=self.pad_odd_dim,
        )

    @torch.compiler.disable
    def forward(self, values: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        projected = self.mlp(self.features(values))
        if self.as_tokens:
            return projected.unsqueeze(1).split(1, dim=0)
        return projected
