"""Sampling helpers for flow-matching action experts."""

from __future__ import annotations

import torch
from torch import Tensor

from praxis.policies.core.action_expert.expert import ActionExpert
from praxis.policies.core.action_expert.prefix_cache import PrefixCache


@torch.no_grad()
def sample_actions_euler(
    action_expert: ActionExpert,
    *,
    cache: PrefixCache,
    action_dim: int,
    action_horizon: int,
    steps: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample continuous actions by Euler integration from noise to data."""
    device = cache.mask.device
    dtype = next(action_expert.parameters()).dtype
    batch_size = int(cache.mask.shape[0])
    x_t = torch.randn(
        (batch_size, int(action_horizon), int(action_dim)),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    dt = -1.0 / int(steps)
    time = torch.ones((batch_size,), device=device, dtype=dtype)
    for _ in range(int(steps)):
        velocity = action_expert(
            cache=cache,
            noisy_actions=x_t,
            timesteps=time,
        )
        x_t = x_t + velocity * dt
        time = time + dt
    return x_t
