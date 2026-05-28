"""Flow-matching sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FlowSample:
    """One minRF/OpenPI-style training sample."""

    noisy_actions: Tensor
    target_velocity: Tensor
    timesteps: Tensor
    noise: Tensor


def sample_flow_training_point(
    actions: Tensor,
    *,
    sampling: str,
    beta_alpha: float,
    beta_beta: float,
    min_time: float,
) -> FlowSample:
    """Sample a flow point with ``t=1`` noise and ``t=0`` data."""
    batch_size = int(actions.shape[0])
    device = actions.device
    dtype = actions.dtype
    noise = torch.randn_like(actions)
    if sampling == "uniform":
        timesteps = torch.rand(batch_size, device=device, dtype=dtype)
    elif sampling == "beta":
        dist = torch.distributions.Beta(
            torch.tensor(float(beta_alpha), device=device, dtype=torch.float32),
            torch.tensor(float(beta_beta), device=device, dtype=torch.float32),
        )
        timesteps = dist.sample((batch_size,)).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported flow time sampling mode {sampling!r}.")
    min_time_t = torch.as_tensor(float(min_time), device=device, dtype=dtype)
    timesteps = timesteps * (1.0 - min_time_t) + min_time_t
    t = timesteps.view(batch_size, 1, 1)
    noisy_actions = t * noise + (1.0 - t) * actions
    return FlowSample(
        noisy_actions=noisy_actions,
        target_velocity=noise - actions,
        timesteps=timesteps,
        noise=noise,
    )
