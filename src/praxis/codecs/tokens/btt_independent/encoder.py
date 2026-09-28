# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Independent-arm continuous encoder for the BTT sharing ablation."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from praxis.codecs.tokens.btt.modeling import (
    CausalTemporalDownsampler,
    SharedCausalSelfAttention,
)
from praxis.codecs.tokens.btt_independent.config import BTTIndependentConfig
from praxis.codecs.tokens.btt_independent.modeling import (
    IndependentCausalCrossAttention,
)
from praxis.nn.embeddings.positional import SinusoidPosEmbed


class BTTIndependentEncoder(nn.Module):
    """Independent arm dynamics with bidirectional, untied inter-arm coordination."""

    config: BTTIndependentConfig

    def __init__(
        self, config: dict[str, Any] | BTTIndependentConfig | None = None
    ) -> None:
        super().__init__()
        if config is None:
            cfg = BTTIndependentConfig()
        elif isinstance(config, BTTIndependentConfig):
            cfg = config
        else:
            cfg = BTTIndependentConfig.from_dict(config)
        self.config = cfg

        self.register_buffer(
            "right_arm_sign",
            torch.tensor(cfg.right_arm_sign, dtype=torch.float32),
            persistent=True,
        )
        for arm in ("left", "right"):
            for suffix in ("min", "max"):
                key = f"{arm}_action_{suffix}"
                self.register_buffer(
                    key,
                    torch.tensor(getattr(cfg, key), dtype=torch.float32),
                    persistent=True,
                )
        # Separate instances: no weights, biases or LayerNorm affine parameters are tied.
        self.left_arm_encoder = CausalTemporalDownsampler(
            arm_dim=cfg.arm_dim,
            model_dim=cfg.model_dim,
            kernel_sizes=cfg.conv_kernel_sizes,
            strides=cfg.conv_strides,
            expected_horizon=cfg.latent_horizon_per_arm,
        )
        self.right_arm_encoder = CausalTemporalDownsampler(
            arm_dim=cfg.arm_dim,
            model_dim=cfg.model_dim,
            kernel_sizes=cfg.conv_kernel_sizes,
            strides=cfg.conv_strides,
            expected_horizon=cfg.latent_horizon_per_arm,
        )
        # Match QueST exactly: fixed interleaved sin/cos positions with the
        # default temperature of 10_000, added after temporal downsampling.
        self.temporal_position = SinusoidPosEmbed(
            cfg.model_dim, max_shape=[cfg.action_horizon]
        )
        self.left_self_attention = SharedCausalSelfAttention(
            cfg.model_dim, cfg.num_heads, cfg.dropout
        )
        self.right_self_attention = SharedCausalSelfAttention(
            cfg.model_dim, cfg.num_heads, cfg.dropout
        )
        self.cross_attention = IndependentCausalCrossAttention(
            cfg.model_dim, cfg.num_heads, cfg.dropout
        )

    @property
    def latent_horizon(self) -> int:
        """Total number of continuous positions after concatenating both arms."""

        return 2 * self.config.latent_horizon_per_arm

    def canonicalize_right_arm(self, right_actions: Tensor) -> Tensor:
        """Map physical right-arm actions into the shared left-arm convention."""

        self._validate_arm_shape(right_actions)
        sign = self.right_arm_sign.to(
            device=right_actions.device, dtype=right_actions.dtype
        )
        return right_actions * sign

    def decanonicalize_right_arm(self, canonical_actions: Tensor) -> Tensor:
        """Map canonical right-arm actions back to the physical convention."""

        # A sign-only convention with entries in {-1, 1} is self-inverse.
        return self.canonicalize_right_arm(canonical_actions)

    def split_and_canonicalize(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Return physical left and canonicalized right single-arm trajectories."""

        self._validate_action_shape(actions)
        left, right = actions.split(self.config.arm_dim, dim=-1)
        return left, self.canonicalize_right_arm(right)

    def normalize_arm_actions(
        self,
        canonical_actions: Tensor,
        *,
        arm: Literal["left", "right"],
    ) -> Tensor:
        """Map one canonical arm to ``[-1, 1]`` with the selected arm joint ranges."""

        self._validate_arm_shape(canonical_actions)
        lower = self._arm_bound(arm, "min").to(
            device=canonical_actions.device, dtype=canonical_actions.dtype
        )
        upper = self._arm_bound(arm, "max").to(
            device=canonical_actions.device, dtype=canonical_actions.dtype
        )
        normalized = 2.0 * (canonical_actions - lower) / (upper - lower) - 1.0
        if self.config.clamp_normalized_actions:
            normalized = normalized.clamp(-1.0, 1.0)
        return normalized

    def unnormalize_arm_actions(
        self,
        normalized_actions: Tensor,
        *,
        arm: Literal["left", "right"],
    ) -> Tensor:
        """Invert the selected arm affine map for one canonical arm."""

        self._validate_arm_shape(normalized_actions)
        lower = self._arm_bound(arm, "min").to(
            device=normalized_actions.device, dtype=normalized_actions.dtype
        )
        upper = self._arm_bound(arm, "max").to(
            device=normalized_actions.device, dtype=normalized_actions.dtype
        )
        return (normalized_actions + 1.0) * (upper - lower) / 2.0 + lower

    def split_canonicalize_and_normalize(
        self, actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Prepare both arms using convention-first, arm-specific normalization."""

        left, canonical_right = self.split_and_canonicalize(actions)
        return (
            self.normalize_arm_actions(left, arm="left"),
            self.normalize_arm_actions(canonical_right, arm="right"),
        )

    def unnormalize_and_decanonicalize(
        self, normalized_left: Tensor, normalized_right: Tensor
    ) -> Tensor:
        """Restore one normalized arm pair to the physical 14-D convention."""

        left = self.unnormalize_arm_actions(normalized_left, arm="left")
        canonical_right = self.unnormalize_arm_actions(normalized_right, arm="right")
        right = self.decanonicalize_right_arm(canonical_right)
        return torch.cat((left, right), dim=-1)

    def encode_single_arm_dynamics(
        self,
        arm_actions: Tensor,
        *,
        arm: Literal["left", "right"],
    ) -> Tensor:
        """Encode one canonical arm using its own temporal/self-attention path."""

        self._validate_arm_name(arm)
        features = getattr(self, f"{arm}_arm_encoder")(arm_actions)
        features = features + self.temporal_position(
            [features.shape[1]], device=features.device, dtype=features.dtype
        )
        return getattr(self, f"{arm}_self_attention")(features)

    def forward(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        """Return separate coordinated features ``(C_L, C_R)``."""

        left, canonical_right = self.split_canonicalize_and_normalize(actions)
        left_single = self.encode_single_arm_dynamics(left, arm="left")
        right_single = self.encode_single_arm_dynamics(canonical_right, arm="right")
        return self.cross_attention(left_single, right_single)

    def encode_concatenated(self, actions: Tensor) -> Tensor:
        """Return ``Concat(C_L, C_R, dim=1)`` with 16 continuous positions."""

        left, right = self(actions)
        return torch.cat((left, right), dim=1)

    @staticmethod
    def _validate_arm_name(arm: str) -> None:
        if arm not in ("left", "right"):
            raise ValueError(f"Expected left or right arm, got {arm!r}.")

    def _arm_bound(self, arm: str, suffix: str) -> Tensor:
        self._validate_arm_name(arm)
        return getattr(self, f"{arm}_action_{suffix}")

    def _validate_action_shape(self, actions: Tensor) -> None:
        expected = f"[B, {self.config.action_horizon}, {self.config.action_dim}]"
        if (
            actions.ndim != 3
            or actions.shape[1] != self.config.action_horizon
            or actions.shape[2] != self.config.action_dim
        ):
            raise ValueError(
                f"BTTIndependent expects bimanual actions shaped {expected}, "
                f"got {tuple(actions.shape)}."
            )

    def _validate_arm_shape(self, actions: Tensor) -> None:
        if actions.ndim != 3 or actions.shape[-1] != self.config.arm_dim:
            raise ValueError(
                "BTTIndependent expects single-arm actions shaped "
                f"[B, T, {self.config.arm_dim}], got {tuple(actions.shape)}."
            )
