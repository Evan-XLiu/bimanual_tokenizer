# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the trainable X-Tokenizer latent codec."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from praxis.codecs.core import dataclass_config_to_dict
from praxis.codecs.latents.core import build_latent_autoencoder_config


@dataclass
class XTokenizerModelConfig:
    """X-Tokenizer model geometry and pretraining controls.

    The Bridge training path consumes absolute action chunks shaped
    ``[batch, action_horizon, action_dim]``. The encoder compresses the time
    axis to ``action_horizon // compression_ratio`` latent slots, and the RVQ
    emits ``num_quantizers`` token ids per slot.
    """

    # Factory identity. These select the registered latent codec implementation;
    # they do not change the network geometry.
    name: str = "xtokenizer"
    kind: str = "latents"

    # Optional checkpoint consumed by the codec factory. ``None`` constructs a
    # new model; a path loads weights and their saved model configuration.
    pretrained_path: str | None = None

    # Action/sample geometry
    # Number of channels in one action step. The dataset-aware trainer fills
    # this from the canonical action layout when it is left unset.
    action_dim: int | None = None

    # Number of future action steps reconstructed by one sample.
    action_horizon: int = 32

    # Largest chunk length supported by learned action/latent position tables.
    # The Bridge recipe trains at 32 steps; the paper supports chunks up to 64.
    max_action_horizon: int = 64

    # Temporal compression factor. This controls the number of latent queries,
    # not a hard local-attention window: every query can attend to the full chunk.
    compression_ratio: int = 4

    # Named, absolute action-channel semantics (for example 3D position, 6D
    # rotation, and gripper). It determines which channels use scalar or SO(3)
    # reconstruction losses.
    action_layout: dict[str, Any] | None = None

    # Encoder/decoder dimensions
    # Transformer token width for actions, state anchors, embodiment embeddings,
    # latent queries, decoder queries, and the MAM head.
    hidden_dim: int = 1024

    # Width of each continuous encoder latent and each RVQ codeword.
    latent_dim: int = 1024

    # Paper encoder geometry: 12 temporal layers with eight attention heads.
    encoder_num_heads: int = 8
    encoder_depth: int = 12

    # Paper decoder geometry: four temporal layers with eight attention heads.
    decoder_num_heads: int = 8
    decoder_depth: int = 4

    # Paper MAM head geometry: two layers with four attention heads.
    mam_num_heads: int = 4
    mam_depth: int = 2

    # Paper next-frame predictor geometry: two layers with four attention heads.
    prediction_num_heads: int = 4
    prediction_depth: int = 2

    # Attention, MLP, and residual dropout used during training.
    dropout: float = 0.1

    # Semantic residual quantization
    # Number of entries in each residual-quantizer codebook.
    codebook_size: int = 2048

    # Residual quantization levels per latent slot. The token tensor therefore
    # has shape ``[batch, latent_horizon, num_quantizers]``.
    num_quantizers: int = 4

    # EMA retention for codebook counts and sums; codebooks are not updated by
    # ordinary optimizer gradients.
    ema_decay: float = 0.8

    # Codes with an EMA assignment count below this value are reinitialized
    # from valid batch latents. Set to zero to disable dead-code replacement.
    dead_code_threshold: int = 2

    # Initialize every residual codebook from its first valid residual batch.
    kmeans_init: bool = True

    # Paper initialization runs 100 k-means iterations per residual level.
    kmeans_iters: int = 100

    # Embodiment and offline vision-language conditioning
    # Size of the robot-type embedding table. Bridge currently uses robot id 0.
    embodiment_vocab_size: int = 1024

    # Width of cached Qwen features used by alignment and next-frame prediction.
    vl_feature_dim: int = 3584

    # Independent classifier-free corruption probabilities from Appendix A.1.
    state_dropout_probability: float = 0.2
    embodiment_dropout_probability: float = 0.2
    embodiment_none_probability: float = 0.1

    # Objective weights
    # L1 reconstruction weight for non-rotation channels such as position and
    # gripper. Named 6D rotation channels are excluded from this term.
    reconstruction_weight: float = 1.0

    # Mean commitment loss over all RVQ residual levels. This pulls encoder
    # latents toward their selected (EMA-updated) codewords.
    commitment_weight: float = 0.25

    # SO(3) geodesic reconstruction weight for the named 6D rotation channels.
    rotation_weight: float = 0.2

    # Optional temporal-frequency reconstruction loss. Zero disables it.
    dct_weight: float = 0.5

    # Optional first-difference reconstruction loss on non-rotation channels.
    smoothness_weight: float = 0.3

    # Cross-entropy weight for predicting masked top-level (q0) RVQ ids.
    mam_weight: float = 0.1

    # Weight for the mean of global and slot-level symmetric InfoNCE between
    # pre-quantization action latents and aligned cached VL features.
    alignment_weight: float = 0.5

    # L1 weight for predicting the detached next-frame VL feature from pooled
    # quantized action latents.
    prediction_weight: float = 0.2

    # Initial learnable cosine-logit multiplier for InfoNCE. A scale of 10 is
    # equivalent to temperature 0.1 in the usual logits / temperature notation.
    infonce_scale_init: float = 10.0

    # Masked-action-modeling schedule
    # Probability of selecting each valid q0 latent-slot representation for
    # BERT-style 80/10/10 corruption.
    mam_mask_probability: float = 0.15

    # MAM remains disabled until this epoch threshold is reached.
    mam_warmup_epochs: int = 10

    # Additional global-step threshold for MAM; both warm-up gates must pass.
    mam_warmup_steps: int = 0

    def __post_init__(self) -> None:
        for name in (
            "action_horizon",
            "max_action_horizon",
            "compression_ratio",
            "hidden_dim",
            "latent_dim",
            "encoder_num_heads",
            "encoder_depth",
            "decoder_num_heads",
            "decoder_depth",
            "mam_num_heads",
            "mam_depth",
            "prediction_num_heads",
            "prediction_depth",
            "codebook_size",
            "num_quantizers",
            "kmeans_iters",
            "embodiment_vocab_size",
            "vl_feature_dim",
        ):
            setattr(self, name, _positive_int(getattr(self, name), name=name))

        if self.action_dim is not None:
            self.action_dim = _positive_int(self.action_dim, name="action_dim")
            (
                self.action_layout,
                self._position_slices,
                self._rotation_slices,
                self._scalar_channel_indices,
            ) = _resolve_action_layout(self.action_dim, self.action_layout)
        elif self.action_layout is not None:
            raise ValueError("action_layout requires action_dim to be configured.")
        else:
            self._position_slices = ()
            self._rotation_slices = ()
            self._scalar_channel_indices = ()

        if self.action_horizon % self.compression_ratio != 0:
            raise ValueError(
                "action_horizon must be divisible by compression_ratio, "
                f"got {self.action_horizon} and {self.compression_ratio}."
            )
        if self.max_action_horizon % self.compression_ratio != 0:
            raise ValueError(
                "max_action_horizon must be divisible by compression_ratio, "
                f"got {self.max_action_horizon} and {self.compression_ratio}."
            )
        if self.action_horizon > self.max_action_horizon:
            raise ValueError(
                "action_horizon must not exceed max_action_horizon, "
                f"got {self.action_horizon} and {self.max_action_horizon}."
            )

        self.dropout = float(self.dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}.")

        self.ema_decay = float(self.ema_decay)
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {self.ema_decay}.")

        self.dead_code_threshold = _non_negative_int(
            self.dead_code_threshold, name="dead_code_threshold"
        )
        if not isinstance(self.kmeans_init, bool):
            raise ValueError(
                f"kmeans_init must be a boolean, got {self.kmeans_init!r}."
            )

        for name in (
            "state_dropout_probability",
            "embodiment_dropout_probability",
            "embodiment_none_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
            setattr(self, name, value)

        for name in (
            "reconstruction_weight",
            "commitment_weight",
            "rotation_weight",
            "dct_weight",
            "smoothness_weight",
            "mam_weight",
            "alignment_weight",
            "prediction_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}.")
            setattr(self, name, value)

        self.mam_mask_probability = float(self.mam_mask_probability)
        if not 0.0 <= self.mam_mask_probability <= 1.0:
            raise ValueError(
                "mam_mask_probability must be in [0, 1], "
                f"got {self.mam_mask_probability}."
            )

        self.infonce_scale_init = float(self.infonce_scale_init)
        if (
            not math.isfinite(self.infonce_scale_init)
            or not 0.0 < self.infonce_scale_init <= 100.0
        ):
            raise ValueError(
                "infonce_scale_init must be finite and in (0, 100], "
                f"got {self.infonce_scale_init}."
            )

        for name in ("mam_warmup_epochs", "mam_warmup_steps"):
            setattr(
                self,
                name,
                _non_negative_int(getattr(self, name), name=name),
            )

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> XTokenizerModelConfig:
        return build_latent_autoencoder_config(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)

    @property
    def position_slice(self) -> slice | None:
        return self._position_slices[0] if self._position_slices else None

    @property
    def position_slices(self) -> tuple[slice, ...]:
        return self._position_slices

    @property
    def rotation_slice(self) -> slice | None:
        return self._rotation_slices[0] if self._rotation_slices else None

    @property
    def rotation_slices(self) -> tuple[slice, ...]:
        return self._rotation_slices

    @property
    def scalar_channel_indices(self) -> tuple[int, ...]:
        return self._scalar_channel_indices


def _resolve_action_layout(
    action_dim: int,
    value: dict[str, Any] | None,
) -> tuple[dict[str, Any], tuple[slice, ...], tuple[slice, ...], tuple[int, ...]]:
    if value is None:
        value = _default_action_layout(action_dim)
    if not isinstance(value, dict) or set(value) != {
        "convention",
        "dof_dims",
        "dof_order",
    }:
        raise ValueError(
            "action_layout must contain convention, dof_dims, and dof_order."
        )
    if value["convention"] != "absolute":
        raise ValueError("action_layout.convention must be 'absolute'.")
    raw_dims = value["dof_dims"]
    raw_order = value["dof_order"]
    if not isinstance(raw_dims, dict) or not isinstance(raw_order, (list, tuple)):
        raise ValueError("action_layout dof_dims/dof_order are malformed.")
    if (
        not raw_order
        or not all(isinstance(name, str) and name for name in raw_order)
        or len(raw_order) != len(set(raw_order))
        or set(raw_order) != set(raw_dims)
    ):
        raise ValueError("action_layout.dof_order must list every DoF exactly once.")
    dims: dict[str, int] = {}
    for name, raw_dim in raw_dims.items():
        if not isinstance(name, str) or not name:
            raise ValueError("action_layout DoF names must be non-empty strings.")
        dims[name] = _positive_int(raw_dim, name=f"action_layout.dof_dims[{name!r}]")
    if sum(dims.values()) != action_dim:
        raise ValueError(
            "action_layout dimensions must sum to action_dim, got "
            f"{sum(dims.values())} and {action_dim}."
        )

    slices: dict[str, slice] = {}
    offset = 0
    order = tuple(raw_order)
    for name in order:
        width = dims[name]
        slices[name] = slice(offset, offset + width)
        offset += width
    position_names = [
        name
        for name in order
        if name.endswith("_ee_cartesian_pos")
        or name.endswith("_ee_cartesian_pos_relative")
    ]
    rotation_names = [
        name
        for name in order
        if name.endswith("_ee_rotation_6D") or name.endswith("_ee_rotation_6D_relative")
    ]
    if bool(position_names) != bool(rotation_names):
        raise ValueError(
            "action_layout must define both named position and rotation DoFs."
        )
    if len(position_names) != len(rotation_names):
        raise ValueError(
            "action_layout must define matching named position/rotation groups."
        )
    position_slices = tuple(slices[name] for name in position_names)
    rotation_slices = tuple(slices[name] for name in rotation_names)
    if any(segment.stop - segment.start != 3 for segment in position_slices):
        raise ValueError("action_layout named position DoFs must have dimension 3.")
    if any(segment.stop - segment.start != 6 for segment in rotation_slices):
        raise ValueError("action_layout named rotation DoFs must have dimension 6.")
    rotation_indices = {
        index
        for segment in rotation_slices
        for index in range(segment.start, segment.stop)
    }
    scalar_indices = tuple(
        index for index in range(action_dim) if index not in rotation_indices
    )
    canonical = {
        "convention": "absolute",
        "dof_dims": {name: dims[name] for name in order},
        "dof_order": list(order),
    }
    return canonical, position_slices, rotation_slices, scalar_indices


def _default_action_layout(action_dim: int) -> dict[str, Any]:
    if action_dim == 26:
        dims = {
            "follow_left_ee_cartesian_pos": 3,
            "follow_left_ee_rotation_6D": 6,
            "follow_right_ee_cartesian_pos": 3,
            "follow_right_ee_rotation_6D": 6,
            "follow_left_gripper": 1,
            "follow_right_gripper": 1,
            "base_velocity": 3,
            "lift_height": 1,
            "head_pitch_yaw": 2,
        }
        return {
            "convention": "absolute",
            "dof_dims": dims,
            "dof_order": list(dims),
        }
    if action_dim < 9:
        return {
            "convention": "absolute",
            "dof_dims": {"action": action_dim},
            "dof_order": ["action"],
        }
    dims = {
        "follow_left_ee_cartesian_pos": 3,
        "follow_left_ee_rotation_6D": 6,
    }
    remaining = action_dim - 9
    if remaining == 1:
        dims["follow_left_gripper"] = 1
    elif remaining > 1:
        dims["action_aux"] = remaining
    return {
        "convention": "absolute",
        "dof_dims": dims,
        "dof_order": list(dims),
    }


def _positive_int(value: Any, *, name: str) -> int:
    resolved = _integer(value, name=name)
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")
    return resolved


def _non_negative_int(value: Any, *, name: str) -> int:
    resolved = _integer(value, name=name)
    if resolved < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return resolved


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    if resolved != value:
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    return resolved
