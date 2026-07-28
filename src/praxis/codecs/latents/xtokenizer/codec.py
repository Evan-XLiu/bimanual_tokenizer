# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Trainable X-Tokenizer codec and semantic pretraining heads."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from praxis.codecs.latents.core import BaseLatentCodec, split_decay_param_groups
from praxis.codecs.latents.xtokenizer.action_space import (
    from_observation_relative_actions,
    to_observation_relative_actions,
)
from praxis.codecs.latents.xtokenizer.config import XTokenizerModelConfig
from praxis.codecs.latents.xtokenizer.modeling import (
    ActionDecoder,
    RelativeActionEncoder,
)
from praxis.codecs.latents.xtokenizer.objectives import (
    masked_action_l1,
    rotation_geodesic_loss,
    symmetric_global_local_infonce,
)
from praxis.codecs.latents.xtokenizer.quantizer import SemanticResidualQuantizer
from praxis.core.checkpoint_artifacts import collect_normalizer_stats
from praxis.core.normalizer import Normalize, Unnormalize
from praxis.core.types import NormMode

INFERENCE_ARTIFACT_FILENAME = "xtokenizer_inference.pt"
INFERENCE_MANIFEST_FILENAME = "manifest.json"

__all__ = [
    "INFERENCE_ARTIFACT_FILENAME",
    "INFERENCE_MANIFEST_FILENAME",
    "MAMHead",
    "NextFramePredictionHead",
    "XTokenizerCodec",
]


class MAMHead(nn.Module):
    """Predict masked top-level residual-quantizer IDs."""

    def __init__(self, config: XTokenizerModelConfig) -> None:
        super().__init__()
        self.codebook_size = int(config.codebook_size)
        self.latent_dim = int(config.latent_dim)
        self.latent_horizon = int(config.action_horizon // config.compression_ratio)
        self.mask_token = nn.Parameter(torch.empty(self.latent_dim))
        self.input_projection: nn.Module
        if config.latent_dim == config.hidden_dim:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(config.latent_dim, config.hidden_dim)
        self.position = nn.Embedding(16, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.mam_num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=config.mam_depth,
            norm=nn.LayerNorm(config.hidden_dim),
            enable_nested_tensor=False,
        )
        self.predictor = nn.Linear(config.hidden_dim, self.codebook_size)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(
        self,
        q0_representations: Tensor,
        q0_codebook: Tensor,
        masked_positions: Tensor,
        valid_mask: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        self._validate_input(
            q0_representations,
            q0_codebook,
            masked_positions,
            valid_mask,
        )

        corrupted = self._corrupt_mam_representations(
            q0_representations,
            masked_positions,
            q0_codebook=q0_codebook,
            generator=generator,
        )
        corrupted = corrupted.masked_fill(~valid_mask.unsqueeze(-1), 0)
        hidden = self.input_projection(corrupted)
        positions = torch.arange(
            q0_representations.shape[1],
            device=hidden.device,
        )
        hidden = hidden + self.position(positions).unsqueeze(0)
        hidden = self.context(
            hidden,
            src_key_padding_mask=_safe_padding_mask(~valid_mask),
            is_causal=False,
        )
        return self.predictor(hidden)

    def _validate_input(
        self,
        q0_representations: Tensor,
        q0_codebook: Tensor,
        masked_positions: Tensor,
        valid_mask: Tensor,
    ) -> None:
        expected_shape = (
            q0_representations.shape[0],
            self.latent_horizon,
            self.latent_dim,
        )
        if q0_representations.shape != expected_shape:
            raise ValueError(
                f"q0_representations must have shape [B, {self.latent_horizon}, "
                f"{self.latent_dim}], got {tuple(q0_representations.shape)}."
            )
        if q0_codebook.shape != (self.codebook_size, self.latent_dim):
            raise ValueError(
                "q0_codebook must have shape "
                f"[{self.codebook_size}, {self.latent_dim}]."
            )
        if (
            masked_positions.shape != valid_mask.shape
            or valid_mask.shape != (q0_representations.shape[:2])
        ):
            raise ValueError("MAM masks must match representation shape [B, M].")
        if masked_positions.dtype != torch.bool or valid_mask.dtype != torch.bool:
            raise ValueError("MAM masks must have boolean dtypes.")
        if bool((masked_positions & ~valid_mask).any()):
            raise ValueError("masked_positions must be a subset of valid_mask.")

    def _corrupt_mam_representations(
        self,
        representations: Tensor,
        selected: Tensor,
        *,
        q0_codebook: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Apply BERT-style 80/10/10 corruption to selected q0 slots."""
        draws = torch.rand(
            selected.shape,
            device=representations.device,
            generator=generator,
        )
        random_codes = torch.randint(
            0,
            q0_codebook.shape[0],
            selected.shape,
            device=representations.device,
            generator=generator,
        )
        mask_positions = selected & (draws < 0.8)
        random_positions = selected & (draws >= 0.8) & (draws < 0.9)
        random_replacements = F.embedding(
            random_codes,
            q0_codebook.detach(),
        ).to(dtype=representations.dtype)
        corrupted = torch.where(
            mask_positions.unsqueeze(-1),
            self.mask_token.to(dtype=representations.dtype),
            representations,
        )
        return torch.where(
            random_positions.unsqueeze(-1),
            random_replacements,
            corrupted,
        )


class NextFramePredictionHead(nn.Module):
    """Predict a detached next-frame VL feature from summed RVQ latents."""

    def __init__(
        self,
        latent_dim: int,
        vl_feature_dim: int,
        *,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        max_length: int = 16,
    ) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.position = nn.Embedding(max_length, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)
        self.projection = nn.Linear(latent_dim, vl_feature_dim)

    def forward(self, quantized: Tensor, valid_mask: Tensor) -> Tensor:
        self._validate_input(quantized, valid_mask)

        positions = torch.arange(quantized.shape[1], device=quantized.device)
        hidden = quantized.masked_fill(~valid_mask.unsqueeze(-1), 0)
        hidden = hidden + self.position(positions).unsqueeze(0)
        hidden = self.context(
            hidden,
            src_key_padding_mask=_safe_padding_mask(~valid_mask),
            is_causal=False,
        )
        last_valid = valid_mask.sum(dim=1).sub(1).clamp_min(0)
        readout = hidden[
            torch.arange(hidden.shape[0], device=hidden.device), last_valid
        ]
        return self.projection(self.output_norm(readout))

    def _validate_input(self, quantized: Tensor, valid_mask: Tensor) -> None:
        if quantized.ndim != 3 or valid_mask.shape != quantized.shape[:2]:
            raise ValueError(
                "quantized must have shape [B, M, H] and valid_mask [B, M]."
            )
        if valid_mask.dtype != torch.bool or valid_mask.device != quantized.device:
            raise ValueError(
                "valid_mask must be boolean and on the same device as quantized."
            )
        if quantized.shape[1] > self.position.num_embeddings:
            raise ValueError(
                "Prediction token length exceeds the configured maximum of "
                f"{self.position.num_embeddings}."
            )


class XTokenizerCodec(BaseLatentCodec):
    """Encoder-RVQ-decoder codec with X-Tokenizer pretraining objectives."""

    config: XTokenizerModelConfig

    def __init__(
        self,
        config: dict[str, Any] | XTokenizerModelConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if config is None:
            cfg = XTokenizerModelConfig()
        elif isinstance(config, XTokenizerModelConfig):
            cfg = config
        else:
            cfg = XTokenizerModelConfig.from_dict(config)
        if cfg.action_dim is None:
            raise ValueError(
                f"{self.__class__.__name__} requires action_dim at construction time. "
                "Set codec.action_dim explicitly or build via the dataset-aware path."
            )

        self.config = cfg
        self.action_dim = int(cfg.action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self.compression_ratio = int(cfg.compression_ratio)
        self.latent_horizon = self.action_horizon // self.compression_ratio
        self.latent_dim = int(cfg.latent_dim)
        self.codebook_size = int(cfg.codebook_size)
        self.num_quantizers = int(cfg.num_quantizers)
        self.position_slices = cfg.position_slices
        self.rotation_slices = cfg.rotation_slices
        self.position_slice = cfg.position_slice
        self.rotation_slice = cfg.rotation_slice
        self.scalar_channel_indices = cfg.scalar_channel_indices

        self.encoder = RelativeActionEncoder(cfg)
        self.quantizer = SemanticResidualQuantizer(
            dim=self.latent_dim,
            codebook_size=self.codebook_size,
            num_quantizers=self.num_quantizers,
            decay=cfg.ema_decay,
            dead_code_threshold=cfg.dead_code_threshold,
            kmeans_init=cfg.kmeans_init,
            kmeans_iters=cfg.kmeans_iters,
        )
        self.decoder = ActionDecoder(cfg)
        self.mam_head = MAMHead(cfg)
        self.alignment_projection = nn.Linear(
            cfg.vl_feature_dim,
            self.latent_dim,
        )
        self.vl_view_logits = nn.Parameter(torch.zeros(3))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(cfg.infonce_scale_init)))
        self.prediction_head = NextFramePredictionHead(
            self.latent_dim,
            cfg.vl_feature_dim,
            num_heads=cfg.prediction_num_heads,
            depth=cfg.prediction_depth,
            dropout=cfg.dropout,
            max_length=cfg.max_action_horizon // cfg.compression_ratio,
        )
        self._training_global_step = 0
        self._training_epoch: int | None = 0
        self.normalize_observation: Normalize | None = None
        self.unnormalize_observation: Unnormalize | None = None
        self._canonical_stats_identity: dict[str, Any] | None = None
        self._normalization_stats_horizon: int | None = None

    def set_training_context(self, context: Any) -> None:
        """Publish trainer progress used by the MAM step warm-up."""
        self._training_global_step = int(context.global_step)

    def set_training_epoch(self, epoch: int) -> None:
        """Publish epoch progress when the trainer exposes it."""
        if isinstance(epoch, bool) or int(epoch) != epoch or int(epoch) < 0:
            raise ValueError(f"epoch must be a non-negative integer, got {epoch!r}.")
        self._training_epoch = int(epoch)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        actions = _required_tensor(batch, self.action_key)
        anchor = self._reshape_anchor(_required_tensor(batch, "observation.state"))
        robot_type = _required_tensor(batch, "robot_type")
        action_valid = _action_valid_mask(actions, batch.get("action_is_pad"))
        action_dof_valid = _action_dof_valid_mask(
            actions,
            action_valid,
            batch.get("action_dof_mask"),
        )
        relative_actions = self._to_relative_actions(actions, anchor)
        normalized_actions = self._normalize_relative_actions(relative_actions)
        normalized_actions = normalized_actions.masked_fill(~action_dof_valid, 0)
        normalized_anchor = self._normalize_anchor(anchor)
        normalized_anchor = normalized_anchor.masked_fill(
            ~action_dof_valid.any(dim=1, keepdim=True),
            0,
        )
        conditioning_anchor = normalized_anchor
        conditioning_robot_type = robot_type
        embodiment_valid: Tensor | None = None
        if self.training:
            (
                conditioning_anchor,
                conditioning_robot_type,
                embodiment_valid,
            ) = _corrupt_conditioning(
                normalized_anchor,
                robot_type,
                state_dropout_probability=self.config.state_dropout_probability,
                embodiment_dropout_probability=(
                    self.config.embodiment_dropout_probability
                ),
                embodiment_none_probability=self.config.embodiment_none_probability,
                embodiment_none_id=self.config.embodiment_vocab_size - 1,
                generator=None,
            )

        encoded = self.encoder(
            normalized_actions,
            conditioning_anchor,
            conditioning_robot_type,
            action_valid,
            embodiment_valid=embodiment_valid,
        )
        quantized = self.quantizer(encoded.latents, encoded.latent_mask)
        reconstruction = self.decoder(
            quantized.quantized,
            conditioning_anchor,
            conditioning_robot_type,
            action_valid,
            embodiment_valid=embodiment_valid,
        )
        reconstruction = reconstruction.masked_fill(~action_dof_valid, 0)

        terms: list[Tensor] = []
        metrics: dict[str, float] = {}
        scalar_reconstruction, scalar_target = _non_rotation_channels(
            reconstruction,
            normalized_actions,
            scalar_channel_indices=self.scalar_channel_indices,
        )
        scalar_dof_valid = action_dof_valid[..., list(self.scalar_channel_indices)]
        action_l1 = masked_action_l1(
            scalar_reconstruction,
            scalar_target,
            scalar_dof_valid,
        )
        _append_weighted_term(
            terms,
            metrics,
            name="action_l1_loss",
            value=action_l1,
            weight=self.config.reconstruction_weight,
        )
        if self.rotation_slices:
            rotation_loss = _multi_rotation_geodesic_loss(
                reconstruction,
                normalized_actions,
                self._action_stats(),
                action_dof_valid,
                normalization_mode=self._action_norm_mode(),
                rotation_slices=self.rotation_slices,
            )
            _append_weighted_term(
                terms,
                metrics,
                name="rotation_geodesic_loss",
                value=rotation_loss,
                weight=self.config.rotation_weight,
            )
        _append_weighted_term(
            terms,
            metrics,
            name="commitment_loss",
            value=quantized.commitment_loss,
            weight=self.config.commitment_weight,
        )
        if self.config.dct_weight > 0.0:
            dct_loss = _masked_dct_l1(
                reconstruction,
                normalized_actions,
                action_valid,
                dof_valid_mask=action_dof_valid,
            )
            _append_weighted_term(
                terms,
                metrics,
                name="dct_loss",
                value=dct_loss,
                weight=self.config.dct_weight,
            )
        if self.config.smoothness_weight > 0.0:
            smoothness_loss = _masked_smoothness_l1(
                reconstruction,
                normalized_actions,
                action_valid,
                scalar_channel_indices=self.scalar_channel_indices,
                dof_valid_mask=action_dof_valid,
            )
            _append_weighted_term(
                terms,
                metrics,
                name="smoothness_loss",
                value=smoothness_loss,
                weight=self.config.smoothness_weight,
            )

        if self._objective_is_active("mam", batch):
            mam_mask = _sample_mam_mask(
                encoded.latent_mask,
                probability=self.config.mam_mask_probability,
            )
            q0_indices = quantized.indices[..., 0]
            mam_logits = self.mam_head(
                quantized.q0_quantized,
                self.quantizer.codebooks[0],
                mam_mask,
                encoded.latent_mask,
            )
            mam_loss = _masked_cross_entropy(mam_logits, q0_indices, mam_mask)
            _append_weighted_term(
                terms,
                metrics,
                name="mam_loss",
                value=mam_loss,
                weight=self.config.mam_weight,
            )

        if self._objective_is_active("alignment", batch):
            vl_features = _required_tensor(batch, "vl_features")
            vl_feature_is_pad = _required_tensor(batch, "vl_feature_is_pad")
            if (
                vl_features.ndim != 4
                or not 1 <= vl_features.shape[2] <= self.vl_view_logits.numel()
                or vl_features.shape[-1] != self.config.vl_feature_dim
            ):
                raise ValueError(
                    "vl_features must preserve one to three per-view streams "
                    "with shape [B, T_v, V, H_vl]."
                )
            projected_vl = self.alignment_projection(vl_features.detach())
            vl_features, vl_valid = _fuse_vl_views(
                projected_vl,
                vl_feature_is_pad,
                self.vl_view_logits[: vl_features.shape[2]],
            )
            action_features = encoded.latents
            action_features, vl_features, alignment_valid = _align_temporal_features(
                action_features=action_features,
                action_valid_mask=encoded.latent_mask,
                vl_features=vl_features,
                vl_valid_mask=vl_valid,
            )
            global_loss, local_loss = symmetric_global_local_infonce(
                action_features,
                vl_features,
                alignment_valid,
                self.logit_scale,
            )
            alignment_loss = 0.5 * (global_loss + local_loss)
            _record_metric(metrics, "alignment_global_loss", global_loss)
            _record_metric(metrics, "alignment_local_loss", local_loss)
            _append_weighted_term(
                terms,
                metrics,
                name="alignment_loss",
                value=alignment_loss,
                weight=self.config.alignment_weight,
            )

        if self._objective_is_active("prediction", batch):
            next_vl_features = _required_tensor(batch, "next_vl_features")
            prediction = self.prediction_head(
                quantized.quantized,
                encoded.latent_mask,
            )
            prediction_valid = encoded.latent_mask.any(dim=1)
            prediction_valid = prediction_valid & _next_vl_valid_mask(
                next_vl_features,
                batch.get("next_vl_feature_is_pad"),
            )
            prediction_loss = masked_action_l1(
                prediction,
                next_vl_features.detach(),
                prediction_valid,
            )
            _append_weighted_term(
                terms,
                metrics,
                name="prediction_loss",
                value=prediction_loss,
                weight=self.config.prediction_weight,
            )

        for level, perplexity in enumerate(quantized.perplexity):
            _record_metric(metrics, f"codebook_perplexity_q{level}", perplexity)
        _record_metric(
            metrics,
            "codebook_perplexity",
            quantized.perplexity.mean(),
        )
        total_loss = sum(terms, normalized_actions.new_zeros(()))
        if not bool(torch.isfinite(total_loss)):
            raise FloatingPointError(
                "XTokenizerCodec produced a non-finite total loss."
            )
        _record_metric(metrics, "total_loss", total_loss)
        return total_loss, metrics

    @torch.no_grad()
    def encode(self, actions: Tensor, **kwargs: Any) -> Tensor:
        anchor, robot_type, action_mask = self._inference_conditioning(
            actions,
            kwargs,
        )
        anchor = self._reshape_anchor(anchor)
        action_dof_valid = _action_dof_valid_mask(
            actions,
            action_mask,
            kwargs.get("action_dof_mask"),
        )
        relative_actions = self._to_relative_actions(actions, anchor)
        normalized_actions = self._normalize_relative_actions(relative_actions)
        normalized_actions = normalized_actions.masked_fill(~action_dof_valid, 0)
        normalized_anchor = self._normalize_anchor(anchor)
        normalized_anchor = normalized_anchor.masked_fill(
            ~action_dof_valid.any(dim=1, keepdim=True),
            0,
        )
        encoded = self.encoder(
            normalized_actions,
            normalized_anchor,
            robot_type,
            action_mask,
        )
        return self.quantizer.encode(encoded.latents)

    @torch.no_grad()
    def decode(self, indices: Tensor, **kwargs: Any) -> Tensor:
        if indices.ndim != 3:
            raise ValueError(
                f"indices must have shape [B, M, Q], got {tuple(indices.shape)}."
            )
        placeholder = torch.empty(
            indices.shape[0],
            self.action_horizon,
            self.action_dim,
            device=indices.device,
            dtype=self.decoder.output_projection.weight.dtype,
        )
        anchor, robot_type, action_mask = self._inference_conditioning(
            placeholder,
            kwargs,
        )
        anchor = self._reshape_anchor(anchor)
        action_dof_valid = _action_dof_valid_mask(
            placeholder,
            action_mask,
            kwargs.get("action_dof_mask"),
        )
        normalized_anchor = self._normalize_anchor(anchor)
        normalized_anchor = normalized_anchor.masked_fill(
            ~action_dof_valid.any(dim=1, keepdim=True),
            0,
        )
        latents = self.quantizer.decode(indices).to(
            dtype=self.decoder.output_projection.weight.dtype
        )
        normalized_actions = self.decoder(
            latents,
            normalized_anchor,
            robot_type,
            action_mask,
        )
        relative_actions = self._unnormalize_action_tensor(normalized_actions)
        actions = self._from_relative_actions(relative_actions, anchor)
        return actions.masked_fill(~action_dof_valid, 0)

    @torch.no_grad()
    def autoencode(self, actions: Tensor, **kwargs: Any) -> Tensor:
        indices = self.encode(actions, **kwargs)
        return self.decode(indices, **kwargs)

    @torch.no_grad()
    def reconstruct_with_info(
        self,
        actions: Tensor,
        *,
        batch: dict[str, Any],
        action_key: str,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Reconstruct actions once and retain the encoded RVQ assignments."""
        conditioning = {
            key: batch[key]
            for key in ("observation.state", "robot_type")
            if key in batch
        }
        pad_key = f"{action_key}_is_pad"
        if pad_key in batch:
            conditioning["action_is_pad"] = batch[pad_key]
        indices = self.encode(actions, **conditioning)
        reconstruction = self.decode(indices, **conditioning)
        return reconstruction, {"xtokenizer_indices": indices}

    def inference_state_dict(self) -> dict[str, Tensor]:
        """Return the complete inference state without pretraining heads."""
        state_dict: dict[str, Tensor] = {}
        for prefix, module in (
            ("encoder", self.encoder),
            ("quantizer", self.quantizer),
            ("decoder", self.decoder),
        ):
            state_dict.update(
                {
                    f"{prefix}.{key}": value.detach().cpu()
                    for key, value in module.state_dict().items()
                }
            )
        state_dict["quantizer.codebooks"] = state_dict["quantizer.codebooks"].float()
        return state_dict

    def export_inference_artifact(
        self,
        output_dir: Path,
        *,
        source_checkpoint_sha256: str | None = None,
        dof_layout: dict[str, Any] | None = None,
        embodiment_registry: dict[str, int] | None = None,
        training_data_identity: dict[str, Any] | None = None,
    ) -> Path:
        """Atomically write a strict, training-private-free inference artifact."""
        if not bool(self.quantizer.initialized.all()):
            pending = (~self.quantizer.initialized).nonzero(as_tuple=False).flatten()
            raise RuntimeError(
                "Cannot export X-Tokenizer before k-means has initialized every "
                f"RVQ level; pending levels: {pending.tolist()}."
            )
        from praxis.codecs.latents.xtokenizer.inference import (
            export_codec_inference_artifact,
        )

        return export_codec_inference_artifact(
            self,
            output_dir,
            source_checkpoint_sha256=source_checkpoint_sha256,
            dof_layout=dof_layout,
            embodiment_registry=embodiment_registry,
            training_data_identity=training_data_identity,
        )

    def get_optim_params(self) -> list[dict[str, Any]]:
        return split_decay_param_groups(
            self,
            no_decay_weight_decay=0.0,
        )

    def set_normalization_stats(
        self,
        stats: dict[str, dict[str, Tensor]],
        modes: dict[str, NormMode] | None = None,
    ) -> None:
        """Install the generic BaseCodec action normalizer lifecycle."""
        super().set_normalization_stats(stats, modes=modes)
        self.normalize_observation = None
        self.unnormalize_observation = None
        self._canonical_stats_identity = None
        self._normalization_stats_horizon = None

    def set_xtokenizer_normalization_stats(
        self,
        *,
        predict_stats: dict[str, dict[str, Tensor]],
        observation_stats: dict[str, dict[str, Tensor]],
        canonical_stats_identity: Mapping[str, Any] | None,
        stats_horizon: int,
    ) -> None:
        """Install the paper's independent predict and observation MinMax roles."""
        if canonical_stats_identity is None:
            raise ValueError(
                "X-Tokenizer paper normalization requires a canonical statistics "
                "identity."
            )
        from praxis.data.xtokenizer_canonical_stats import (
            validate_xtokenizer_quantile_identity,
        )

        canonical_identity = validate_xtokenizer_quantile_identity(
            canonical_stats_identity,
            expected_action_dim=self.action_dim,
            expected_stats_horizon=stats_horizon,
        )
        if stats_horizon != self.action_horizon:
            raise ValueError(
                "X-Tokenizer normalization stats_horizon must match action_horizon: "
                f"{stats_horizon} != {self.action_horizon}."
            )
        predict_values = predict_stats.get(self.action_key)
        observation_values = observation_stats.get("observation.state")
        if predict_values is None or observation_values is None:
            raise ValueError(
                "X-Tokenizer normalization requires predict stats for "
                f"{self.action_key!r} and observation stats for "
                "'observation.state'."
            )
        validated_predict = _validated_xtokenizer_role_stats(
            predict_values,
            action_dim=self.action_dim,
            role="predict_action",
            canonical_role=canonical_identity["normalization"]["predict_action"],
        )
        validated_observation = _validated_xtokenizer_role_stats(
            observation_values,
            action_dim=self.action_dim,
            role="observation_state",
            canonical_role=canonical_identity["normalization"]["observation_state"],
        )

        super().set_normalization_stats(
            {self.action_key: validated_predict},
            modes={self.action_key: NormMode.MIN_MAX},
        )
        self.normalize_observation = Normalize(
            {"observation.state": validated_observation},
            {"observation.state": NormMode.MIN_MAX},
        )
        self.unnormalize_observation = Unnormalize(
            {"observation.state": validated_observation},
            {"observation.state": NormMode.MIN_MAX},
        )
        self._canonical_stats_identity = canonical_identity
        self._normalization_stats_horizon = stats_horizon

    def _export_norm_stats_for_checkpoint(
        self,
    ) -> tuple[dict[str, dict[str, Tensor]], dict[str, str]]:
        """Return both relative-action and absolute-observation statistics."""
        stats, modes = super()._export_norm_stats_for_checkpoint()
        collect_normalizer_stats(self.normalize_observation, stats, modes)
        return stats, modes

    def _normalization_checkpoint_payload(self) -> dict[str, Any]:
        payload = super()._normalization_checkpoint_payload()
        if (
            self.normalize_observation is not None
            and self.unnormalize_observation is not None
        ):
            if (
                self._canonical_stats_identity is None
                or self._normalization_stats_horizon is None
            ):
                raise ValueError(
                    "Marked X-Tokenizer normalization is missing its canonical "
                    "statistics identity."
                )
            payload["normalization_kind"] = "xtokenizer_dual_role_v1"
            payload["canonical_stats_identity"] = self._canonical_stats_identity
            payload["stats_horizon"] = self._normalization_stats_horizon
        return payload

    def _load_normalization_checkpoint_payload(self, data: dict[str, Any]) -> None:
        normalization_kind = data.get("normalization_kind")
        if normalization_kind is None:
            if self._canonical_stats_identity is not None:
                raise ValueError(
                    "Unmarked checkpoint normalization cannot replace the active "
                    "canonical statistics identity."
                )
            super()._load_normalization_checkpoint_payload(data)
            return
        if normalization_kind != "xtokenizer_dual_role_v1":
            raise ValueError(
                f"Unsupported X-Tokenizer normalization_kind {normalization_kind!r}."
            )
        required_fields = {
            "normalization_kind",
            "canonical_stats_identity",
            "stats_horizon",
            "stats",
            "modes",
        }
        if set(data) != required_fields:
            raise ValueError(
                "Marked X-Tokenizer normalization identity fields are malformed."
            )

        stats = data.get("stats", {})
        modes = data.get("modes", {})
        required_keys = {self.action_key, "observation.state"}
        if set(stats) != required_keys or set(modes) != required_keys:
            raise ValueError(
                "Marked X-Tokenizer normalization requires exactly action and "
                "observation.state statistics and modes."
            )
        resolved_modes = {key: NormMode(value) for key, value in modes.items()}
        if any(mode != NormMode.MIN_MAX for mode in resolved_modes.values()):
            raise ValueError(
                "Marked X-Tokenizer normalization requires MIN_MAX for both roles."
            )
        checkpoint_identity = data["canonical_stats_identity"]
        checkpoint_horizon = data["stats_horizon"]
        if self._canonical_stats_identity is not None:
            if checkpoint_identity != self._canonical_stats_identity:
                raise ValueError(
                    "Checkpoint normalization does not match the active canonical "
                    "statistics identity."
                )
            if checkpoint_horizon != self._normalization_stats_horizon:
                raise ValueError(
                    "Checkpoint normalization stats_horizon does not match the "
                    "active canonical statistics horizon."
                )
        self.set_xtokenizer_normalization_stats(
            predict_stats={self.action_key: stats[self.action_key]},
            observation_stats={"observation.state": stats["observation.state"]},
            canonical_stats_identity=checkpoint_identity,
            stats_horizon=checkpoint_horizon,
        )

    def _normalize_relative_actions(self, actions: Tensor) -> Tensor:
        normalized = self._normalize_action_tensor(actions)
        if self.normalize_observation is None:
            return normalized
        return normalized.clamp(-1.0, 1.0)

    def _normalize_anchor(self, anchor: Tensor) -> Tensor:
        if self.normalize_observation is None:
            return self._normalize_action_tensor(anchor)
        normalized = self.normalize_observation({"observation.state": anchor})[
            "observation.state"
        ]
        return normalized.clamp(-1.0, 1.0)

    def _reshape_anchor(self, anchor: Tensor) -> Tensor:
        if anchor.ndim == 2:
            anchor = anchor.unsqueeze(1)
        if anchor.ndim != 3 or anchor.shape[1:] != (1, self.action_dim):
            raise ValueError(
                f"observation.state must have shape [B, {self.action_dim}] or "
                f"[B, 1, {self.action_dim}], got {tuple(anchor.shape)}."
            )
        return anchor

    def _to_relative_actions(self, actions: Tensor, anchor: Tensor) -> Tensor:
        return to_observation_relative_actions(
            actions,
            anchor,
            position_slices=self.position_slices,
            rotation_slices=self.rotation_slices,
        )

    def _from_relative_actions(self, actions: Tensor, anchor: Tensor) -> Tensor:
        return from_observation_relative_actions(
            actions,
            anchor,
            position_slices=self.position_slices,
            rotation_slices=self.rotation_slices,
        )

    def _inference_conditioning(
        self,
        actions: Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if actions.ndim != 3:
            raise ValueError(
                f"actions must have shape [B, T, D], got {tuple(actions.shape)}."
            )
        batch_size = int(actions.shape[0])
        anchor = kwargs.get(
            "anchor",
            kwargs.get("observation.state", kwargs.get("observation_state")),
        )
        if anchor is None:
            raise ValueError(
                "X-Tokenizer absolute-action inference requires an observation.state "
                "anchor from immediately before the action chunk."
            )
        if not isinstance(anchor, Tensor):
            raise TypeError("anchor must be a tensor.")
        robot_type = kwargs.get("robot_type")
        if robot_type is None:
            robot_type = torch.zeros(
                batch_size,
                device=actions.device,
                dtype=torch.long,
            )
        if not isinstance(robot_type, Tensor):
            raise TypeError("robot_type must be a tensor.")
        action_mask = kwargs.get("action_mask")
        action_is_pad = kwargs.get("action_is_pad")
        if action_mask is not None and action_is_pad is not None:
            raise ValueError("Pass only one of action_mask or action_is_pad.")
        if action_mask is None:
            action_mask = _action_valid_mask(actions, action_is_pad)
        if not isinstance(action_mask, Tensor):
            raise TypeError("action_mask must be a tensor.")
        return anchor, robot_type, action_mask

    def _objective_is_active(
        self,
        name: str,
        batch: dict[str, Tensor],
    ) -> bool:
        weight = float(getattr(self.config, f"{name}_weight"))
        if weight <= 0.0:
            return False
        for key in (f"{name}_active", f"{name}_warmup_mask"):
            value = batch.get(key)
            if value is not None and not _tensor_flag(value, name=key):
                return False
        if name != "mam":
            return True
        if self._training_global_step < self.config.mam_warmup_steps:
            return False
        epoch_value = batch.get("epoch")
        epoch = (
            _scalar_int(epoch_value, name="epoch")
            if epoch_value is not None
            else self._training_epoch
        )
        return epoch is None or epoch >= self.config.mam_warmup_epochs

    def _action_stats(self) -> dict[str, Tensor]:
        if self.normalize_action is None:
            return {}
        return dict(self.normalize_action.stats.get(self.action_key, {}))

    def _action_norm_mode(self) -> NormMode | None:
        if self.normalize_action is None:
            return None
        return self.normalize_action.modes.get(self.action_key)


def _validated_xtokenizer_role_stats(
    values: Mapping[str, Any],
    *,
    action_dim: int,
    role: str,
    canonical_role: Mapping[str, Any],
) -> dict[str, Tensor]:
    if not isinstance(values, Mapping) or set(values) != {"min", "max"}:
        raise ValueError(
            f"X-Tokenizer {role} statistics require exactly min and max values."
        )
    tensors = {
        name: torch.as_tensor(value, dtype=torch.float32)
        for name, value in values.items()
    }
    for name, tensor in tensors.items():
        if tensor.shape != (action_dim,):
            raise ValueError(
                f"X-Tokenizer {role}.{name} must have shape ({action_dim},)."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"X-Tokenizer {role}.{name} must contain finite values.")
    if not bool((tensors["min"] < tensors["max"]).all()):
        raise ValueError(f"X-Tokenizer {role} statistics require min < max.")
    for name, canonical_name in (("min", "q001"), ("max", "q999")):
        expected = torch.tensor(canonical_role[canonical_name], dtype=torch.float32)
        if not torch.equal(tensors[name].detach().cpu(), expected):
            raise ValueError(
                f"X-Tokenizer {role} statistics do not match canonical statistics."
            )
    return tensors


def _required_tensor(batch: dict[str, Tensor], key: str) -> Tensor:
    try:
        value = batch[key]
    except KeyError:
        raise KeyError(f"XTokenizerCodec requires batch[{key!r}].") from None
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a tensor.")
    return value


def _action_valid_mask(actions: Tensor, action_is_pad: Tensor | None) -> Tensor:
    if actions.ndim != 3:
        raise ValueError(
            f"actions must have shape [B, T, D], got {tuple(actions.shape)}."
        )
    expected = actions.shape[:2]
    if action_is_pad is None:
        return torch.ones(expected, device=actions.device, dtype=torch.bool)
    if action_is_pad.shape != expected:
        raise ValueError(
            f"action_is_pad must have shape {tuple(expected)}, "
            f"got {tuple(action_is_pad.shape)}."
        )
    if action_is_pad.dtype != torch.bool:
        raise ValueError("action_is_pad must have boolean dtype.")
    if action_is_pad.device != actions.device:
        raise ValueError("action_is_pad and actions must be on the same device.")
    return ~action_is_pad


def _action_dof_valid_mask(
    actions: Tensor,
    action_valid_mask: Tensor,
    action_dof_mask: Tensor | None,
) -> Tensor:
    temporal = action_valid_mask.unsqueeze(-1).expand_as(actions)
    if action_dof_mask is None:
        return temporal
    if action_dof_mask.shape != actions.shape:
        raise ValueError(
            "action_dof_mask must match action shape, got "
            f"{tuple(action_dof_mask.shape)} and {tuple(actions.shape)}."
        )
    if action_dof_mask.dtype != torch.bool:
        raise ValueError("action_dof_mask must have boolean dtype.")
    if action_dof_mask.device != actions.device:
        raise ValueError("action_dof_mask and actions must share a device.")
    return temporal & action_dof_mask


def _corrupt_conditioning(
    anchor: Tensor,
    robot_type: Tensor,
    *,
    state_dropout_probability: float,
    embodiment_dropout_probability: float,
    embodiment_none_probability: float,
    embodiment_none_id: int,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply the paper's independent state/embodiment CFG corruption."""
    if anchor.ndim != 3 or anchor.shape[1] != 1:
        raise ValueError("anchor must have shape [B, 1, D].")
    if robot_type.shape != anchor.shape[:1]:
        raise ValueError("robot_type must have shape [B] matching anchor.")
    if robot_type.device != anchor.device:
        raise ValueError("robot_type and anchor must share a device.")
    for name, probability in (
        ("state_dropout_probability", state_dropout_probability),
        ("embodiment_dropout_probability", embodiment_dropout_probability),
        ("embodiment_none_probability", embodiment_none_probability),
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {probability}.")

    shape = robot_type.shape
    state_dropped = (
        torch.rand(shape, device=anchor.device, generator=generator)
        < state_dropout_probability
    )
    embodiment_dropped = (
        torch.rand(shape, device=anchor.device, generator=generator)
        < embodiment_dropout_probability
    )
    use_none = (
        torch.rand(shape, device=anchor.device, generator=generator)
        < embodiment_none_probability
    )
    corrupted_anchor = anchor.masked_fill(state_dropped[:, None, None], 0)
    corrupted_robot_type = robot_type.clone()
    corrupted_robot_type[use_none] = embodiment_none_id
    embodiment_valid = ~embodiment_dropped | use_none
    return corrupted_anchor, corrupted_robot_type, embodiment_valid


def _fuse_vl_views(
    projected_features: Tensor,
    vl_feature_is_pad: Tensor,
    view_logits: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fuse separately projected views with learned missing-view-aware weights."""
    if projected_features.ndim != 4:
        raise ValueError(
            "projected VL features must have shape [B, T_v, V, H], "
            f"got {tuple(projected_features.shape)}."
        )
    if vl_feature_is_pad.shape != projected_features.shape[:3]:
        raise ValueError(
            "vl_feature_is_pad must have shape [B, T_v, V] matching features."
        )
    if vl_feature_is_pad.dtype != torch.bool:
        raise ValueError("vl_feature_is_pad must have boolean dtype.")
    if vl_feature_is_pad.device != projected_features.device:
        raise ValueError("VL padding mask and features must share a device.")
    num_views = projected_features.shape[2]
    if view_logits.shape != (num_views,):
        raise ValueError(
            f"view_logits must have shape ({num_views},), got {tuple(view_logits.shape)}."
        )
    if view_logits.device != projected_features.device:
        raise ValueError("view_logits and VL features must share a device.")

    valid = ~vl_feature_is_pad.all(dim=-1)
    logits = view_logits.reshape(1, 1, num_views).expand(*projected_features.shape[:3])
    logits = logits.masked_fill(vl_feature_is_pad, float("-inf"))
    logits = logits.masked_fill(~valid.unsqueeze(-1), 0)
    weights = torch.softmax(logits.float(), dim=-1).to(projected_features.dtype)
    weights = weights.masked_fill(vl_feature_is_pad, 0)
    clean = projected_features.masked_fill(vl_feature_is_pad.unsqueeze(-1), 0)
    fused = (clean * weights.unsqueeze(-1)).sum(dim=2)
    return fused.masked_fill(~valid.unsqueeze(-1), 0), valid


def _next_vl_valid_mask(
    next_vl_features: Tensor,
    next_vl_feature_is_pad: Tensor | None,
) -> Tensor:
    if next_vl_features.ndim != 2:
        raise ValueError(
            "next_vl_features must have shape [B, H_vl], "
            f"got {tuple(next_vl_features.shape)}."
        )
    if next_vl_feature_is_pad is None:
        return torch.ones(
            next_vl_features.shape[0],
            device=next_vl_features.device,
            dtype=torch.bool,
        )
    if next_vl_feature_is_pad.shape != next_vl_features.shape[:1]:
        raise ValueError(
            "next_vl_feature_is_pad must have shape [B], got "
            f"{tuple(next_vl_feature_is_pad.shape)}."
        )
    if next_vl_feature_is_pad.dtype != torch.bool:
        raise ValueError("next_vl_feature_is_pad must have boolean dtype.")
    if next_vl_feature_is_pad.device != next_vl_features.device:
        raise ValueError(
            "next_vl_feature_is_pad and next_vl_features must share a device."
        )
    return ~next_vl_feature_is_pad


def _align_temporal_features(
    *,
    action_features: Tensor,
    action_valid_mask: Tensor,
    vl_features: Tensor,
    vl_valid_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pool only the longer stream to the shorter temporal length."""
    if action_features.ndim != 3 or vl_features.ndim != 3:
        raise ValueError("Action and VL features must have shape [B, T, H].")
    if action_features.shape[0] != vl_features.shape[0]:
        raise ValueError("Action and VL features must have the same batch size.")
    if action_features.shape[2] != vl_features.shape[2]:
        raise ValueError("Action and VL features must have the same hidden width.")
    if action_features.device != vl_features.device:
        raise ValueError("Action and VL features must share a device.")
    _validate_temporal_mask(action_valid_mask, action_features)
    _validate_temporal_mask(vl_valid_mask, vl_features)

    common_length = min(action_features.shape[1], vl_features.shape[1])
    aligned_action, action_valid = _adaptive_masked_pool(
        action_features,
        action_valid_mask,
        output_length=common_length,
    )
    aligned_vl, vl_valid = _adaptive_masked_pool(
        vl_features,
        vl_valid_mask,
        output_length=common_length,
    )
    return aligned_action, aligned_vl, action_valid & vl_valid


def _adaptive_masked_pool(
    features: Tensor,
    valid_mask: Tensor,
    *,
    output_length: int,
) -> tuple[Tensor, Tensor]:
    if features.shape[1] == output_length:
        return features, valid_mask
    clean = features.masked_fill(~valid_mask.unsqueeze(-1), 0)
    pooled_features = F.adaptive_avg_pool1d(
        clean.transpose(1, 2),
        output_length,
    ).transpose(1, 2)
    pooled_weights = F.adaptive_avg_pool1d(
        valid_mask.float().unsqueeze(1),
        output_length,
    ).squeeze(1)
    pooled_features = pooled_features / pooled_weights.clamp_min(1e-12).unsqueeze(-1)
    pooled_valid = pooled_weights > 0
    return pooled_features.masked_fill(~pooled_valid.unsqueeze(-1), 0), pooled_valid


def _validate_temporal_mask(valid_mask: Tensor, features: Tensor) -> None:
    if valid_mask.shape != features.shape[:2]:
        raise ValueError(
            "Temporal valid mask must match feature shape, got "
            f"{tuple(valid_mask.shape)} and {tuple(features.shape[:2])}."
        )
    if valid_mask.dtype != torch.bool or valid_mask.device != features.device:
        raise ValueError(
            "Temporal valid mask must be boolean and share the feature device."
        )


def _sample_mam_mask(valid_mask: Tensor, *, probability: float) -> Tensor:
    if probability <= 0.0:
        return torch.zeros_like(valid_mask)
    selected = (
        torch.rand(valid_mask.shape, device=valid_mask.device) < probability
    ) & (valid_mask)
    missing = valid_mask.any(dim=1) & ~selected.any(dim=1)
    if bool(missing.any()):
        first_valid = valid_mask.float().argmax(dim=1)
        selected[missing, first_valid[missing]] = True
    return selected


def _masked_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask]
    if valid_targets.numel() == 0:
        return logits.masked_fill(~valid_mask.unsqueeze(-1), 0).sum() * 0.0
    return F.cross_entropy(valid_logits, valid_targets.long())


def _masked_mean(features: Tensor, valid_mask: Tensor) -> Tensor:
    clean = features.masked_fill(~valid_mask.unsqueeze(-1), 0)
    counts = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return clean.sum(dim=1) / counts


def _non_rotation_channels(
    reconstruction: Tensor,
    target: Tensor,
    *,
    scalar_channel_indices: tuple[int, ...],
) -> tuple[Tensor, Tensor]:
    return (
        reconstruction[..., list(scalar_channel_indices)],
        target[..., list(scalar_channel_indices)],
    )


def _multi_rotation_geodesic_loss(
    reconstruction: Tensor,
    target: Tensor,
    stats: dict[str, Tensor],
    dof_valid_mask: Tensor,
    *,
    normalization_mode: NormMode | None,
    rotation_slices: tuple[slice, ...],
) -> Tensor:
    weighted_losses: list[Tensor] = []
    valid_counts: list[Tensor] = []
    for segment in rotation_slices:
        rotation_valid = dof_valid_mask[..., segment].all(dim=-1)
        count = rotation_valid.sum()
        valid_counts.append(count)
        weighted_losses.append(
            rotation_geodesic_loss(
                reconstruction,
                target,
                stats,
                rotation_valid,
                normalization_mode=normalization_mode,
                rotation_slice=segment,
            )
            * count
        )
    total_count = sum(valid_counts, dof_valid_mask.new_zeros((), dtype=torch.long))
    if int(total_count.item()) == 0:
        return reconstruction.masked_fill(~dof_valid_mask, 0).sum() * 0.0
    return sum(weighted_losses, reconstruction.new_zeros(())) / total_count


def _masked_dct_l1(
    reconstruction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    dof_valid_mask: Tensor | None = None,
) -> Tensor:
    feature_valid = valid_mask.unsqueeze(-1).expand_as(reconstruction)
    if dof_valid_mask is not None:
        if dof_valid_mask.shape != reconstruction.shape:
            raise ValueError("dof_valid_mask must match DCT action tensors.")
        feature_valid = feature_valid & dof_valid_mask
    clean_reconstruction = reconstruction.masked_fill(~feature_valid, 0)
    clean_target = target.masked_fill(~feature_valid, 0)
    horizon = int(reconstruction.shape[1])
    positions = torch.arange(
        horizon,
        device=reconstruction.device,
        dtype=reconstruction.dtype,
    )
    frequencies = positions.unsqueeze(1)
    basis = torch.cos(torch.pi * (positions + 0.5).unsqueeze(0) * frequencies / horizon)
    basis[0] *= math.sqrt(1.0 / horizon)
    if horizon > 1:
        basis[1:] *= math.sqrt(2.0 / horizon)
    reconstruction_dct = torch.einsum("kt,btd->bkd", basis, clean_reconstruction)
    target_dct = torch.einsum("kt,btd->bkd", basis, clean_target)
    valid_frequencies = feature_valid.any(dim=1, keepdim=True).expand_as(
        reconstruction_dct
    )
    return masked_action_l1(
        reconstruction_dct,
        target_dct,
        valid_frequencies,
    )


def _masked_smoothness_l1(
    reconstruction: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    scalar_channel_indices: tuple[int, ...],
    dof_valid_mask: Tensor | None = None,
) -> Tensor:
    scalar_reconstruction, scalar_target = _non_rotation_channels(
        reconstruction,
        target,
        scalar_channel_indices=scalar_channel_indices,
    )
    pair_mask: Tensor = valid_mask[:, 1:] & valid_mask[:, :-1]
    if dof_valid_mask is not None:
        if dof_valid_mask.shape != reconstruction.shape:
            raise ValueError("dof_valid_mask must match smoothness action tensors.")
        scalar_dof_valid = dof_valid_mask[..., list(scalar_channel_indices)]
        pair_mask = (
            pair_mask.unsqueeze(-1) & scalar_dof_valid[:, 1:] & scalar_dof_valid[:, :-1]
        )
    reconstruction_delta = scalar_reconstruction[:, 1:] - scalar_reconstruction[:, :-1]
    target_delta = scalar_target[:, 1:] - scalar_target[:, :-1]
    return masked_action_l1(reconstruction_delta, target_delta, pair_mask)


def _append_weighted_term(
    terms: list[Tensor],
    metrics: dict[str, float],
    *,
    name: str,
    value: Tensor,
    weight: float,
) -> None:
    if weight <= 0.0:
        return
    weighted = value * float(weight)
    terms.append(weighted)
    _record_metric(metrics, name, value)
    _record_metric(metrics, f"{name}_weighted", weighted)


def _record_metric(metrics: dict[str, float], name: str, value: Tensor) -> None:
    detached = value.detach().float()
    if detached.numel() != 1:
        raise ValueError(f"metric {name!r} must be scalar, got {tuple(value.shape)}.")
    if not bool(torch.isfinite(detached)):
        raise FloatingPointError(f"metric {name!r} is non-finite.")
    metrics[name] = float(detached.item())


def _safe_padding_mask(padding_mask: Tensor) -> Tensor:
    all_padded = padding_mask.all(dim=1)
    if not bool(all_padded.any()):
        return padding_mask
    safe = padding_mask.clone()
    safe[all_padded, 0] = False
    return safe


def _tensor_flag(value: Tensor, *, name: str) -> bool:
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must have boolean dtype.")
    return bool(value.any())


def _scalar_int(value: Tensor, *, name: str) -> int:
    if value.numel() != 1 or value.is_floating_point() or value.is_complex():
        raise ValueError(f"{name} must be an integer scalar tensor.")
    resolved = int(value.item())
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative, got {resolved}.")
    return resolved
