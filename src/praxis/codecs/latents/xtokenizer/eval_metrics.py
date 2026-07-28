# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""X-Tokenizer reconstruction metrics."""

from __future__ import annotations

import math
from typing import Any, cast

import torch

from praxis.codecs.latents.xtokenizer.codec import XTokenizerCodec
from praxis.codecs.latents.xtokenizer.objectives import (
    masked_action_l1,
    rotation_geodesic_loss,
)
from praxis.training.core.eval_collectors import (
    ReconstructionBatchContext,
    ReconstructionMetricCollector,
    register_reconstruction_metric_collector_factory,
)


class XTokenizerReconstructionMetricCollector:
    """Aggregate physical reconstruction and RVQ usage metrics."""

    owns_default_reconstruction_loss = True

    def __init__(self, codec: XTokenizerCodec, *, device: torch.device) -> None:
        self.num_quantizers = int(codec.num_quantizers)
        self.codebook_size = int(codec.codebook_size)
        self.compression_ratio = int(codec.compression_ratio)
        self.position_slice = codec.position_slice
        self.rotation_slice = codec.rotation_slice
        self.metric_sums = torch.zeros(6, device=device, dtype=torch.float64)
        self.assignment_counts = torch.zeros(
            self.num_quantizers,
            self.codebook_size,
            device=device,
            dtype=torch.float64,
        )

    def update(self, ctx: ReconstructionBatchContext) -> None:
        valid = _reconstruction_action_valid_mask(ctx)
        valid_count = valid.sum(dtype=torch.float64)
        if bool(valid_count):
            valid_pred = ctx.pred_action[valid]
            valid_target = ctx.gt_action[valid]
            element_count = torch.tensor(
                valid_target.numel(),
                device=valid.device,
                dtype=torch.float64,
            )
            reconst_mse = (valid_pred - valid_target).square().mean()
            self.metric_sums[0] += reconst_mse.double() * element_count
            self.metric_sums[1] += element_count

            if self.position_slice is not None:
                translation_l1 = masked_action_l1(
                    ctx.pred_action[..., self.position_slice],
                    ctx.gt_action[..., self.position_slice],
                    valid,
                )
                self.metric_sums[2] += translation_l1.double() * valid_count
                self.metric_sums[3] += valid_count

            if self.rotation_slice is not None:
                normalized_rotation_geodesic = rotation_geodesic_loss(
                    ctx.pred_action,
                    ctx.gt_action,
                    {},
                    valid,
                    rotation_slice=self.rotation_slice,
                )
                self.metric_sums[4] += (
                    normalized_rotation_geodesic.double() * valid_count
                )
                self.metric_sums[5] += valid_count

        indices = ctx.reconst_info.get("xtokenizer_indices")
        if not isinstance(indices, torch.Tensor):
            raise KeyError(
                "X-Tokenizer reconstruction metrics require encoded token indices."
            )
        expected_shape = (
            ctx.gt_action.shape[0],
            ctx.gt_action.shape[1] // self.compression_ratio,
            self.num_quantizers,
        )
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                "X-Tokenizer reconstruction indices must have shape "
                f"{expected_shape}, got {tuple(indices.shape)}."
            )
        latent_valid = valid.reshape(
            valid.shape[0],
            indices.shape[1],
            self.compression_ratio,
        ).any(dim=-1)
        for level in range(self.num_quantizers):
            valid_indices = indices[..., level][latent_valid]
            self.assignment_counts[level] += torch.bincount(
                valid_indices,
                minlength=self.codebook_size,
            ).to(device=self.assignment_counts.device, dtype=torch.float64)

    def reduce(self, runtime: Any) -> None:
        self.metric_sums = cast(
            torch.Tensor,
            runtime.reduce(self.metric_sums, reduction="sum"),
        )
        self.assignment_counts = cast(
            torch.Tensor,
            runtime.reduce(self.assignment_counts, reduction="sum"),
        )

    def finalize(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if bool(self.metric_sums[1]):
            metrics["val_reconst_loss"] = (
                self.metric_sums[0] / self.metric_sums[1]
            ).item()
        if bool(self.metric_sums[3]):
            metrics["val_translation_l1"] = (
                self.metric_sums[2] / self.metric_sums[3]
            ).item()
        if bool(self.metric_sums[5]):
            metrics["val_rotation_geodesic"] = (
                self.metric_sums[4] / self.metric_sums[5] * math.pi
            ).item()
        for level, counts in enumerate(self.assignment_counts):
            total = counts.sum()
            if not bool(total):
                continue
            probabilities = counts / total
            entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum()
            metrics[f"val_codebook_perplexity_q{level}"] = entropy.exp().item()
        return metrics


def _reconstruction_action_valid_mask(
    ctx: ReconstructionBatchContext,
) -> torch.Tensor:
    pad_key = f"{ctx.action_key}_is_pad"
    action_is_pad = ctx.batch.get(pad_key)
    expected_shape = ctx.gt_action.shape[:2]
    if action_is_pad is None:
        return torch.ones(
            expected_shape,
            device=ctx.gt_action.device,
            dtype=torch.bool,
        )
    if not isinstance(action_is_pad, torch.Tensor):
        raise TypeError(f"batch[{pad_key!r}] must be a tensor.")
    if action_is_pad.dtype != torch.bool or action_is_pad.shape != expected_shape:
        raise ValueError(
            f"batch[{pad_key!r}] must be boolean with shape {expected_shape}, "
            f"got dtype={action_is_pad.dtype} shape={tuple(action_is_pad.shape)}."
        )
    if action_is_pad.device != ctx.gt_action.device:
        raise ValueError(
            f"batch[{pad_key!r}] must be on device {ctx.gt_action.device}, "
            f"got {action_is_pad.device}."
        )
    return ~action_is_pad


@register_reconstruction_metric_collector_factory
def _build_xtokenizer_reconstruction_metric_collectors(
    *,
    policy: Any,
    cfg: Any,
    device: torch.device,
) -> list[ReconstructionMetricCollector]:
    del cfg
    if not isinstance(policy, XTokenizerCodec):
        return []
    return [XTokenizerReconstructionMetricCollector(policy, device=device)]
