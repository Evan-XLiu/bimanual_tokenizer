# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""EMA semantic residual quantization for X-Tokenizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ResidualQuantizerOutput:
    """Outputs from semantic residual quantization."""

    indices: Tensor
    quantized: Tensor
    q0_quantized: Tensor
    commitment_loss: Tensor
    perplexity: Tensor


class SemanticResidualQuantizer(nn.Module):
    """Quantize latent slots with a stack of EMA-updated codebooks."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        num_quantizers: int,
        decay: float,
        dead_code_threshold: int,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
    ) -> None:
        super().__init__()
        self.dim = _positive_int(dim, name="dim")
        self.codebook_size = _positive_int(
            codebook_size,
            name="codebook_size",
        )
        self.num_quantizers = _positive_int(
            num_quantizers,
            name="num_quantizers",
        )
        self.decay = float(decay)
        if not 0.0 <= self.decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}.")
        self.dead_code_threshold = _non_negative_int(
            dead_code_threshold,
            name="dead_code_threshold",
        )
        if not isinstance(kmeans_init, bool):
            raise ValueError(f"kmeans_init must be a boolean, got {kmeans_init!r}.")
        self.kmeans_init = kmeans_init
        self.kmeans_iters = _positive_int(kmeans_iters, name="kmeans_iters")

        if self.kmeans_init:
            codebooks = torch.zeros(
                self.num_quantizers,
                self.codebook_size,
                self.dim,
                dtype=torch.float32,
            )
        else:
            codebooks = torch.randn(
                self.num_quantizers,
                self.codebook_size,
                self.dim,
                dtype=torch.float32,
            )
            codebooks.mul_(self.dim**-0.5)
        self.register_buffer("codebooks", codebooks)
        self.register_buffer(
            "ema_counts",
            torch.ones(
                self.num_quantizers,
                self.codebook_size,
                dtype=torch.float32,
            ),
        )
        self.register_buffer("ema_sums", codebooks.clone())
        self.register_buffer(
            "initialized",
            torch.full(
                (self.num_quantizers,),
                not self.kmeans_init,
                dtype=torch.bool,
            ),
        )

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> SemanticResidualQuantizer:
        fp32_buffers = {
            name: self._buffers[name]
            for name in ("codebooks", "ema_counts", "ema_sums")
        }
        result = super()._apply(fn, recurse)
        for name, original in fp32_buffers.items():
            converted = self._buffers[name]
            if original is not None and converted is not None:
                self._buffers[name] = original.to(device=converted.device)
        return result

    def forward(
        self,
        latents: Tensor,
        valid_mask: Tensor | None = None,
    ) -> ResidualQuantizerOutput:
        mask = self._validate_inputs(latents, valid_mask)
        encoded = latents.float()
        residual = encoded
        quantized_sum = torch.zeros_like(encoded)
        indices_by_level: list[Tensor] = []
        commitment_by_level: list[Tensor] = []
        perplexity_by_level: list[Tensor] = []
        q0_quantized: Tensor | None = None

        for level in range(self.num_quantizers):
            level_input = residual
            if (
                self.training
                and torch.is_grad_enabled()
                and not bool(self.initialized[level])
            ):
                self._initialize_codebook(level, level_input.detach(), mask)
            indices = self._nearest_indices(level_input, level)
            codewords = F.embedding(indices, self.codebooks[level])
            counts = self._assignment_counts(indices, mask)

            indices_by_level.append(indices)
            quantized_sum = quantized_sum + codewords
            if level == 0:
                q0_quantized = _q0_straight_through(latents, codewords)
            commitment_by_level.append(
                self._masked_mse(level_input, codewords.detach(), mask)
            )
            perplexity_by_level.append(self._perplexity(counts))

            if self.training and torch.is_grad_enabled():
                self._ema_update(
                    level,
                    level_input.detach(),
                    indices,
                    counts,
                    mask,
                )

            residual = level_input - codewords.detach()

        quantized = (encoded + (quantized_sum - encoded).detach()).to(
            dtype=latents.dtype
        )
        if q0_quantized is None:
            raise AssertionError("Residual quantization must produce a q0 codeword.")
        return ResidualQuantizerOutput(
            indices=torch.stack(indices_by_level, dim=-1).long(),
            quantized=quantized,
            q0_quantized=q0_quantized,
            commitment_loss=torch.stack(commitment_by_level).mean(),
            perplexity=torch.stack(perplexity_by_level),
        )

    @torch.no_grad()
    def encode(self, latents: Tensor) -> Tensor:
        self._validate_inputs(latents, None)
        residual = latents.float()
        indices_by_level: list[Tensor] = []

        for level in range(self.num_quantizers):
            indices = self._nearest_indices(residual, level)
            indices_by_level.append(indices)
            residual = residual - F.embedding(indices, self.codebooks[level])

        return torch.stack(indices_by_level, dim=-1).long()

    @torch.no_grad()
    def _initialize_codebook(
        self,
        level: int,
        latents: Tensor,
        valid_mask: Tensor,
    ) -> None:
        local_candidates = latents.reshape(-1, self.dim)[valid_mask.reshape(-1)].float()
        candidates = self._gather_candidates(local_candidates)
        if candidates.shape[0] == 0:
            return

        distributed = self._distributed_is_initialized()
        is_source = not distributed or dist.get_rank() == 0
        if is_source:
            codebook, counts = _kmeans(
                candidates,
                num_clusters=self.codebook_size,
                num_iters=self.kmeans_iters,
            )
        else:
            codebook = torch.empty_like(self.codebooks[level])
            counts = torch.empty_like(self.ema_counts[level])

        if distributed:
            dist.broadcast(codebook, src=0)
            dist.broadcast(counts, src=0)

        self.codebooks[level].copy_(codebook)
        self.ema_counts[level].copy_(counts)
        self.ema_sums[level].copy_(codebook * counts.unsqueeze(-1))
        self.initialized[level] = True

    def _gather_candidates(self, candidates: Tensor) -> Tensor:
        if not self._distributed_is_initialized():
            return candidates

        world_size = dist.get_world_size()
        local_size = torch.tensor(
            candidates.shape[0],
            device=candidates.device,
            dtype=torch.long,
        )
        sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(sizes, local_size)
        size_values = [int(size.item()) for size in sizes]
        max_size = max(size_values)
        if max_size == 0:
            return candidates.new_empty((0, self.dim))

        padded = candidates.new_zeros((max_size, self.dim))
        padded[: candidates.shape[0]].copy_(candidates)
        gathered = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat(
            [
                rank_candidates[:size]
                for rank_candidates, size in zip(
                    gathered,
                    size_values,
                    strict=True,
                )
            ]
        )

    def decode(self, indices: Tensor) -> Tensor:
        if indices.ndim != 3:
            raise ValueError(
                f"indices must have shape [B, M, Q], got {tuple(indices.shape)}."
            )
        if indices.shape[-1] != self.num_quantizers:
            raise ValueError(
                "indices quantizer dimension must match num_quantizers, "
                f"got {indices.shape[-1]} and {self.num_quantizers}."
            )
        if indices.is_floating_point() or indices.is_complex():
            raise ValueError("indices must have an integer dtype.")
        if indices.numel() > 0 and (
            bool((indices < 0).any()) or bool((indices >= self.codebook_size).any())
        ):
            raise ValueError(
                f"indices must be in [0, {self.codebook_size}), got out-of-range ids."
            )

        quantized = torch.zeros(
            (*indices.shape[:-1], self.dim),
            device=indices.device,
            dtype=self.codebooks.dtype,
        )
        for level in range(self.num_quantizers):
            quantized = quantized + F.embedding(
                indices[..., level].long(),
                self.codebooks[level],
            )
        return quantized

    def _validate_inputs(
        self,
        latents: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if latents.ndim != 3:
            raise ValueError(
                f"latents must have shape [B, M, H], got {tuple(latents.shape)}."
            )
        if latents.shape[-1] != self.dim:
            raise ValueError(
                "latent dimension must match quantizer dim, "
                f"got {latents.shape[-1]} and {self.dim}."
            )
        if not latents.is_floating_point():
            raise ValueError("latents must have a floating-point dtype.")

        expected_shape = latents.shape[:2]
        if valid_mask is None:
            return torch.ones(
                expected_shape,
                device=latents.device,
                dtype=torch.bool,
            )
        if valid_mask.shape != expected_shape:
            raise ValueError(
                "valid_mask must have shape [B, M] matching latents, "
                f"got {tuple(valid_mask.shape)} and {tuple(expected_shape)}."
            )
        if valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must have boolean dtype.")
        if valid_mask.device != latents.device:
            raise ValueError("valid_mask and latents must be on the same device.")
        return valid_mask

    def _nearest_indices(self, latents: Tensor, level: int) -> Tensor:
        with torch.autocast(device_type=latents.device.type, enabled=False):
            flat = latents.float().reshape(-1, self.dim)
            codebook = self.codebooks[level]
            distances = (
                flat.square().sum(dim=-1, keepdim=True)
                - 2.0 * flat @ codebook.t()
                + codebook.square().sum(dim=-1).unsqueeze(0)
            )
        return distances.argmin(dim=-1).reshape(latents.shape[:2])

    def _assignment_counts(self, indices: Tensor, valid_mask: Tensor) -> Tensor:
        valid_indices = indices.reshape(-1)[valid_mask.reshape(-1)]
        return torch.bincount(
            valid_indices,
            minlength=self.codebook_size,
        ).to(dtype=torch.float32)

    def _masked_mse(
        self,
        inputs: Tensor,
        targets: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        flat_mask = valid_mask.reshape(-1)
        valid_inputs = inputs.reshape(-1, self.dim)[flat_mask]
        valid_targets = targets.reshape(-1, self.dim)[flat_mask]
        squared_error = (valid_inputs - valid_targets).square()
        denominator = max(valid_inputs.numel(), 1)
        return squared_error.sum() / denominator

    def _perplexity(self, counts: Tensor) -> Tensor:
        total = counts.sum()
        probabilities = counts / total.clamp_min(1.0)
        entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum()
        return torch.where(total > 0, entropy.exp(), total)

    @torch.no_grad()
    def _ema_update(
        self,
        level: int,
        latents: Tensor,
        indices: Tensor,
        counts: Tensor,
        valid_mask: Tensor,
    ) -> None:
        flat_mask = valid_mask.reshape(-1)
        reduced_counts = counts.clone()
        self._all_reduce_sum(reduced_counts)
        if not bool(reduced_counts.sum()):
            return

        valid_latents = latents.reshape(-1, self.dim)[flat_mask].float()
        valid_indices = indices.reshape(-1)[flat_mask]
        sums = torch.zeros_like(self.ema_sums[level])
        if valid_indices.numel() > 0:
            sums.index_add_(0, valid_indices, valid_latents)
        self._all_reduce_sum(sums)

        self.ema_counts[level].mul_(self.decay).add_(
            reduced_counts,
            alpha=1.0 - self.decay,
        )
        self.ema_sums[level].mul_(self.decay).add_(
            sums,
            alpha=1.0 - self.decay,
        )

        active = self.ema_counts[level] > 0
        normalized = self.ema_sums[level] / self.ema_counts[level].clamp_min(
            1e-5
        ).unsqueeze(-1)
        self.codebooks[level].copy_(
            torch.where(active.unsqueeze(-1), normalized, self.codebooks[level])
        )
        self._reset_dead_codes(level, valid_latents)

    @torch.no_grad()
    def _reset_dead_codes(self, level: int, valid_latents: Tensor) -> None:
        if self.dead_code_threshold <= 0:
            return

        dead = self.ema_counts[level] < self.dead_code_threshold
        num_dead = int(dead.sum().item())
        if num_dead == 0:
            return

        source_rank = self._replacement_source_rank(valid_latents)
        if source_rank is None:
            return

        distributed = self._distributed_is_initialized()
        samples_on_rank = not distributed or dist.get_rank() == source_rank
        if samples_on_rank:
            replacements = self._sample_replacements(valid_latents, num_dead)
        else:
            replacements = torch.empty(
                num_dead,
                self.dim,
                device=valid_latents.device,
                dtype=torch.float32,
            )
        if distributed:
            dist.broadcast(replacements, src=source_rank)

        replacement_count = float(self.dead_code_threshold)

        self.codebooks[level, dead] = replacements
        self.ema_counts[level, dead] = replacement_count
        self.ema_sums[level, dead] = replacements * replacement_count

    def _replacement_source_rank(self, valid_latents: Tensor) -> int | None:
        if not self._distributed_is_initialized():
            return 0 if valid_latents.shape[0] > 0 else None

        world_size = dist.get_world_size()
        source_rank = torch.tensor(
            dist.get_rank() if valid_latents.shape[0] > 0 else world_size,
            device=valid_latents.device,
            dtype=torch.long,
        )
        dist.all_reduce(source_rank, op=dist.ReduceOp.MIN)
        selected = int(source_rank.item())
        return None if selected == world_size else selected

    @staticmethod
    def _sample_replacements(candidates: Tensor, num_dead: int) -> Tensor:
        num_latents = int(candidates.shape[0])
        if num_latents >= num_dead:
            sampled = torch.randperm(
                num_latents,
                device=candidates.device,
            )[:num_dead]
        else:
            sampled = torch.randint(
                num_latents,
                (num_dead,),
                device=candidates.device,
            )
        return candidates[sampled]

    @staticmethod
    def _distributed_is_initialized() -> bool:
        return dist.is_available() and dist.is_initialized()

    @classmethod
    def _all_reduce_sum(cls, tensor: Tensor) -> None:
        if cls._distributed_is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    codebooks: Tensor
    ema_counts: Tensor
    ema_sums: Tensor
    initialized: Tensor


def _q0_straight_through(latents: Tensor, q0_codewords: Tensor) -> Tensor:
    """Use hard q0 values in forward while routing gradients to encoder latents."""
    if latents.shape != q0_codewords.shape:
        raise ValueError("latents and q0_codewords must have matching shapes.")
    if not latents.is_floating_point() or not q0_codewords.is_floating_point():
        raise ValueError("latents and q0_codewords must be floating-point tensors.")
    if latents.device != q0_codewords.device:
        raise ValueError("latents and q0_codewords must share a device.")
    hard_q0 = q0_codewords.detach().to(dtype=latents.dtype)
    return latents + (hard_q0 - latents).detach()


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


@torch.no_grad()
def _kmeans(
    samples: Tensor,
    *,
    num_clusters: int,
    num_iters: int,
) -> tuple[Tensor, Tensor]:
    num_samples = int(samples.shape[0])
    if num_samples >= num_clusters:
        initial_indices = torch.randperm(
            num_samples,
            device=samples.device,
        )[:num_clusters]
    else:
        initial_indices = torch.randint(
            num_samples,
            (num_clusters,),
            device=samples.device,
        )
    centers = samples[initial_indices].float()

    for _ in range(num_iters):
        assignments = _euclidean_assignments(samples, centers)
        counts = torch.bincount(assignments, minlength=num_clusters)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, samples.float())
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(-1)

    assignments = _euclidean_assignments(samples, centers)
    counts = torch.bincount(
        assignments,
        minlength=num_clusters,
    ).float()
    return centers, counts


def _euclidean_assignments(samples: Tensor, centers: Tensor) -> Tensor:
    distances = (
        samples.float().square().sum(dim=-1, keepdim=True)
        - 2.0 * samples.float() @ centers.t()
        + centers.square().sum(dim=-1).unsqueeze(0)
    )
    return distances.argmin(dim=-1)


__all__ = ["ResidualQuantizerOutput", "SemanticResidualQuantizer"]
