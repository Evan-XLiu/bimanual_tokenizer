# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared vector quantization module (EMA codebook updates)."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def _sample_vectors(samples: Tensor, num: int) -> Tensor:
    n = int(samples.shape[0])
    if n >= num:
        indices = torch.randperm(n, device=samples.device)[:num]
    else:
        indices = torch.randint(0, n, (num,), device=samples.device)
    return samples[indices].float()


@torch.no_grad()
def _kmeans(
    samples: Tensor, num_clusters: int, num_iters: int
) -> tuple[Tensor, Tensor]:
    dim = int(samples.shape[-1])
    means = _sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        dists = (
            samples.float().pow(2).sum(1, keepdim=True)
            - 2 * samples.float() @ means.t()
            + means.t().float().pow(2).sum(0, keepdim=True)
        )
        buckets = (-dists).argmax(dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_clamped = bins.masked_fill(zero_mask, 1)

        new_means = torch.zeros(
            (num_clusters, dim), device=samples.device, dtype=torch.float32
        )
        new_means.scatter_add_(0, buckets.unsqueeze(1).expand(-1, dim), samples.float())
        new_means = new_means / bins_clamped.unsqueeze(1)
        means = torch.where(zero_mask.unsqueeze(1), means, new_means)

    dists = (
        samples.float().pow(2).sum(1, keepdim=True)
        - 2 * samples.float() @ means.t()
        + means.t().float().pow(2).sum(0, keepdim=True)
    )
    buckets = (-dists).argmax(dim=-1)
    bins = torch.bincount(buckets, minlength=num_clusters).float()
    return means, bins


class VectorQuantization(nn.Module):
    """EMA vector quantizer inspired by vector-quantize-pytorch style usage.

    Input shape: ``(B, L, D_in)``
    Output:
    - quantized: ``(B, L, D_in)``
    - indices: ``(B, L)``
    - commitment_loss: scalar tensor
    - codebook_loss: scalar tensor (always 0 for EMA updates)
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        commitment: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead: int = 2,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim or input_dim)
        self.commitment = float(commitment)
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.threshold_ema_dead = int(threshold_ema_dead)
        self.kmeans_init = bool(kmeans_init)
        self.kmeans_iters = int(kmeans_iters)

        if self.input_dim != self.codebook_dim:
            self.in_project = nn.Linear(self.input_dim, self.codebook_dim)
            self.out_project = nn.Linear(self.codebook_dim, self.input_dim)
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        init_fn = torch.zeros if self.kmeans_init else torch.randn
        self.register_buffer(
            "codebook", init_fn(self.codebook_size, self.codebook_dim).float()
        )
        self.register_buffer(
            "inited", torch.tensor([not self.kmeans_init], dtype=torch.bool)
        )
        self.register_buffer("cluster_size", torch.ones(self.codebook_size).float())
        self.register_buffer("embed_avg", self.codebook.clone().float())

    @torch.no_grad()
    def _ema_inplace(self, moving_avg: Tensor, new: Tensor) -> None:
        moving_avg.data.mul_(self.decay).add_(new.float(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def _update_codebook_from_ema(self) -> None:
        cluster_size = (self.cluster_size + self.epsilon) / (
            self.cluster_size.sum() + self.codebook_size * self.epsilon
        )
        cluster_size = cluster_size * self.cluster_size.sum()
        self.codebook.copy_(self.embed_avg / cluster_size.unsqueeze(1))

    @torch.no_grad()
    def _init_codebook(self, encodings: Tensor) -> None:
        if bool(self.inited.item()):
            return

        if not dist.is_initialized() or dist.get_rank() == 0:
            embed, cluster_sizes = _kmeans(
                encodings.float(), self.codebook_size, self.kmeans_iters
            )
        else:
            embed = torch.zeros(
                self.codebook_size,
                self.codebook_dim,
                device=encodings.device,
                dtype=torch.float32,
            )
            cluster_sizes = torch.zeros(
                self.codebook_size,
                device=encodings.device,
                dtype=torch.float32,
            )

        if dist.is_initialized():
            dist.broadcast(embed, src=0)
            dist.broadcast(cluster_sizes, src=0)

        self.cluster_size.copy_(cluster_sizes)
        self.embed_avg.copy_(embed * cluster_sizes.unsqueeze(1))
        self._update_codebook_from_ema()
        self.inited.fill_(True)

    @torch.no_grad()
    def _replace_dead_codes(self, encodings: Tensor) -> None:
        if self.threshold_ema_dead <= 0:
            return
        dead_mask = self.cluster_size < self.threshold_ema_dead
        if not bool(dead_mask.any().item()):
            return
        num_dead = int(dead_mask.sum().item())

        if not dist.is_initialized() or dist.get_rank() == 0:
            samples = _sample_vectors(encodings.float(), num_dead)
        else:
            samples = torch.zeros(
                num_dead,
                self.codebook_dim,
                device=encodings.device,
                dtype=torch.float32,
            )

        if dist.is_initialized():
            dist.broadcast(samples, src=0)

        replacement_cluster = torch.full(
            (num_dead,),
            float(self.threshold_ema_dead),
            device=encodings.device,
            dtype=torch.float32,
        )
        self.codebook[dead_mask] = samples.to(dtype=self.codebook.dtype)
        self.embed_avg[dead_mask] = (samples * replacement_cluster.unsqueeze(1)).to(
            dtype=self.embed_avg.dtype
        )
        self.cluster_size[dead_mask] = replacement_cluster.to(
            dtype=self.cluster_size.dtype
        )

    @torch.no_grad()
    def _ema_update(self, encodings: Tensor, onehot: Tensor) -> None:
        cluster_size_new = onehot.sum(dim=0)
        embed_sum = encodings.t() @ onehot

        if dist.is_initialized():
            dist.all_reduce(cluster_size_new, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        self._ema_inplace(self.cluster_size, cluster_size_new)
        self._ema_inplace(self.embed_avg, embed_sum.t())
        self._update_codebook_from_ema()

    def _lookup_codebook(self, z_e: Tensor) -> Tensor:
        encodings = z_e.reshape(-1, self.codebook_dim)
        dists = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ self.codebook.t()
            + self.codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dists).argmax(dim=1)
        return indices.reshape(z_e.shape[0], z_e.shape[1])

    def indices_to_embedding(self, indices: Tensor) -> Tensor:
        return F.embedding(indices.long(), self.codebook)

    def indices_to_quantized(self, indices: Tensor) -> Tensor:
        z_q = self.indices_to_embedding(indices)
        return self.out_project(z_q)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        z_e = self.in_project(z.float())
        flat = z_e.reshape(-1, self.codebook_dim)

        if self.kmeans_init and not bool(self.inited.item()):
            self._init_codebook(flat)

        indices = self._lookup_codebook(z_e)
        z_q = self.indices_to_embedding(indices).float()

        commitment_loss = F.mse_loss(z_e, z_q.detach()) * self.commitment
        codebook_loss = torch.tensor(0.0, device=z.device, dtype=torch.float32)

        if self.training and torch.is_grad_enabled():
            onehot = F.one_hot(indices.reshape(-1), self.codebook_size).float()
            self._ema_update(flat, onehot)
            self._replace_dead_codes(flat)

        z_q = (z_q - z_e).detach() + z_e
        z_q = self.out_project(z_q).float()
        return z_q, indices.long(), commitment_loss, codebook_loss

    codebook: Tensor
    inited: Tensor
    cluster_size: Tensor
    embed_avg: Tensor


__all__ = ["VectorQuantization"]
