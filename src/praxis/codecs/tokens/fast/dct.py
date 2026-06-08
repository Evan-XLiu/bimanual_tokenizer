# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Pure PyTorch DCT-II and IDCT-II (DCT-III) transforms.

Replaces ``scipy.fft.dct``/``idct`` with orthonormal matrix multiplies so
the entire FAST tokenizer pipeline stays on-device and torch-scriptable.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch import Tensor


@lru_cache(maxsize=32)
def _dct_matrix(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Build the orthonormal DCT-II matrix of size ``(n, n)``.

    ``C[k, i] = w_k * cos(pi * k * (2*i + 1) / (2*n))``
    where ``w_0 = sqrt(1/n)`` and ``w_k = sqrt(2/n)`` for k > 0.
    """
    idx = torch.arange(n, dtype=dtype, device=device)
    # outer products for cosine argument: k * (2*i + 1) * pi / (2*n)
    k = idx.unsqueeze(1)  # (n, 1)
    i = idx.unsqueeze(0)  # (1, n)
    mat = torch.cos(torch.pi * k * (2.0 * i + 1.0) / (2.0 * n))
    # orthonormal scaling
    mat[0] *= (1.0 / n) ** 0.5
    mat[1:] *= (2.0 / n) ** 0.5
    return mat


def dct(x: Tensor, dim: int = -2) -> Tensor:
    """Orthonormal DCT-II along *dim* via matrix multiply.

    Default ``dim=-2`` matches the time-axis convention ``(B, T, D)``.
    """
    n = x.shape[dim]
    mat = _dct_matrix(n, x.dtype, x.device)  # (N, N)
    # Move target dim to -2 so we can matmul: (..., N, D) @ () -> (..., N, D)
    x = x.transpose(dim, -2)
    out = mat @ x  # (N, N) @ (..., N, D) -> (..., N, D)
    return out.transpose(dim, -2)


def idct(x: Tensor, dim: int = -2) -> Tensor:
    """Orthonormal IDCT (DCT-III) along *dim*.

    Since the DCT-II matrix is orthogonal, the inverse is just ``C^T @ x``.
    """
    n = x.shape[dim]
    mat = _dct_matrix(n, x.dtype, x.device)  # (N, N)
    x = x.transpose(dim, -2)
    out = mat.t() @ x
    return out.transpose(dim, -2)
