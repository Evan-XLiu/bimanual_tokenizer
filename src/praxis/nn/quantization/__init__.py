# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared quantization modules."""

from praxis.nn.quantization.fsq import FiniteScalarQuantization
from praxis.nn.quantization.vq import VectorQuantization

__all__ = [
    "FiniteScalarQuantization",
    "VectorQuantization",
]
