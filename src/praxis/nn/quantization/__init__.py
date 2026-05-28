"""Shared quantization modules."""

from praxis.nn.quantization.fsq import FiniteScalarQuantization
from praxis.nn.quantization.vq import VectorQuantization

__all__ = [
    "FiniteScalarQuantization",
    "VectorQuantization",
]
