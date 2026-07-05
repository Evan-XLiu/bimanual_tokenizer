# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared neural network architectures."""

from praxis.nn.architectures.transformer import Transformer, TransformerDecoder

__all__ = [
    "Transformer",
    "TransformerDecoder",
]
