# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared vision encoders and utilities."""

from praxis.nn.vision.rgb_encoder import (
    ResNetSpatialSoftmaxEncoder,
    SpatialSoftmax,
    replace_bn_with_gn,
)

__all__ = [
    "ResNetSpatialSoftmaxEncoder",
    "SpatialSoftmax",
    "replace_bn_with_gn",
]
