# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared primitives for action codecs."""

from praxis.codecs.core.base import BaseCodec
from praxis.codecs.core.config import (
    build_dataclass_config,
    dataclass_config_to_dict,
)
from praxis.codecs.core.prefix import pad_code_sequence

__all__ = [
    "BaseCodec",
    "build_dataclass_config",
    "dataclass_config_to_dict",
    "pad_code_sequence",
]
