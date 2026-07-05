# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for BinTokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from praxis.codecs.core import build_dataclass_config, dataclass_config_to_dict


@dataclass
class BinTokenizerConfig:
    """Hyperparameters for deterministic action binning."""

    name: str = "bin"
    pretrained_path: str | None = None

    action_dim: int | None = None
    action_horizon: int = 32

    num_bins: int = 256
    min_val: float = -1.0
    max_val: float = 1.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> BinTokenizerConfig:
        return build_dataclass_config(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)
