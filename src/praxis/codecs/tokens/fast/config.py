# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for FASTTokenizer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from praxis.codecs.core import build_dataclass_config, dataclass_config_to_dict

FAST_BPE_TOKENIZER_PATH = "physical-intelligence/fast"


@dataclass
class FASTTokenizerConfig:
    """Hyperparameters for the FAST (DCT + BPE) action tokenizer."""

    name: str = "fast"
    pretrained_path: str | None = None

    action_dim: int | None = None
    action_horizon: int = 32

    # Robust action normalization used before DCT/BPE.  The default preserves
    # the historical dataset min/max contract; quantile_01_99 maps per-DOF
    # q01/q99 to [-1, 1] and can clip outliers for robust FAST fitting.
    action_normalization_mode: str = "dataset_min_max"
    clip_quantile_normalized_actions: bool = True

    scale: float = 10.0
    min_token: int = 0
    max_token_length: int = 128
    strict_decode: bool = False
    bpe_tokenizer_path: str | None = FAST_BPE_TOKENIZER_PATH
    vocab_size: int = 1024

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> FASTTokenizerConfig:
        return build_dataclass_config(cls, cfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclass_config_to_dict(self)


def default_fast_tokenizer_config() -> dict[str, object]:
    """Return a fresh default codec-factory config payload for FAST."""
    return {
        "kind": "tokens",
        "name": "fast",
        "pretrained_path": None,
        "bpe_tokenizer_path": FAST_BPE_TOKENIZER_PATH,
        "action_normalization_mode": "dataset_min_max",
        "clip_quantile_normalized_actions": True,
    }
