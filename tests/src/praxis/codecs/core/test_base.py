# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared codec contract."""

from __future__ import annotations

import torch

from praxis.codecs.core import BaseCodec
from praxis.core.types import NormMode


def test_base_codec_declares_generic_code_sequence_defaults() -> None:
    codec = BaseCodec()

    assert codec.variable_length_codes is False
    assert codec.max_code_seq_len is None
    assert codec.supports_prefix_decode is False
    assert not hasattr(codec, "block_schedule")


def test_base_codec_normalization_checkpoint_payload_preserves_schema() -> None:
    codec = BaseCodec()
    codec.set_normalization_stats(
        {
            "action": {
                "mean": torch.tensor([1.0, 2.0]),
                "std": torch.tensor([2.0, 4.0]),
            }
        },
        modes={"action": NormMode.MEAN_STD},
    )

    payload = codec._normalization_checkpoint_payload()
    restored = BaseCodec()
    restored._load_normalization_checkpoint_payload(payload)

    assert set(payload) == {"stats", "modes"}
    assert payload["modes"] == {"action": NormMode.MEAN_STD.value}
    assert restored.normalize_action is not None
    torch.testing.assert_close(
        restored.normalize_action.stats["action"]["mean"],
        torch.tensor([1.0, 2.0]),
    )
