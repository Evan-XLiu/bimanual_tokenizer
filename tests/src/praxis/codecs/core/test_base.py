"""Tests for the shared codec contract."""

from __future__ import annotations

from praxis.codecs.core import BaseCodec


def test_base_codec_declares_generic_code_sequence_defaults() -> None:
    codec = BaseCodec()

    assert codec.variable_length_codes is False
    assert codec.max_code_seq_len is None
    assert codec.supports_prefix_decode is False
    assert not hasattr(codec, "block_schedule")
