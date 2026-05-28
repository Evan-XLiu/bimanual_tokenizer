"""Tests for shared prefix-code helpers."""

from __future__ import annotations

import torch

from praxis.codecs.core import pad_code_sequence


def test_pad_code_sequence_pads_to_horizon() -> None:
    codes = torch.tensor([[1, 2]], dtype=torch.long)

    out = pad_code_sequence(codes, 4, pad_value=9)

    assert out.equal(torch.tensor([[1, 2, 9, 9]], dtype=torch.long))


def test_pad_code_sequence_pads_latent_prefixes() -> None:
    codes = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    out = pad_code_sequence(codes, 4)

    expected = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0], [0.0, 0.0]]])
    assert out.equal(expected)
