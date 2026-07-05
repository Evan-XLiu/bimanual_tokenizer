# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import pytest
import torch

from scripts.tokenizer.boat import independence_diagnostic
from scripts.tokenizer.fast import decode_failures


class _FakeBpe:
    def batch_decode(self, rows):
        return ["x" * len(row) for row in rows]


class _FakeFastTokenizer:
    action_horizon = 1
    action_dim = 2
    _pad_token_id = 0

    def __init__(self) -> None:
        self.strict_decode_values: list[object] = []

    def _require_bpe(self) -> _FakeBpe:
        return _FakeBpe()

    def detokenize(self, tokens, *, strict_decode=None, **kwargs):
        del kwargs
        self.strict_decode_values.append(strict_decode)
        return tokens


def test_boat_independence_diagnostic_help_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["independence_diagnostic.py", "--help"],
    )

    with pytest.raises(SystemExit) as exc_info:
        independence_diagnostic.parse_args()

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--tokenizer-path" in output
    assert "--data-repo-id" in output
    assert "--delta-threshold" in output


def test_fast_decode_failures_help_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["decode_failures.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        decode_failures.parse_args()

    assert exc_info.value.code == 0
    assert "scripts/tokenizer/fast/decode_failures.py" in (
        decode_failures.__doc__ or ""
    )
    assert "--policy" in capsys.readouterr().out


def test_fast_decode_failure_hook_counts_strict_failures_but_decodes_leniently() -> (
    None
):
    tokenizer = _FakeFastTokenizer()
    stats = decode_failures.install_detokenize_hook(tokenizer)

    result = tokenizer.detokenize(
        torch.tensor(
            [
                [1, 2, 0],
                [1, 0, 0],
            ]
        ),
        strict_decode=True,
    )

    assert torch.equal(result, torch.tensor([[1, 2, 0], [1, 0, 0]]))
    assert stats == {"failures": 1, "total": 2}
    assert tokenizer.strict_decode_values == [False]
