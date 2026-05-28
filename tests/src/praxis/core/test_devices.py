from __future__ import annotations

import pytest
import torch

from praxis.core.devices import require_available_device


def test_require_available_device_allows_cpu() -> None:
    assert require_available_device("cpu", purpose="Test") == torch.device("cpu")


def test_require_available_device_rejects_cuda_without_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="device=cuda"):
        require_available_device("cuda", purpose="Test")


def test_require_available_device_allows_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert require_available_device("cuda:0", purpose="Test") == torch.device("cuda:0")
