# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared checkpoint artifact resolution helpers."""

from __future__ import annotations

import pytest
from huggingface_hub.errors import EntryNotFoundError

from praxis.core import checkpoint_artifacts
from praxis.core.checkpoint_artifacts import (
    resolve_pretrained_weight_file,
    resolve_sidecar_path,
)


def test_resolve_sidecar_path_returns_none_for_missing_local_path(
    tmp_path, monkeypatch
):
    def _raise_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("missing absolute local paths must not call the Hub")

    monkeypatch.setattr(checkpoint_artifacts, "hf_hub_download", _raise_if_called)

    assert resolve_sidecar_path(tmp_path / "missing", "policy_io.json") is None


def test_resolve_sidecar_path_returns_none_for_missing_relative_local_path(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def _raise_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("missing relative local paths must not call the Hub")

    monkeypatch.setattr(checkpoint_artifacts, "hf_hub_download", _raise_if_called)

    assert (
        resolve_sidecar_path(
            "outputs/train_policy/run/checkpoints/latest/pretrained_model",
            "policy_io.json",
        )
        is None
    )


def test_resolve_pretrained_weight_file_rejects_missing_relative_local_path(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def _raise_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("missing relative local paths must not call the Hub")

    monkeypatch.setattr(checkpoint_artifacts, "hf_hub_download", _raise_if_called)

    with pytest.raises(FileNotFoundError, match="Pretrained model path does not exist"):
        resolve_pretrained_weight_file(
            "outputs/train_policy/run/checkpoints/latest/pretrained_model"
        )


def test_resolve_sidecar_path_returns_none_for_missing_hub_entry(monkeypatch):
    def _raise_entry_not_found(*, repo_id, filename, **kwargs):
        del repo_id, filename, kwargs
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(checkpoint_artifacts, "hf_hub_download", _raise_entry_not_found)

    assert resolve_sidecar_path("org/missing-sidecar", "policy_io.json") is None


def test_resolve_sidecar_path_propagates_non_missing_hub_errors(monkeypatch):
    def _raise_network_error(*, repo_id, filename, **kwargs):
        del repo_id, filename, kwargs
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(checkpoint_artifacts, "hf_hub_download", _raise_network_error)

    with pytest.raises(RuntimeError, match="network unavailable"):
        resolve_sidecar_path("org/network-error", "policy_io.json")
