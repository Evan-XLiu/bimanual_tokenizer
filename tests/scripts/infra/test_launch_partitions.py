# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared Slurm partition helpers."""

from __future__ import annotations

import pytest

from scripts.infra.launch.partitions import (
    partition_spec_has_suffix,
    reject_mixed_standalone_partitions,
    resolve_mapped_account_for_partition_spec,
    resolve_required_account_env,
    resolve_static_account_for_partition_spec,
    split_partition_spec,
    validate_partition_list,
)


def test_split_partition_spec_strips_empty_csv_entries() -> None:
    assert split_partition_spec(" gpu, ,gpu_h200,, seas_gpu ") == [
        "gpu",
        "gpu_h200",
        "seas_gpu",
    ]


def test_validate_partition_list_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported test partition"):
        validate_partition_list(
            ["known", "unknown"],
            supported_partitions=("known",),
            label="test partition",
        )


def test_reject_mixed_standalone_partitions() -> None:
    with pytest.raises(ValueError, match="standalone partition"):
        reject_mixed_standalone_partitions(
            ["gpu_test", "gpu"],
            standalone_partitions={"gpu_test"},
        )


def test_static_account_resolution_validates_partition_spec() -> None:
    assert (
        resolve_static_account_for_partition_spec(
            "gpu,gpu_h200",
            supported_partitions=("gpu", "gpu_h200"),
            account="acct",
        )
        == "acct"
    )
    with pytest.raises(ValueError, match="partition_spec"):
        resolve_static_account_for_partition_spec(
            "",
            supported_partitions=("gpu",),
            account="acct",
        )


def test_required_account_env_errors_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRAXIS_TEST_ACCOUNT", raising=False)
    with pytest.raises(RuntimeError, match="PRAXIS_TEST_ACCOUNT"):
        resolve_required_account_env("PRAXIS_TEST_ACCOUNT", context="test cluster")


def test_mapped_account_resolution_rejects_mixed_accounts() -> None:
    with pytest.raises(ValueError, match="mixes account families"):
        resolve_mapped_account_for_partition_spec(
            "kempner,gpu",
            partition_account_map={"kempner": "acct_a", "gpu": "acct_b"},
            supported_partitions=("kempner", "gpu"),
        )


def test_partition_spec_has_suffix() -> None:
    assert partition_spec_has_suffix("gpu,gpu_requeue", "requeue") is True
    assert partition_spec_has_suffix("gpu,gpu_h200", "requeue") is False
