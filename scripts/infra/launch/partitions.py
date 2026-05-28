"""Shared Slurm partition parsing and validation helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


def split_partition_spec(partition_spec: str) -> list[str]:
    return [part.strip() for part in str(partition_spec).split(",") if part.strip()]


def validate_partition_list(
    partitions: list[str],
    *,
    supported_partitions: Sequence[str],
    label: str = "partition",
) -> list[str]:
    if not partitions:
        return []
    unknown = [part for part in partitions if part not in supported_partitions]
    if unknown:
        supported = ", ".join(supported_partitions)
        raise ValueError(
            f"Unsupported {label}(s): {unknown}. Supported partitions: {supported}."
        )
    return partitions


def resolve_required_account_env(env_var: str, *, context: str) -> str:
    account = os.environ.get(env_var, "").strip()
    if account:
        return account
    raise RuntimeError(
        f"Set {env_var} to the Slurm account for {context}. "
        "Public launcher templates do not ship personal account names."
    )


def reject_mixed_standalone_partitions(
    partitions: Sequence[str],
    *,
    standalone_partitions: set[str],
) -> None:
    mixed = [part for part in partitions if part in standalone_partitions]
    if mixed and len(partitions) > 1:
        names = ", ".join(sorted(mixed))
        raise ValueError(
            f"{names} must be used as a standalone partition "
            "(do not combine it with other partitions)."
        )


def resolve_static_account_for_partition_spec(
    partition_spec: str,
    *,
    supported_partitions: Sequence[str],
    account: str,
    label: str = "partition",
) -> str:
    parts = split_partition_spec(partition_spec)
    if not parts:
        raise ValueError("partition_spec must contain at least one partition.")
    validate_partition_list(
        parts,
        supported_partitions=supported_partitions,
        label=label,
    )
    return account


def resolve_mapped_account_for_partition_spec(
    partition_spec: str,
    *,
    partition_account_map: Mapping[str, str],
    supported_partitions: Sequence[str],
) -> str:
    parts = split_partition_spec(partition_spec)
    if not parts:
        raise ValueError("partition_spec must contain at least one partition.")

    unsupported = [part for part in parts if part not in partition_account_map]
    if unsupported:
        raise ValueError(
            "Unsupported partition(s) in partition_spec: "
            f"{unsupported}. Supported partitions: {', '.join(supported_partitions)}."
        )

    accounts = {partition_account_map[part] for part in parts}
    if len(accounts) > 1:
        raise ValueError(
            "Partition list mixes account families and cannot be submitted as one job. "
            f"partition_spec={partition_spec!r}. "
            "Submit separate jobs for kempner* vs non-kempner partitions."
        )
    return next(iter(accounts))


def partition_spec_has_suffix(partition_spec: str, suffix: str) -> bool:
    return any(part.endswith(suffix) for part in split_partition_spec(partition_spec))
