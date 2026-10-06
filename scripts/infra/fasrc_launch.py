#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""submitit Slurm launcher for Praxis experiments on the FASRC Slurm surface.

Usage:
    uv run python scripts/infra/fasrc_launch.py \\
        train-tokenizer --codec oat --dataset libero_10 --partition kempner

    uv run python scripts/infra/fasrc_launch.py \\
        train-policy --method bar --codec oat --backbone pali --dataset libero_10 \\
        --tokenizer-ckpt <codec-checkpoint>/pretrained_model

    uv run python scripts/infra/fasrc_launch.py \\
        train-policy --method bar --codec fast --backbone qwen --dataset libero_spatial \\
        --tokenizer-ckpt <codec-checkpoint>/pretrained_model

    uv run python scripts/infra/fasrc_launch.py \\
        eval-monitor --train-dir <policy-train-output-dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

_INFRA_DIR = Path(__file__).resolve().parent
if str(_INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(_INFRA_DIR))

from launch.source_checkout import install_source_checkout_imports  # noqa: E402

install_source_checkout_imports(_INFRA_DIR)

from launch.partitions import (  # noqa: E402
    partition_spec_has_suffix,
    reject_mixed_standalone_partitions,
    resolve_mapped_account_for_partition_spec,
    resolve_required_account_env,
    validate_partition_list as validate_known_partitions,
)
from launch.profile import SlurmClusterProfile  # noqa: E402
from launch.runner import run_launcher  # noqa: E402

PARTITIONS = (
    "kempner",
    "kempner_h100",
    "kempner_interactive",
    "kempner_requeue",
    "remoteviz",
    "test",
    "serial_requeue",
    "shared",
    "sapphire",
    "unrestricted",
    "seas_compute",
    "gpu",
    "gpu_h200",
    "gpu_requeue",
    "gpu_test",
    "seas_gpu",
)
KEMPNER_ACCOUNT_ENV = "PRAXIS_FASRC_KEMPNER_ACCOUNT"
DEFAULT_ACCOUNT_ENV = "PRAXIS_FASRC_ACCOUNT"
PARTITION_ACCOUNT_ENV = {
    "kempner": KEMPNER_ACCOUNT_ENV,
    "kempner_h100": KEMPNER_ACCOUNT_ENV,
    "kempner_interactive": KEMPNER_ACCOUNT_ENV,
    "kempner_requeue": KEMPNER_ACCOUNT_ENV,
    "remoteviz": DEFAULT_ACCOUNT_ENV,
    "test": DEFAULT_ACCOUNT_ENV,
    "serial_requeue": DEFAULT_ACCOUNT_ENV,
    "shared": DEFAULT_ACCOUNT_ENV,
    "sapphire": DEFAULT_ACCOUNT_ENV,
    "unrestricted": DEFAULT_ACCOUNT_ENV,
    "seas_compute": DEFAULT_ACCOUNT_ENV,
    "gpu": DEFAULT_ACCOUNT_ENV,
    "gpu_h200": DEFAULT_ACCOUNT_ENV,
    "gpu_requeue": DEFAULT_ACCOUNT_ENV,
    "gpu_test": DEFAULT_ACCOUNT_ENV,
    "seas_gpu": DEFAULT_ACCOUNT_ENV,
}


def validate_partition_list(partitions: list[str]) -> list[str]:
    validate_known_partitions(partitions, supported_partitions=PARTITIONS)
    reject_mixed_standalone_partitions(
        partitions,
        standalone_partitions={"gpu_test", "kempner_requeue"},
    )
    return partitions


def _resolve_partition(
    stage: str,
    backbone: str,
    requested_partition: str | None,
) -> str:
    if requested_partition is not None:
        return requested_partition
    if stage == "policy":
        if backbone == "smol":
            return "kempner"
        if backbone in {"qwen", "pali", "molmo"}:
            return "kempner_h100"
    return "kempner"


def _resolve_partition_spec(
    stage: str,
    backbone: str,
    requested_partitions: list[str],
) -> str:
    if requested_partitions:
        return ",".join(requested_partitions)
    return _resolve_partition(stage, backbone, None)


def _resolve_account_for_partition_spec(partition_spec: str) -> str:
    account_map = {
        partition: resolve_required_account_env(
            env_var,
            context=f"FASRC partition {partition}",
        )
        for partition, env_var in PARTITION_ACCOUNT_ENV.items()
    }
    return resolve_mapped_account_for_partition_spec(
        partition_spec,
        partition_account_map=account_map,
        supported_partitions=PARTITIONS,
    )


def _is_requeue_partition_spec(partition_spec: str) -> bool:
    return partition_spec_has_suffix(partition_spec, "requeue")


FASRC_PROFILE = SlurmClusterProfile(
    validate_partition_list=validate_partition_list,
    resolve_partition_spec=_resolve_partition_spec,
    resolve_account_for_partition_spec=_resolve_account_for_partition_spec,
    is_requeue_partition_spec=_is_requeue_partition_spec,
)


def main() -> None:
    run_launcher(profile=FASRC_PROFILE)


if __name__ == "__main__":
    main()
