#!/usr/bin/env python3
"""submitit Slurm launcher for Praxis experiments on PSC Bridges-2.

Usage:
    uv run python scripts/infra/psc_launch.py \
        train-tokenizer --codec oat --dataset libero_10 --partition GPU-shared

    uv run python scripts/infra/psc_launch.py \
        train-policy --method bar --codec oat --backbone pali --dataset libero_10 \
        --tokenizer-ckpt outputs/train_codecs/20260301/libero10_oat_codec/checkpoints/latest/pretrained_model

    uv run python scripts/infra/psc_launch.py \
        train-policy --method bar --codec fast --backbone qwen --dataset libero_spatial \
        --tokenizer-ckpt outputs/train_codecs/20260301/liberospatial_fast_codec/checkpoints/latest/pretrained_model

    uv run python scripts/infra/psc_launch.py \
        eval-monitor --train-dir outputs/train_policy/20260312/libero10_bar_oat_qwen
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
    resolve_required_account_env,
    resolve_static_account_for_partition_spec,
    validate_partition_list as validate_known_partitions,
)
from launch.profile import SlurmClusterProfile  # noqa: E402
from launch.runner import run_launcher  # noqa: E402

PARTITIONS = (
    "GPU-shared",
    "GPU",
    "GPU-small",
)
PSC_ACCOUNT_ENV = "PRAXIS_PSC_ACCOUNT"


def validate_partition_list(partitions: list[str]) -> list[str]:
    return validate_known_partitions(partitions, supported_partitions=PARTITIONS)


def _resolve_partition(
    stage: str,
    backbone: str,
    requested_partition: str | None,
) -> str:
    del stage, backbone
    if requested_partition is not None:
        return requested_partition
    return "GPU-shared"


def _resolve_partition_spec(
    stage: str,
    backbone: str,
    requested_partitions: list[str],
) -> str:
    if requested_partitions:
        return ",".join(requested_partitions)
    return _resolve_partition(stage, backbone, None)


def _resolve_account_for_partition_spec(partition_spec: str) -> str:
    return resolve_static_account_for_partition_spec(
        partition_spec,
        supported_partitions=PARTITIONS,
        account=resolve_required_account_env(
            PSC_ACCOUNT_ENV,
            context="PSC Bridges-2",
        ),
    )


def _is_requeue_partition_spec(partition_spec: str) -> bool:
    del partition_spec
    return False


def _resolve_gres(gpus: int, requested_gres: str | None) -> str | None:
    if requested_gres is not None:
        return requested_gres
    if gpus <= 0:
        return None
    return f"gpu:{gpus}"


PSC_PROFILE = SlurmClusterProfile(
    validate_partition_list=validate_partition_list,
    resolve_partition_spec=_resolve_partition_spec,
    resolve_account_for_partition_spec=_resolve_account_for_partition_spec,
    is_requeue_partition_spec=_is_requeue_partition_spec,
)


def main() -> None:
    run_launcher(profile=PSC_PROFILE, resolve_gres=_resolve_gres)


if __name__ == "__main__":
    main()
