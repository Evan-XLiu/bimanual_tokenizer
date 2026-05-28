#!/usr/bin/env python3
"""submitit Slurm launcher for Praxis experiments on NCSA Delta/DeltaAI.

Usage:
    uv run python scripts/infra/delta_launch.py \
        train-tokenizer --codec oat --dataset libero_10

    uv run python scripts/infra/delta_launch.py \
        train-policy --method bar --codec oat --backbone pali --dataset libero_10 \
        --tokenizer-ckpt outputs/train_codecs/20260301/libero10_oat_codec/checkpoints/latest/pretrained_model

    uv run python scripts/infra/delta_launch.py \
        eval-monitor --train-dir outputs/train_policy/20260312/libero10_bar_oat_qwen
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path

_INFRA_DIR = Path(__file__).resolve().parent
if str(_INFRA_DIR) not in sys.path:
    sys.path.insert(0, str(_INFRA_DIR))

from launch.source_checkout import install_source_checkout_imports  # noqa: E402

install_source_checkout_imports(_INFRA_DIR)

from launch.args import build_arg_parser as _shared_build_arg_parser  # noqa: E402
from launch.partitions import (  # noqa: E402
    partition_spec_has_suffix,
    resolve_required_account_env,
    split_partition_spec,
    validate_partition_list as validate_known_partitions,
)
from launch.profile import SlurmClusterProfile  # noqa: E402
from launch.runner import run_launcher  # noqa: E402


@dataclass(frozen=True)
class DeltaSurface:
    name: str
    slurm_cluster_name: str
    account_env: str
    default_partition: str
    partitions: tuple[str, ...]
    partition_gpu_capacity: dict[str, int]
    interactive_partitions: frozenset[str]
    interactive_max_time_min: int
    default_gres_name: str | None


DELTAAI_SURFACE = DeltaSurface(
    name="DeltaAI",
    slurm_cluster_name="delta-gh",
    account_env="PRAXIS_DELTAAI_ACCOUNT",
    default_partition="ghx4",
    partitions=(
        "ghx4",
        "ghx4-interactive",
    ),
    partition_gpu_capacity={
        "ghx4": 4,
        "ghx4-interactive": 4,
    },
    interactive_partitions=frozenset({"ghx4-interactive"}),
    interactive_max_time_min=120,
    default_gres_name="nvidia_gh200_120gb",
)

DELTA_SURFACE = DeltaSurface(
    name="Delta",
    slurm_cluster_name="delta",
    account_env="PRAXIS_DELTA_ACCOUNT",
    default_partition="gpuA40x4",
    partitions=(
        "gpuA40x4",
        "gpuA40x4-interactive",
        "gpuA40x4-preempt",
        "gpuA100x4",
        "gpuA100x4-interactive",
        "gpuA100x4-preempt",
        "gpuA100x8",
        "gpuA100x8-interactive",
        "gpuH200x8",
        "gpuH200x8-interactive",
    ),
    partition_gpu_capacity={
        "gpuA40x4": 4,
        "gpuA40x4-interactive": 4,
        "gpuA40x4-preempt": 4,
        "gpuA100x4": 4,
        "gpuA100x4-interactive": 4,
        "gpuA100x4-preempt": 4,
        "gpuA100x8": 8,
        "gpuA100x8-interactive": 8,
        "gpuH200x8": 8,
        "gpuH200x8-interactive": 8,
    },
    interactive_partitions=frozenset(
        {
            "gpuA40x4-interactive",
            "gpuA100x4-interactive",
            "gpuA100x8-interactive",
            "gpuH200x8-interactive",
        }
    ),
    interactive_max_time_min=60,
    default_gres_name=None,
)

SURFACES_BY_SLURM_CLUSTER = {
    DELTAAI_SURFACE.slurm_cluster_name: DELTAAI_SURFACE,
    DELTA_SURFACE.slurm_cluster_name: DELTA_SURFACE,
}
SURFACES_BY_OVERRIDE = {
    "deltaai": DELTAAI_SURFACE,
    "delta-gh": DELTAAI_SURFACE,
    "dtai": DELTAAI_SURFACE,
    "delta": DELTA_SURFACE,
    "classic-delta": DELTA_SURFACE,
}


def build_arg_parser() -> ArgumentParser:
    return _shared_build_arg_parser()


def _query_slurm_cluster_name() -> str | None:
    try:
        result = subprocess.run(
            ["scontrol", "show", "config"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    for line in result.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "ClusterName":
            return value.strip()
    return None


def detect_delta_surface() -> DeltaSurface:
    override = os.environ.get("PRAXIS_DELTA_SURFACE")
    if override:
        key = override.strip().lower()
        try:
            return SURFACES_BY_OVERRIDE[key]
        except KeyError as exc:
            supported = ", ".join(sorted(SURFACES_BY_OVERRIDE))
            raise RuntimeError(
                f"Unsupported PRAXIS_DELTA_SURFACE={override!r}. "
                f"Supported values: {supported}."
            ) from exc

    cluster_name = _query_slurm_cluster_name()
    if cluster_name in SURFACES_BY_SLURM_CLUSTER:
        return SURFACES_BY_SLURM_CLUSTER[cluster_name]

    hostname = socket.getfqdn()
    if hostname.startswith("gh") or ".deltaai." in hostname:
        return DELTAAI_SURFACE
    if hostname.startswith(("dt-", "gpub", "gpua", "gpuc", "gpue")):
        return DELTA_SURFACE

    known = ", ".join(sorted(SURFACES_BY_SLURM_CLUSTER))
    raise RuntimeError(
        "Could not identify this host as NCSA Delta or DeltaAI. "
        f"Slurm ClusterName={cluster_name!r}, hostname={hostname!r}. "
        f"Expected one of: {known}."
    )


def validate_partition_list(
    partitions: list[str], *, surface: DeltaSurface | None = None
) -> list[str]:
    active_surface = surface or detect_delta_surface()
    return validate_known_partitions(
        partitions,
        supported_partitions=active_surface.partitions,
        label=f"{active_surface.name} partition",
    )


def _resolve_partition(
    stage: str,
    backbone: str,
    requested_partition: str | None,
    *,
    surface: DeltaSurface | None = None,
) -> str:
    del stage, backbone
    if requested_partition is not None:
        return requested_partition
    active_surface = surface or detect_delta_surface()
    return active_surface.default_partition


def _resolve_partition_spec(
    stage: str,
    backbone: str,
    requested_partitions: list[str],
    *,
    surface: DeltaSurface | None = None,
) -> str:
    if requested_partitions:
        return ",".join(requested_partitions)
    return _resolve_partition(stage, backbone, None, surface=surface)


def _resolve_account_for_partition_spec(
    partition_spec: str, *, surface: DeltaSurface | None = None
) -> str:
    active_surface = surface or detect_delta_surface()
    parts = split_partition_spec(partition_spec)
    if not parts:
        raise ValueError("partition_spec must contain at least one partition.")

    unsupported = [part for part in parts if part not in active_surface.partitions]
    if unsupported:
        raise ValueError(
            f"Unsupported {active_surface.name} partition(s) in partition_spec: "
            f"{unsupported}. Supported partitions: {', '.join(active_surface.partitions)}."
        )
    return resolve_required_account_env(
        active_surface.account_env,
        context=active_surface.name,
    )


def _is_requeue_partition_spec(partition_spec: str) -> bool:
    return partition_spec_has_suffix(partition_spec, "-preempt")


def _resolve_gres(
    gpus: int, requested_gres: str | None, *, surface: DeltaSurface | None = None
) -> str | None:
    if requested_gres is not None:
        return requested_gres
    if gpus <= 0:
        return None
    active_surface = surface or detect_delta_surface()
    if active_surface.default_gres_name is None:
        return None
    return f"gpu:{active_surface.default_gres_name}:{gpus}"


def _validate_delta_args(
    parser: ArgumentParser,
    args: Namespace,
    requested_partitions: list[str],
    *,
    surface: DeltaSurface | None = None,
) -> None:
    active_surface = surface or detect_delta_surface()
    partition_spec = _resolve_partition_spec(
        str(args.stage),
        str(args.backbone),
        requested_partitions,
        surface=active_surface,
    )
    partitions = split_partition_spec(partition_spec)

    interactive = sorted(
        part for part in partitions if part in active_surface.interactive_partitions
    )
    if interactive and int(args.time) > active_surface.interactive_max_time_min:
        parser.error(
            f"{active_surface.name} interactive partition(s) {interactive} have "
            f"MaxTime={active_surface.interactive_max_time_min}min; pass "
            f"--time <= {active_surface.interactive_max_time_min} or use a batch "
            "partition."
        )

    gpus = int(args.gpus) if args.gpus is not None else None
    if gpus is None:
        return

    too_small = [
        part
        for part in partitions
        if gpus > active_surface.partition_gpu_capacity[part]
    ]
    if too_small:
        capacities = {
            part: active_surface.partition_gpu_capacity[part] for part in too_small
        }
        parser.error(
            f"Requested --gpus {gpus}, but {active_surface.name} partition(s) "
            f"cannot satisfy that per-node GPU count: {capacities}."
        )


def build_delta_profile(surface: DeltaSurface | None = None) -> SlurmClusterProfile:
    return SlurmClusterProfile(
        validate_partition_list=lambda partitions: validate_partition_list(
            partitions, surface=surface
        ),
        resolve_partition_spec=lambda stage, backbone, partitions: (
            _resolve_partition_spec(stage, backbone, partitions, surface=surface)
        ),
        resolve_account_for_partition_spec=lambda partition_spec: (
            _resolve_account_for_partition_spec(partition_spec, surface=surface)
        ),
        is_requeue_partition_spec=_is_requeue_partition_spec,
    )


def main() -> None:
    run_launcher(
        profile=build_delta_profile(),
        cluster_arg_validator=lambda parser, args, partitions: _validate_delta_args(
            parser, args, partitions
        ),
        resolve_gres=lambda gpus, requested_gres: _resolve_gres(gpus, requested_gres),
    )


if __name__ == "__main__":
    main()
