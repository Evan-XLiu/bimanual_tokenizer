"""Tests for scripts/infra/delta_launch.py."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from scripts.infra import delta_launch
from scripts.infra.launch.stages import resolve_stage_resources
from tests._paths import repo_root

DELTAAI_PROFILE = delta_launch.build_delta_profile(delta_launch.DELTAAI_SURFACE)
DELTA_PROFILE = delta_launch.build_delta_profile(delta_launch.DELTA_SURFACE)


@pytest.fixture(autouse=True)
def _set_public_launcher_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRAXIS_DELTAAI_ACCOUNT", "deltaai_account")
    monkeypatch.setenv("PRAXIS_DELTA_ACCOUNT", "delta_account")


def test_deltaai_default_resources_use_ghx4_and_gh200_gres() -> None:
    resources = resolve_stage_resources(
        stage="policy",
        backbone="pali",
        requested_partitions=[],
        gpus_override=None,
        mem_override=None,
        num_nodes=1,
        profile=DELTAAI_PROFILE,
    )

    assert resources.partition == "ghx4"
    assert resources.account == "deltaai_account"
    assert resources.gpus == 4
    assert delta_launch._resolve_gres(
        resources.gpus, None, surface=delta_launch.DELTAAI_SURFACE
    ) == ("gpu:nvidia_gh200_120gb:4")


def test_deltaai_eval_resources_use_one_gh200_by_default() -> None:
    resources = resolve_stage_resources(
        stage="eval",
        backbone="qwen",
        requested_partitions=[],
        gpus_override=None,
        mem_override=None,
        num_nodes=1,
        profile=DELTAAI_PROFILE,
    )

    assert resources.partition == "ghx4"
    assert resources.account == "deltaai_account"
    assert resources.gpus == 1
    assert delta_launch._resolve_gres(
        resources.gpus, None, surface=delta_launch.DELTAAI_SURFACE
    ) == ("gpu:nvidia_gh200_120gb:1")


def test_deltaai_accepts_interactive_partition_override() -> None:
    partitions = delta_launch.validate_partition_list(
        ["ghx4-interactive"], surface=delta_launch.DELTAAI_SURFACE
    )
    resources = resolve_stage_resources(
        stage="eval-monitor",
        backbone="qwen",
        requested_partitions=partitions,
        gpus_override=1,
        mem_override="64G",
        num_nodes=1,
        profile=DELTAAI_PROFILE,
    )

    assert resources.partition == "ghx4-interactive"
    assert resources.account == "deltaai_account"


def test_deltaai_rejects_classic_delta_partition() -> None:
    with pytest.raises(ValueError, match="Unsupported DeltaAI partition"):
        delta_launch.validate_partition_list(
            ["gpuA40x4"], surface=delta_launch.DELTAAI_SURFACE
        )


def test_delta_default_resources_use_a40_partition_and_gpus_per_node() -> None:
    resources = resolve_stage_resources(
        stage="policy",
        backbone="pali",
        requested_partitions=[],
        gpus_override=None,
        mem_override=None,
        num_nodes=1,
        profile=DELTA_PROFILE,
    )

    assert resources.partition == "gpuA40x4"
    assert resources.account == "delta_account"
    assert resources.gpus == 4
    assert (
        delta_launch._resolve_gres(
            resources.gpus, None, surface=delta_launch.DELTA_SURFACE
        )
        is None
    )


@pytest.mark.parametrize("partition", ["gpuA40x4", "gpuA100x4", "gpuH200x8"])
def test_delta_accepts_classic_delta_gpu_partitions(partition: str) -> None:
    partitions = delta_launch.validate_partition_list(
        [partition], surface=delta_launch.DELTA_SURFACE
    )
    resources = resolve_stage_resources(
        stage="eval",
        backbone="qwen",
        requested_partitions=partitions,
        gpus_override=1,
        mem_override="64G",
        num_nodes=1,
        profile=DELTA_PROFILE,
    )

    assert resources.partition == partition
    assert resources.account == "delta_account"


def test_delta_rejects_deltaai_partition() -> None:
    with pytest.raises(ValueError, match="Unsupported Delta partition"):
        delta_launch.validate_partition_list(
            ["ghx4"], surface=delta_launch.DELTA_SURFACE
        )


def test_delta_launch_respects_explicit_gres_override() -> None:
    assert (
        delta_launch._resolve_gres(2, "gpu:h200:2", surface=delta_launch.DELTA_SURFACE)
        == "gpu:h200:2"
    )


def test_deltaai_rejects_interactive_partition_with_default_batch_time() -> None:
    parser = delta_launch.build_arg_parser()
    args = parser.parse_args(
        [
            "train-tokenizer",
            "--codec",
            "oat",
            "--dataset",
            "libero_10",
            "--partition",
            "ghx4-interactive",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        delta_launch._validate_delta_args(
            parser,
            args,
            ["ghx4-interactive"],
            surface=delta_launch.DELTAAI_SURFACE,
        )

    assert exc.value.code == 2


def test_delta_rejects_interactive_partition_with_default_batch_time() -> None:
    parser = delta_launch.build_arg_parser()
    args = parser.parse_args(
        [
            "train-tokenizer",
            "--codec",
            "oat",
            "--dataset",
            "libero_10",
            "--partition",
            "gpuA40x4-interactive",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        delta_launch._validate_delta_args(
            parser,
            args,
            ["gpuA40x4-interactive"],
            surface=delta_launch.DELTA_SURFACE,
        )

    assert exc.value.code == 2


def test_delta_rejects_gpu_count_that_partition_cannot_satisfy() -> None:
    parser = delta_launch.build_arg_parser()
    args = parser.parse_args(
        [
            "train-policy",
            "--codec",
            "oat",
            "--dataset",
            "libero_10",
            "--tokenizer-ckpt",
            "outputs/train_codecs/example/checkpoints/latest/pretrained_model",
            "--partition",
            "gpuA40x4",
            "--gpus",
            "8",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        delta_launch._validate_delta_args(
            parser,
            args,
            ["gpuA40x4"],
            surface=delta_launch.DELTA_SURFACE,
        )

    assert exc.value.code == 2


def test_delta_allows_eight_gpus_on_eight_gpu_partition() -> None:
    parser = delta_launch.build_arg_parser()
    args = parser.parse_args(
        [
            "train-policy",
            "--codec",
            "oat",
            "--dataset",
            "libero_10",
            "--tokenizer-ckpt",
            "outputs/train_codecs/example/checkpoints/latest/pretrained_model",
            "--partition",
            "gpuH200x8",
            "--gpus",
            "8",
        ]
    )

    delta_launch._validate_delta_args(
        parser,
        args,
        ["gpuH200x8"],
        surface=delta_launch.DELTA_SURFACE,
    )


def test_delta_surface_override_selects_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRAXIS_DELTA_SURFACE", "deltaai")
    assert delta_launch.detect_delta_surface() is delta_launch.DELTAAI_SURFACE

    monkeypatch.setenv("PRAXIS_DELTA_SURFACE", "delta")
    assert delta_launch.detect_delta_surface() is delta_launch.DELTA_SURFACE


def test_delta_launch_help_does_not_require_cluster_detection() -> None:
    env = os.environ.copy()
    env.pop("PRAXIS_DELTA_SURFACE", None)
    result = subprocess.run(
        [sys.executable, "scripts/infra/delta_launch.py", "--help"],
        cwd=repo_root(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "train-policy" in result.stdout


def test_delta_launch_dry_run_prints_deltaai_gh200_gres() -> None:
    env = os.environ.copy()
    env["PRAXIS_DELTA_SURFACE"] = "deltaai"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/infra/delta_launch.py",
            "train-policy",
            "--codec",
            "oat",
            "--dataset",
            "libero_10",
            "--tokenizer-ckpt",
            "outputs/train_codecs/example/checkpoints/latest/pretrained_model",
            "--dry-run",
        ],
        cwd=repo_root(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "partition=ghx4  account=deltaai_account" in result.stdout
    assert "gres=gpu:nvidia_gh200_120gb:4" in result.stdout
    assert "policy.action_tokenizer.pretrained_path=" in result.stdout


def test_delta_launch_dry_run_prints_classic_delta_gpus_per_node() -> None:
    env = os.environ.copy()
    env["PRAXIS_DELTA_SURFACE"] = "delta"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/infra/delta_launch.py",
            "train-policy",
            "--codec",
            "oat",
            "--dataset",
            "libero_10",
            "--tokenizer-ckpt",
            "outputs/train_codecs/example/checkpoints/latest/pretrained_model",
            "--dry-run",
        ],
        cwd=repo_root(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "partition=gpuA40x4  account=delta_account" in result.stdout
    assert "gpus_per_node=4" in result.stdout
    assert "policy.action_tokenizer.pretrained_path=" in result.stdout
