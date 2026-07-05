# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import ArgumentParser, Namespace

from scripts.infra.launch.profile import SlurmClusterProfile
from scripts.infra.launch.runner import run_launcher
from tests._paths import repo_root


def _profile() -> SlurmClusterProfile:
    return SlurmClusterProfile(
        validate_partition_list=lambda partitions: partitions,
        resolve_partition_spec=lambda _stage, _backbone, partitions: (
            ",".join(partitions) if partitions else "debug"
        ),
        resolve_account_for_partition_spec=lambda _partition: "test_account",
        is_requeue_partition_spec=lambda _partition: False,
    )


def test_shared_launcher_runner_applies_cluster_gres_resolver_and_validator(
    capsys,
) -> None:
    validator_calls: list[tuple[str, list[str]]] = []

    def _validate(
        _parser: ArgumentParser,
        args: Namespace,
        requested_partitions: list[str],
    ) -> None:
        validator_calls.append((args.stage, list(requested_partitions)))

    run_launcher(
        profile=_profile(),
        argv=[
            "train-tokenizer",
            "--codec",
            "oat",
            "--dataset",
            "libero_10",
            "--partition",
            "debug",
            "--gpus",
            "3",
            "--dry-run",
        ],
        cluster_arg_validator=_validate,
        resolve_gres=lambda gpus, requested: requested or f"gpu:test:{gpus}",
        project_root=repo_root(),
    )

    stdout = capsys.readouterr().out
    assert validator_calls == [("tokenizer", ["debug"])]
    assert "partition=debug  account=test_account" in stdout
    assert "gres=gpu:test:3" in stdout
    assert "python -m torch.distributed.run --standalone --nproc_per_node=3" in stdout
