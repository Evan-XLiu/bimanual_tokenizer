# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from .commands import (
    build_eval_command,
    build_eval_monitor_command,
    build_policy_command,
    build_tokenizer_command,
)
from .config import (
    RESOURCE_DEFAULTS,
    LaunchContext,
    StageResources,
    policy_config_name,
)
from .profile import SlurmClusterProfile
from .slurm import (
    build_eval_monitor_dependency,
    run_eval_job,
    run_eval_monitor_job,
    submit_or_print_stage,
)


@dataclass(frozen=True)
class EvalPlan:
    policy_ckpt: str
    eval_output_dir: str
    backbone: str


@dataclass(frozen=True)
class EvalMonitorPlan:
    train_dir: str
    train_job_id: int | None
    backbone: str


@dataclass(frozen=True)
class TrainTokenizerPlan:
    codec: str
    dataset: str
    backbone: str
    skip_fit: bool


@dataclass(frozen=True)
class TrainPolicyPlan:
    method: str
    codec: str | None
    backbone: str
    dataset: str
    tokenizer_ckpt: str | None


def resource_key(stage: str, backbone: str) -> str:
    if stage == "eval-monitor":
        return "eval_monitor"
    if stage == "eval":
        return "eval"
    if stage == "tokenizer":
        return "tokenizer"
    return f"policy_{backbone}"


def resolve_stage_resources(
    *,
    stage: str,
    backbone: str,
    requested_partitions: list[str],
    gpus_override: int | None,
    mem_override: str | None,
    num_nodes: int,
    profile: SlurmClusterProfile,
) -> StageResources:
    """Resolve stage-specific Slurm resources and account selection."""
    key = resource_key(stage, backbone)
    default_gpus, default_mem = RESOURCE_DEFAULTS[key]
    partition = profile.resolve_partition_spec(stage, backbone, requested_partitions)
    account = profile.resolve_account_for_partition_spec(partition)
    gpus = default_gpus if gpus_override is None else int(gpus_override)
    if gpus <= 0:
        raise ValueError(f"gpus must be positive, got {gpus}.")
    return StageResources(
        gpus=gpus,
        mem=mem_override or default_mem,
        partition=partition,
        account=account,
        num_nodes=num_nodes,
    )


def _last_exact_override_value(overrides: list[str], key: str) -> str | None:
    prefix = f"{key}="
    for override in reversed(overrides):
        if override.startswith(prefix):
            return override.split("=", 1)[1]
    return None


def strip_resolved_run_overrides(overrides: list[str]) -> list[str]:
    """Drop run label overrides after they have been resolved into command args."""
    stripped_keys = {"run_name", "output_dir"}
    stripped_prefixes = tuple(f"{key}=" for key in stripped_keys)
    return [
        override for override in overrides if not override.startswith(stripped_prefixes)
    ]


def resolve_stage_run_name_and_output_dir(
    *,
    default_run_name: str,
    default_output_dir: str,
    overrides: list[str],
) -> tuple[str, str]:
    """Honor explicit top-level run/output overrides in stage labels and job names."""
    run_name = _last_exact_override_value(overrides, "run_name") or default_run_name
    output_dir = (
        _last_exact_override_value(overrides, "output_dir") or default_output_dir
    )
    return run_name, output_dir


def launch_eval_stage(
    *,
    plan: EvalPlan,
    requested_partitions: list[str],
    context: LaunchContext,
    make_stage_executor: Callable[..., Any],
    print_stage_slurm_params: Callable[..., None],
    profile: SlurmClusterProfile,
) -> None:
    policy_ckpt = str(Path(plan.policy_ckpt).resolve())
    eval_output_dir = str(Path(plan.eval_output_dir).resolve())
    run_name = f"eval_{Path(eval_output_dir).name}"
    resources = resolve_stage_resources(
        stage="eval",
        backbone=plan.backbone,
        requested_partitions=requested_partitions,
        gpus_override=context.gpus_override,
        mem_override=context.mem_override,
        num_nodes=1,
        profile=profile,
    )
    command = build_eval_command(
        policy_ckpt=policy_ckpt,
        eval_output_dir=eval_output_dir,
        num_eval_per_task=context.num_eval_per_task,
        num_parallel_env=context.num_parallel_env,
        record_episodes_per_task=context.record_episodes_per_task,
        eval_overrides=context.eval_override,
        python_bin=context.python_bin,
    )

    submit_or_print_stage(
        label="eval",
        run_name=run_name,
        command=command,
        gpus=resources.gpus,
        mem=resources.mem,
        partition=resources.partition,
        account=resources.account,
        num_nodes=resources.num_nodes,
        dry_run=context.dry_run,
        print_slurm_params=print_stage_slurm_params,
        make_executor=make_stage_executor,
        submit_job=lambda executor: executor.submit(
            run_eval_job,
            command,
            context.project_root,
            git_ref=context.git_ref,
            git_worktree_dir=context.git_worktree_dir,
        ),
    )


def launch_eval_monitor_stage(
    *,
    plan: EvalMonitorPlan,
    requested_partitions: list[str],
    context: LaunchContext,
    make_stage_executor: Callable[..., Any],
    print_stage_slurm_params: Callable[..., None],
    query_job_state: Callable[[int], str | None],
    profile: SlurmClusterProfile,
) -> None:
    train_dir = str(Path(plan.train_dir).resolve())
    run_name = f"eval_monitor_{Path(train_dir).name}"
    resources = resolve_stage_resources(
        stage="eval-monitor",
        backbone=plan.backbone,
        requested_partitions=requested_partitions,
        gpus_override=context.gpus_override,
        mem_override=context.mem_override,
        num_nodes=1,
        profile=profile,
    )
    command = build_eval_monitor_command(
        train_dir=train_dir,
        num_eval_per_task=context.num_eval_per_task,
        num_parallel_env=context.num_parallel_env,
        record_episodes_per_task=context.record_episodes_per_task,
        stale_timeout_sec=context.stale_timeout_sec,
        eval_timeout_sec=context.eval_timeout_sec,
        eval_overrides=context.eval_override,
        python_bin=context.python_bin,
    )
    extra_slurm, dependency_note, submitted_suffix = build_eval_monitor_dependency(
        plan.train_job_id,
        query_job_state=query_job_state,
    )

    submit_or_print_stage(
        label="eval-monitor",
        run_name=run_name,
        command=command,
        gpus=resources.gpus,
        mem=resources.mem,
        partition=resources.partition,
        account=resources.account,
        num_nodes=resources.num_nodes,
        dry_run=context.dry_run,
        print_slurm_params=print_stage_slurm_params,
        make_executor=partial(make_stage_executor, extra_slurm=extra_slurm),
        submit_job=lambda executor: executor.submit(
            run_eval_monitor_job,
            command,
            context.project_root,
            git_ref=context.git_ref,
            git_worktree_dir=context.git_worktree_dir,
        ),
        dry_run_extra=dependency_note,
        submitted_suffix=submitted_suffix,
    )


def launch_tokenizer_stage(
    *,
    plan: TrainTokenizerPlan,
    requested_partitions: list[str],
    context: LaunchContext,
    make_stage_executor: Callable[..., Any],
    print_stage_slurm_params: Callable[..., None],
    profile: SlurmClusterProfile,
    run_training_job_fn: Callable[..., None],
) -> None:
    default_run_name = f"{context.ds_short}_{plan.codec}_codec"
    default_output_dir = f"outputs/train_codecs/{context.date_str}/{default_run_name}"
    run_name, output_dir = resolve_stage_run_name_and_output_dir(
        default_run_name=default_run_name,
        default_output_dir=default_output_dir,
        overrides=context.override,
    )
    resources = resolve_stage_resources(
        stage="tokenizer",
        backbone=plan.backbone,
        requested_partitions=requested_partitions,
        gpus_override=context.gpus_override,
        mem_override=context.mem_override,
        num_nodes=context.num_nodes,
        profile=profile,
    )
    command = build_tokenizer_command(
        codec=plan.codec,
        dataset=plan.dataset,
        gpus=resources.gpus,
        num_nodes=context.num_nodes,
        main_process_ip=context.main_process_ip,
        main_process_port=context.main_process_port,
        run_name=run_name,
        output_dir=output_dir,
        skip_codec_fit=plan.skip_fit,
        extra_overrides=strip_resolved_run_overrides(context.override),
        python_bin=context.python_bin,
    )

    submit_or_print_stage(
        label="tokenizer",
        run_name=run_name,
        command=command,
        gpus=resources.gpus,
        mem=resources.mem,
        partition=resources.partition,
        account=resources.account,
        num_nodes=resources.num_nodes,
        dry_run=context.dry_run,
        print_slurm_params=print_stage_slurm_params,
        make_executor=make_stage_executor,
        submit_job=lambda executor: executor.submit(
            run_training_job_fn,
            command,
            context.project_root,
            num_nodes=context.num_nodes,
            auto_resume_on_requeue=profile.is_requeue_partition_spec(
                resources.partition
            ),
            git_ref=context.git_ref,
            git_worktree_dir=context.git_worktree_dir,
        ),
    )


def launch_policy_stage(
    *,
    plan: TrainPolicyPlan,
    requested_partitions: list[str],
    context: LaunchContext,
    make_stage_executor: Callable[..., Any],
    print_stage_slurm_params: Callable[..., None],
    profile: SlurmClusterProfile,
    run_training_job_fn: Callable[..., None],
) -> None:
    policy_name = policy_config_name(plan.method, plan.codec, plan.backbone)
    default_run_name = f"{context.ds_short}_{policy_name}"
    default_output_dir = f"outputs/train_policy/{context.date_str}/{default_run_name}"
    run_name, output_dir = resolve_stage_run_name_and_output_dir(
        default_run_name=default_run_name,
        default_output_dir=default_output_dir,
        overrides=context.override,
    )
    resources = resolve_stage_resources(
        stage="policy",
        backbone=plan.backbone,
        requested_partitions=requested_partitions,
        gpus_override=context.gpus_override,
        mem_override=context.mem_override,
        num_nodes=context.num_nodes,
        profile=profile,
    )

    command = build_policy_command(
        method=plan.method,
        codec=plan.codec,
        backbone=plan.backbone,
        dataset=plan.dataset,
        gpus=resources.gpus,
        num_nodes=context.num_nodes,
        main_process_ip=context.main_process_ip,
        main_process_port=context.main_process_port,
        run_name=run_name,
        output_dir=output_dir,
        tokenizer_ckpt=plan.tokenizer_ckpt,
        extra_overrides=strip_resolved_run_overrides(context.override),
        python_bin=context.python_bin,
    )

    submit_or_print_stage(
        label="policy",
        run_name=run_name,
        command=command,
        gpus=resources.gpus,
        mem=resources.mem,
        partition=resources.partition,
        account=resources.account,
        num_nodes=resources.num_nodes,
        dry_run=context.dry_run,
        print_slurm_params=print_stage_slurm_params,
        make_executor=make_stage_executor,
        submit_job=lambda executor: executor.submit(
            run_training_job_fn,
            command,
            context.project_root,
            num_nodes=context.num_nodes,
            auto_resume_on_requeue=profile.is_requeue_partition_spec(
                resources.partition
            ),
            git_ref=context.git_ref,
            git_worktree_dir=context.git_worktree_dir,
        ),
    )
