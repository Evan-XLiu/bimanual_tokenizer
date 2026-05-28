from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .args import (
    build_arg_parser,
    flatten_eval_overrides,
    parse_partition_inputs,
    validate_args,
)
from .commands import resolve_main_process_ip
from .config import LaunchContext, dataset_short
from .profile import SlurmClusterProfile
from .slurm import (
    make_executor,
    print_slurm_params,
    query_slurm_job_state,
    run_training_job,
)
from .stages import (
    EvalMonitorPlan,
    EvalPlan,
    TrainPolicyPlan,
    TrainTokenizerPlan,
    launch_eval_monitor_stage,
    launch_eval_stage,
    launch_policy_stage,
    launch_tokenizer_stage,
)

ClusterArgValidator = Callable[[ArgumentParser, Namespace, list[str]], None]
GresResolver = Callable[[int, str | None], str | None]


def default_gres_resolver(_gpus: int, requested_gres: str | None) -> str | None:
    return requested_gres


def run_launcher(
    *,
    profile: SlurmClusterProfile,
    argv: Sequence[str] | None = None,
    cluster_arg_validator: ClusterArgValidator | None = None,
    resolve_gres: GresResolver = default_gres_resolver,
    project_root: Path | None = None,
    run_training_job_fn: Callable[..., None] = run_training_job,
) -> None:
    """Run the shared Slurm launcher flow for one cluster profile."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        requested_partitions = profile.validate_partition_list(
            parse_partition_inputs(args.partition)
        )
    except ValueError as exc:
        parser.error(str(exc))
    validate_args(parser, args)
    if cluster_arg_validator is not None:
        cluster_arg_validator(parser, args, requested_partitions)

    root = project_root or Path(__file__).resolve().parents[3]
    context = LaunchContext(
        dry_run=args.dry_run,
        project_root=str(root),
        ds_short=dataset_short(args.dataset) if args.dataset else "",
        date_str=datetime.now().strftime("%Y%m%d"),
        gpus_override=args.gpus,
        mem_override=args.mem,
        num_nodes=args.num_nodes,
        main_process_ip=resolve_main_process_ip(args.main_process_ip),
        main_process_port=args.main_process_port,
        override=args.override,
        num_eval_per_task=args.num_eval_per_task,
        num_parallel_env=args.num_parallel_env,
        record_episodes_per_task=args.record_episodes_per_task,
        stale_timeout_sec=args.stale_timeout_sec,
        eval_timeout_sec=args.eval_timeout_sec,
        eval_override=flatten_eval_overrides(args.eval_override),
        python_bin=args.python,
        git_ref=args.git_ref,
        git_worktree_dir=args.git_worktree_dir,
    )

    def make_stage_executor(
        *,
        run_name: str,
        gpus: int,
        mem: str,
        partition: str,
        account: str,
        num_nodes: int,
        extra_slurm: dict[str, object] | None = None,
    ) -> Any:
        effective_gres = resolve_gres(gpus, args.gres)
        return make_executor(
            project_root=context.project_root,
            run_name=run_name,
            gpus=gpus,
            mem=mem,
            partition=partition,
            account=account,
            num_nodes=num_nodes,
            cpus_per_task=args.cpus_per_task,
            time_minutes=args.time,
            constraint=args.constraint,
            gres=effective_gres,
            requeue_policy=args.requeue_policy,
            mail_type=profile.mail_type,
            mail_user=profile.mail_user,
            extra_slurm=extra_slurm,
        )

    def print_stage_slurm_params(
        *,
        gpus: int,
        mem: str,
        partition: str,
        account: str,
        num_nodes: int,
        extra: str = "",
    ) -> None:
        effective_gres = resolve_gres(gpus, args.gres)
        print_slurm_params(
            gpus=gpus,
            mem=mem,
            partition=partition,
            account=account,
            num_nodes=num_nodes,
            cpus_per_task=args.cpus_per_task,
            time_minutes=args.time,
            constraint=args.constraint,
            gres=effective_gres,
            extra=extra,
        )

    stage = str(args.stage)

    if stage == "eval-monitor":
        launch_eval_monitor_stage(
            plan=EvalMonitorPlan(
                train_dir=args.train_dir,
                train_job_id=args.train_job_id,
                backbone=args.backbone,
            ),
            requested_partitions=requested_partitions,
            context=context,
            make_stage_executor=make_stage_executor,
            print_stage_slurm_params=print_stage_slurm_params,
            query_job_state=query_slurm_job_state,
            profile=profile,
        )
        return

    if stage == "eval":
        launch_eval_stage(
            plan=EvalPlan(
                policy_ckpt=args.policy_ckpt,
                eval_output_dir=args.eval_output_dir,
                backbone=args.backbone,
            ),
            requested_partitions=requested_partitions,
            context=context,
            make_stage_executor=make_stage_executor,
            print_stage_slurm_params=print_stage_slurm_params,
            profile=profile,
        )
        return

    if stage == "tokenizer":
        launch_tokenizer_stage(
            plan=TrainTokenizerPlan(
                codec=args.codec,
                dataset=args.dataset,
                backbone=args.backbone,
                skip_fit=args.skip_fit,
            ),
            requested_partitions=requested_partitions,
            context=context,
            make_stage_executor=make_stage_executor,
            print_stage_slurm_params=print_stage_slurm_params,
            profile=profile,
            run_training_job_fn=run_training_job_fn,
        )
        print()
        return

    if stage == "policy":
        launch_policy_stage(
            plan=TrainPolicyPlan(
                method=args.method,
                codec=args.codec,
                backbone=args.backbone,
                dataset=args.dataset,
                tokenizer_ckpt=args.tokenizer_ckpt,
            ),
            requested_partitions=requested_partitions,
            context=context,
            make_stage_executor=make_stage_executor,
            print_stage_slurm_params=print_stage_slurm_params,
            profile=profile,
            run_training_job_fn=run_training_job_fn,
        )
        print()
        return

    parser.error(f"Unsupported launch verb: {args.verb!r}.")
