from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import (
    BACKBONES,
    CODECS,
    DATASETS,
    DEFAULT_BACKBONE,
    DEFAULT_CODEC,
    DEFAULT_METHOD,
    METHODS,
    POLICY_CONFIG_ROOT,
    SKIP_FIT_CODECS,
    method_uses_codec,
    policy_config_name,
    resolve_backbone,
    resolve_codec,
    resolve_method,
    resolve_policy_codec,
)


def flatten_eval_overrides(
    raw_values: Sequence[str | Sequence[str]] | None,
) -> list[str]:
    """Flatten argparse's repeated ``--eval-override`` groups in CLI order."""
    if not raw_values:
        return []
    flattened: list[str] = []
    for value in raw_values:
        if isinstance(value, str):
            flattened.append(value)
        else:
            flattened.extend(value)
    return flattened


def parse_partition_inputs(raw_values: list[str] | None) -> list[str]:
    """Parse repeated/comma-separated partition CLI inputs into a de-duplicated list."""
    if not raw_values:
        return []
    parsed: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        tokens = str(raw).split(",")
        if any(not token.strip() for token in tokens):
            raise ValueError(
                "Partition values cannot be empty. Omit --partition to use "
                "defaults, or pass non-empty partition names."
            )
        for token in tokens:
            p = token.strip()
            if p in seen:
                continue
            parsed.append(p)
            seen.add(p)
    return parsed


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate verb-specific launcher arguments."""
    if args.gpus is not None and args.gpus <= 0:
        parser.error("--gpus must be a positive integer.")
    if args.gres in ("", "none", "null"):
        parser.error(
            "--gres cannot be empty; omit it or pass a valid Slurm --gres expression."
        )
    if args.gres is not None:
        if args.gpus is None:
            parser.error(
                "--gpus is required when --gres is set so torchrun "
                "--nproc_per_node matches the Slurm GPU request."
            )
        gres_count = _parse_gres_gpu_count(args.gres)
        if gres_count is not None and gres_count != args.gpus:
            parser.error(
                "--gres GPU count must match --gpus so torchrun does not spawn "
                f"the wrong number of ranks: --gres requests {gres_count}, "
                f"--gpus={args.gpus}."
            )
    if args.stage == "eval-monitor":
        if args.train_job_id is not None and args.train_job_id <= 0:
            parser.error("--train-job-id must be a positive integer.")
        return

    if args.stage == "eval":
        return

    try:
        if args.stage == "tokenizer":
            args.codec = resolve_codec(args.codec)
        else:
            args.method = resolve_method(args.method)
            args.codec = resolve_policy_codec(args.method, args.codec)
            args.backbone = resolve_backbone(args.backbone)
    except ValueError as exc:
        parser.error(str(exc))
    if args.num_nodes <= 0:
        parser.error("--num-nodes must be >= 1.")
    if args.main_process_ip in ("", "none", "null"):
        parser.error(
            "--main-process-ip cannot be empty; use 'auto' or a concrete IP/hostname."
        )
    if args.constraint in ("", "none", "null"):
        parser.error(
            "--constraint cannot be empty; omit it or pass a valid Slurm constraint string."
        )
    if args.skip_fit and args.codec not in SKIP_FIT_CODECS:
        parser.error("--skip-fit is currently only supported for --codec=bin or fast.")
    if args.stage == "policy":
        policy_name = policy_config_name(args.method, args.codec, args.backbone)
        policy_cfg_path = POLICY_CONFIG_ROOT / f"{policy_name}.yaml"
        if not policy_cfg_path.exists():
            parser.error(
                "Unsupported method/codec/backbone combination for launcher: "
                f"{policy_name}. Expected config at {policy_cfg_path}."
            )
    if (
        args.stage == "policy"
        and method_uses_codec(args.method)
        and not args.tokenizer_ckpt
    ):
        parser.error(
            "--tokenizer-ckpt is required for train-policy with a codec-backed method. "
            "Train/bootstrap the action tokenizer first."
        )
    if (
        args.stage == "policy"
        and not method_uses_codec(args.method)
        and args.tokenizer_ckpt
    ):
        parser.error(
            f"--tokenizer-ckpt is not valid for method {args.method!r}; "
            "this policy family has no action-token codec."
        )


def _parse_gres_gpu_count(gres: str) -> int | None:
    """Best-effort parse of the GPU count from a Slurm GRES expression."""
    count = 0
    saw_gpu = False
    for raw_part in str(gres).split(","):
        part = raw_part.strip()
        if not part:
            continue
        fields = part.split(":")
        if fields[0] != "gpu":
            continue
        saw_gpu = True
        if len(fields) == 1:
            count += 1
            continue
        try:
            count += int(fields[-1])
        except ValueError:
            # Slurm's name[[:type]:count] syntax defaults omitted counts to 1.
            count += 1
    return count if saw_gpu else None


def _add_slurm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--partition",
        action="append",
        default=None,
        help=(
            "Slurm partition override. Supports repeated flags and/or comma-separated values "
            "(e.g. --partition partition_a --partition partition_b or "
            "--partition partition_a,partition_b)."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override GPU count (default: auto from resource table)",
    )
    parser.add_argument(
        "--mem", default=None, help="Override memory, e.g. 256G (default: auto)"
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="Slurm CPUs per task (default: 16).",
    )
    parser.add_argument(
        "--constraint",
        default=None,
        help=(
            "Optional Slurm feature constraint passed as --constraint (e.g. h100, cc9.0, or 'h100&holyndr')."
        ),
    )
    parser.add_argument(
        "--gres",
        default=None,
        help=(
            "Optional Slurm --gres expression (e.g. 'gpu:nvidia_a100-sxm4-80gb:1'). "
            "When set, replaces --gpus-per-node so Slurm constrains by GPU type name; "
            "--gpus then only affects resource defaults / printing, not the sbatch line."
        ),
    )
    parser.add_argument(
        "--time",
        type=int,
        default=2880,
        help="Time limit in minutes (default: 2880 = 48h)",
    )
    parser.add_argument(
        "--requeue-policy",
        default="auto",
        choices=("auto", "allow", "disable"),
        help=(
            "Requeue behavior for submitted jobs. "
            "'auto' keeps legacy defaults (no-requeue except on *requeue partitions*), "
            "'allow' omits --no-requeue, "
            "'disable' always passes --no-requeue."
        ),
    )


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python",
        default=None,
        metavar="PATH",
        help=(
            "Path to the Python binary to use on the compute node "
            "(e.g. /path/to/.venv-robocasa/bin/python or a conda env's python). "
            "When set, executables are resolved from the same bin/ directory instead "
            "of using 'uv run'. Omit to use the default 'uv run' behaviour."
        ),
    )
    parser.add_argument(
        "--git-ref",
        default=None,
        help=(
            "Optional git ref (commit/tag/branch) to run from via an ephemeral "
            "git worktree. When set, job execution checks out this ref in a "
            "temporary worktree and cleans it up after completion."
        ),
    )
    parser.add_argument(
        "--git-worktree-dir",
        default="outputs/worktree",
        help=(
            "Base directory for ephemeral git worktrees (default: outputs/worktree). "
            "Each job gets a unique subdirectory to avoid collisions."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and Slurm params without submitting",
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        choices=DATASETS,
        help="Dataset name.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes for training jobs (default: 1).",
    )
    parser.add_argument(
        "--main-process-ip",
        default="auto",
        help=(
            "Rank-0 rendezvous IP/hostname for torchrun. "
            "Use 'auto' to derive from first host in SLURM_NODELIST (default)."
        ),
    )
    parser.add_argument(
        "--main-process-port",
        type=int,
        default=29500,
        help="Rank-0 rendezvous port for torchrun (default: 29500).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Extra Hydra overrides passed through verbatim",
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-eval-per-task",
        type=int,
        default=10,
        help="Num eval episodes per task (default: 10)",
    )
    parser.add_argument(
        "--num-parallel-env",
        type=int,
        default=10,
        help="Num parallel envs for eval (default: 10)",
    )
    parser.add_argument(
        "--record-episodes-per-task",
        type=int,
        default=0,
        help="Num recorded episodes per task (default: 0)",
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Extra Hydra overrides passed to praxis-eval. Repeat the flag or pass "
            "multiple values after one flag (e.g. eval.policy_kwargs.decode_keep_k=4)."
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit Praxis training jobs to Slurm via submitit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(
        method=DEFAULT_METHOD,
        codec=None,
        backbone=DEFAULT_BACKBONE,
        dataset=None,
        tokenizer_ckpt=None,
        train_dir=None,
        train_job_id=None,
        policy_ckpt=None,
        eval_output_dir=None,
        num_nodes=1,
        main_process_ip="auto",
        main_process_port=29500,
        override=[],
        skip_fit=False,
        num_eval_per_task=10,
        num_parallel_env=10,
        record_episodes_per_task=0,
        stale_timeout_sec=10800,
        eval_timeout_sec=0,
        eval_override=[],
    )

    subparsers = parser.add_subparsers(dest="verb", required=True)

    train_tokenizer = subparsers.add_parser(
        "train-tokenizer",
        help="Submit an action tokenizer training job.",
    )
    train_tokenizer.set_defaults(stage="tokenizer")
    _add_training_args(train_tokenizer)
    _add_slurm_args(train_tokenizer)
    _add_runtime_args(train_tokenizer)
    train_tokenizer.add_argument(
        "--codec",
        default=None,
        choices=CODECS,
        help=f"Action tokenizer family (default: {DEFAULT_CODEC})",
    )
    train_tokenizer.add_argument(
        "--skip-fit",
        action="store_true",
        help=(
            "Skip codec fit() and bootstrap a token codec checkpoint using the "
            "codec config plus dataset action stats. Supported for deterministic "
            "BIN and default-BPE FAST."
        ),
    )

    train_policy = subparsers.add_parser(
        "train-policy",
        help="Submit a policy training job.",
    )
    train_policy.set_defaults(stage="policy")
    _add_training_args(train_policy)
    _add_slurm_args(train_policy)
    _add_runtime_args(train_policy)
    train_policy.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=METHODS,
        help=f"Policy generation method (default: {DEFAULT_METHOD})",
    )
    train_policy.add_argument(
        "--codec",
        default=None,
        choices=CODECS,
        help=f"Action tokenizer for codec-backed policies (default: {DEFAULT_CODEC})",
    )
    train_policy.add_argument(
        "--backbone",
        default=DEFAULT_BACKBONE,
        choices=BACKBONES,
        help=f"Policy backbone (default: {DEFAULT_BACKBONE})",
    )
    train_policy.add_argument(
        "--tokenizer-ckpt",
        default=None,
        help="Pretrained action tokenizer checkpoint for codec-backed policies.",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Submit one checkpoint eval job.",
    )
    eval_parser.set_defaults(stage="eval")
    _add_slurm_args(eval_parser)
    _add_runtime_args(eval_parser)
    _add_eval_args(eval_parser)
    eval_parser.add_argument(
        "--policy-ckpt",
        required=True,
        help="Policy checkpoint path for eval (expects pretrained_model dir).",
    )
    eval_parser.add_argument(
        "--eval-output-dir",
        required=True,
        help="Output dir for eval artifacts.",
    )

    eval_monitor = subparsers.add_parser(
        "eval-monitor",
        help="Submit an eval monitor job for a training run.",
    )
    eval_monitor.set_defaults(stage="eval-monitor")
    _add_slurm_args(eval_monitor)
    _add_runtime_args(eval_monitor)
    _add_eval_args(eval_monitor)
    eval_monitor.add_argument(
        "--train-dir",
        required=True,
        help="Training output dir to monitor.",
    )
    eval_monitor.add_argument(
        "--train-job-id",
        type=int,
        default=None,
        help=(
            "Optional training Slurm job id. If the job is running, eval-monitor "
            "launches immediately; otherwise it is submitted with dependency=after:<jobid>."
        ),
    )
    eval_monitor.add_argument(
        "--stale-timeout-sec",
        type=int,
        default=10800,
        help="Monitor idle timeout in seconds before assuming trainer stalled (default: 10800)",
    )
    eval_monitor.add_argument(
        "--eval-timeout-sec",
        type=int,
        default=0,
        help="Kill one eval subprocess after this many seconds (0 disables timeout)",
    )

    return parser
