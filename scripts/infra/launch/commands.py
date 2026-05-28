from __future__ import annotations

from pathlib import Path

from .config import (
    CODEC_OVERRIDES,
    CODEC_TRAINER_OVERRIDES,
    MASTER_ADDR_PLACEHOLDER,
    POLICY_TRAINER_OVERRIDES,
    SKIP_FIT_CODECS,
    SLURM_MACHINE_RANK_PLACEHOLDER,
    method_uses_codec,
    policy_config_name,
    policy_overrides,
    resolve_backbone,
    resolve_dataset_train_overrides,
    resolve_method,
)


def torchrun_distributed_args(
    *,
    gpus_per_node: int,
    num_nodes: int,
    main_process_ip: str | None,
    main_process_port: int,
) -> list[str]:
    """Build torchrun distributed launch args for single- or multi-node runs."""
    if gpus_per_node <= 0:
        raise ValueError(f"gpus_per_node must be positive, got {gpus_per_node}.")
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}.")

    args = [f"--nproc_per_node={gpus_per_node}"]

    if num_nodes == 1:
        return ["--standalone", *args]

    resolved_main_ip = main_process_ip or MASTER_ADDR_PLACEHOLDER
    args += [
        f"--nnodes={num_nodes}",
        f"--node_rank={SLURM_MACHINE_RANK_PLACEHOLDER}",
        f"--master_addr={resolved_main_ip}",
        f"--master_port={int(main_process_port)}",
    ]

    return args


def resolve_main_process_ip(raw_value: str) -> str | None:
    """Resolve launcher CLI rendezvous input to an explicit value or auto."""
    return None if raw_value == "auto" else raw_value


def infer_eval_env_from_dataset(dataset_name: str) -> tuple[str, str]:
    """Infer ``(env.type, env.task)`` for one-off eval from dataset naming."""
    from praxis_eval.envs.factory import infer_eval_env_target

    return infer_eval_env_target(dataset_name)


def build_eval_overrides_from_train_config(
    train_config_path: Path,
    *,
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
) -> list[str]:
    """Build one-off eval overrides from a checkpoint's train_config."""
    from praxis_eval.evaluation.overrides import (
        build_eval_overrides_from_train_config,
    )

    return build_eval_overrides_from_train_config(
        train_config_path,
        num_eval_per_task=num_eval_per_task,
        num_parallel_env=num_parallel_env,
        record_episodes_per_task=record_episodes_per_task,
        infer_env_from_dataset=infer_eval_env_from_dataset,
    )


def _resolve_executable(python_bin: str | None, name: str) -> list[str]:
    """Return the command prefix for a named executable.

    When *python_bin* is given, the executable is resolved from the same bin/
    directory (works for venvs and conda envs alike).  Otherwise falls back to
    ``uv run <name>``.  Torch distributed launch intentionally uses the module
    entrypoint because some cluster environments install torch without the
    ``torchrun`` console script.
    """
    if name == "torchrun":
        if python_bin is None:
            return ["uv", "run", "python", "-m", "torch.distributed.run"]
        return [python_bin, "-m", "torch.distributed.run"]
    if python_bin is None:
        return ["uv", "run", name]
    return [str(Path(python_bin).parent / name)]


def _normalize_eval_overrides(eval_overrides: list[str]) -> list[str]:
    """Normalize eval overrides without making every launcher import eval deps.

    Tokenizer/policy launch dry-runs should stay lightweight on login nodes. Import
    the eval stack only when a caller actually passes eval overrides.
    """
    if not eval_overrides:
        return []

    from praxis_eval.evaluation.overrides import normalize_eval_overrides

    return normalize_eval_overrides(eval_overrides)


def build_tokenizer_command(
    codec: str,
    dataset: str,
    gpus: int,
    num_nodes: int,
    main_process_ip: str | None,
    main_process_port: int,
    run_name: str,
    output_dir: str,
    skip_codec_fit: bool,
    extra_overrides: list[str],
    python_bin: str | None = None,
) -> list[str]:
    cmd = _resolve_executable(python_bin, "torchrun")
    cmd += torchrun_distributed_args(
        gpus_per_node=gpus,
        num_nodes=num_nodes,
        main_process_ip=main_process_ip,
        main_process_port=main_process_port,
    )
    cmd += ["-m", "praxis.scripts.train", "--config-name", "train_codecs"]
    cmd += CODEC_OVERRIDES[codec]
    if skip_codec_fit and codec not in SKIP_FIT_CODECS:
        raise ValueError(
            f"skip_codec_fit is only supported for codecs: {', '.join(sorted(SKIP_FIT_CODECS))}."
        )
    if skip_codec_fit:
        cmd.append("trainer.skip_codec_fit=true")
    if codec == "fast" and not skip_codec_fit:
        cmd.append("codec.bpe_tokenizer_path=null")
    cmd += [
        *resolve_dataset_train_overrides(dataset),
        "device=cuda",
        f"run_name={run_name}",
        f"output_dir={output_dir}",
    ]
    cmd += CODEC_TRAINER_OVERRIDES
    cmd += extra_overrides
    return cmd


def build_policy_command(
    method: str,
    codec: str | None,
    backbone: str,
    dataset: str,
    gpus: int,
    num_nodes: int,
    main_process_ip: str | None,
    main_process_port: int,
    run_name: str,
    output_dir: str,
    tokenizer_ckpt: str | None,
    extra_overrides: list[str],
    python_bin: str | None = None,
) -> list[str]:
    resolved_method = resolve_method(method)
    resolved_backbone = resolve_backbone(backbone)
    policy_name = policy_config_name(resolved_method, codec, resolved_backbone)

    cmd = _resolve_executable(python_bin, "torchrun")
    cmd += torchrun_distributed_args(
        gpus_per_node=gpus,
        num_nodes=num_nodes,
        main_process_ip=main_process_ip,
        main_process_port=main_process_port,
    )
    cmd += [
        "-m",
        "praxis.scripts.train",
        "--config-name",
        "train_policy",
        f"policy={policy_name}",
        f"train_recipe/policy@_global_={resolved_method}",
    ]
    cmd += policy_overrides(resolved_method, resolved_backbone)
    if method_uses_codec(resolved_method):
        if not tokenizer_ckpt:
            raise ValueError(
                f"Policy method {resolved_method!r} requires an action-token checkpoint."
            )
        cmd.append(f"policy.action_tokenizer.pretrained_path={tokenizer_ckpt}")
    elif tokenizer_ckpt:
        raise ValueError(
            f"Policy method {resolved_method!r} does not use action-token checkpoints."
        )
    cmd += [
        *resolve_dataset_train_overrides(dataset),
        "device=cuda",
        f"run_name={run_name}",
        f"output_dir={output_dir}",
    ]
    cmd += POLICY_TRAINER_OVERRIDES[resolved_backbone]
    cmd += extra_overrides
    return cmd


def build_eval_monitor_command(
    train_dir: str,
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
    stale_timeout_sec: int,
    eval_timeout_sec: int,
    eval_overrides: list[str],
    python_bin: str | None = None,
) -> list[str]:
    cmd = _resolve_executable(python_bin, "praxis-eval-monitor") + [
        "--train-dir",
        train_dir,
        "--num-eval-per-task",
        str(num_eval_per_task),
        "--num-parallel-env",
        str(num_parallel_env),
        "--record-episodes-per-task",
        str(record_episodes_per_task),
        "--stale-timeout-sec",
        str(stale_timeout_sec),
        "--eval-timeout-sec",
        str(eval_timeout_sec),
    ]
    normalized_eval_overrides = _normalize_eval_overrides(eval_overrides)
    if normalized_eval_overrides:
        cmd += ["--eval-override", *normalized_eval_overrides]
    return cmd


def build_eval_command(
    policy_ckpt: str,
    eval_output_dir: str,
    *,
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
    eval_overrides: list[str],
    python_bin: str | None = None,
) -> list[str]:
    train_config_path = Path(policy_ckpt) / "train_config.yaml"
    if not train_config_path.exists():
        raise FileNotFoundError(
            f"Could not find train_config.yaml under policy checkpoint: {policy_ckpt}"
        )

    cmd = _resolve_executable(python_bin, "praxis-eval")
    cmd += build_eval_overrides_from_train_config(
        train_config_path,
        num_eval_per_task=num_eval_per_task,
        num_parallel_env=num_parallel_env,
        record_episodes_per_task=record_episodes_per_task,
    )
    cmd += [
        f"policy.pretrained_path={policy_ckpt}",
        f"eval.output_dir={eval_output_dir}",
    ]
    cmd += _normalize_eval_overrides(eval_overrides)
    return cmd
