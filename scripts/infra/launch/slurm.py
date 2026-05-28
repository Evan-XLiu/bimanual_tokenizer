from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import (
    MASTER_ADDR_PLACEHOLDER,
    SLURM_MACHINE_RANK_PLACEHOLDER,
    SUBMITIT_LOG_BASE,
)


def query_slurm_job_state(job_id: int) -> str | None:
    """Best-effort query of a Slurm job state via squeue/sacct."""
    commands = [
        ["squeue", "--noheader", "--jobs", str(job_id), "--format=%T"],
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "-X",
            "-j",
            str(job_id),
            "--format=State",
        ],
    ]
    for cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if proc.returncode != 0:
            continue
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            state = line.split("|", 1)[0].split()[0].upper()
            if state:
                return state
    return None


def build_eval_monitor_dependency(
    train_job_id: int | None,
    *,
    query_job_state: Callable[[int], str | None] = query_slurm_job_state,
) -> tuple[dict[str, object], str, str]:
    """Return `(extra_slurm, dry_run_extra, submitted_suffix)` for eval-monitor."""
    if train_job_id is None:
        return {}, "", ""

    state = query_job_state(int(train_job_id))
    if state in {"RUNNING", "COMPLETING"}:
        return (
            {},
            f"train_job_id={train_job_id}  train_job_state={state}  no dependency",
            "",
        )

    state_label = state or "UNKNOWN"
    return (
        {"dependency": f"after:{train_job_id}"},
        f"dependency=after:{train_job_id}  train_job_state={state_label}",
        f"  (depends on {train_job_id}; train_state={state_label})",
    )


def submit_or_print_stage(
    *,
    label: str,
    run_name: str,
    command: list[str],
    gpus: int,
    mem: str,
    partition: str,
    account: str,
    num_nodes: int,
    dry_run: bool,
    print_slurm_params: Callable[..., None],
    make_executor: Callable[..., Any],
    submit_job: Callable[[Any], Any],
    dry_run_extra: str = "",
    submitted_suffix: str = "",
) -> Any | None:
    print(f"\n[{label}] {run_name}")
    if dry_run:
        print_slurm_params(
            gpus=gpus,
            mem=mem,
            partition=partition,
            account=account,
            num_nodes=num_nodes,
            extra=dry_run_extra,
        )
        print(f"  command:\n    {shlex.join(command)}\n")
        print()
        return None

    executor = make_executor(
        run_name=run_name,
        gpus=gpus,
        mem=mem,
        partition=partition,
        account=account,
        num_nodes=num_nodes,
    )
    job = submit_job(executor)
    print(f"  Submitted job {job.job_id}{submitted_suffix}")
    print_slurm_params(
        gpus=gpus,
        mem=mem,
        partition=partition,
        account=account,
        num_nodes=num_nodes,
    )
    print()
    return job


def should_set_no_requeue(requeue_policy: str, partition: str) -> bool:
    if requeue_policy == "disable":
        return True
    if requeue_policy == "allow":
        return False
    parts = [p.strip() for p in partition.split(",") if p.strip()]
    return not any(p.endswith("requeue") for p in parts)


def make_executor(
    *,
    project_root: str,
    run_name: str,
    gpus: int,
    mem: str,
    partition: str,
    account: str,
    num_nodes: int,
    cpus_per_task: int,
    time_minutes: int,
    constraint: str | None,
    gres: str | None,
    requeue_policy: str,
    mail_type: str | None = None,
    mail_user: str | None = None,
    extra_slurm: dict[str, object] | None = None,
) -> Any:
    import submitit

    log_dir = SUBMITIT_LOG_BASE / run_name
    executor = submitit.AutoExecutor(folder=str(log_dir))
    additional: dict[str, object] = {
        "gpu-bind": "closest",
    }
    if mail_type:
        additional["mail_type"] = mail_type
    if mail_user:
        additional["mail_user"] = mail_user
    if should_set_no_requeue(requeue_policy, partition):
        additional["no_requeue"] = True
    if constraint:
        additional["constraint"] = constraint
    if gres:
        additional["gres"] = gres
    if extra_slurm:
        additional.update(extra_slurm)

    infra_path = Path(project_root) / "scripts" / "infra"
    pythonpath_setup = (
        f"export PYTHONPATH={shlex.quote(str(infra_path))}:${{PYTHONPATH:-}}"
    )

    params: dict[str, object] = {
        "slurm_job_name": run_name,
        "slurm_partition": partition,
        "slurm_account": account,
        "slurm_nodes": num_nodes,
        "slurm_cpus_per_task": cpus_per_task,
        "slurm_mem": mem,
        "slurm_time": time_minutes,
        "slurm_setup": [pythonpath_setup],
        "slurm_additional_parameters": additional,
    }
    params["slurm_ntasks_per_node"] = 1
    if num_nodes > 1:
        params["slurm_use_srun"] = False
    if not gres:
        params["slurm_gpus_per_node"] = gpus
    executor.update_parameters(**params)
    return executor


def print_slurm_params(
    *,
    gpus: int,
    mem: str,
    partition: str,
    account: str,
    num_nodes: int,
    cpus_per_task: int,
    time_minutes: int,
    constraint: str | None,
    gres: str | None,
    extra: str = "",
) -> None:
    print(f"  partition={partition}  account={account}")
    gpu_field = f"gres={gres}" if gres else f"gpus_per_node={gpus}"
    print(
        f"  nodes={num_nodes}  {gpu_field}  "
        f"cpus={cpus_per_task}  mem={mem}  time={time_minutes}min"
    )
    if constraint:
        print(f"  constraint={constraint}")
    if extra:
        print(f"  {extra}")


def _sanitize_ref_for_dir(ref: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", ref).strip("-.")
    return token or "ref"


def _build_unique_worktree_path(
    *, project_root: str, git_ref: str, git_worktree_dir: str
) -> Path:
    base = Path(git_worktree_dir)
    if not base.is_absolute():
        base = Path(project_root) / base
    uid = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    return base / f"{_sanitize_ref_for_dir(git_ref)}_{uid}"


def _link_worktree_outputs(*, worktree_path: Path, project_root: Path) -> None:
    """Expose the canonical outputs tree inside an ephemeral git worktree.

    Checkpoints may store repo-relative artifact paths such as
    ``outputs/train_codecs/...``. A temporary git worktree only contains
    tracked files, so those relative artifact paths would otherwise resolve
    against an empty/missing ``outputs`` tree. Linking ``outputs`` preserves the
    checkpoint path contract while the runnable source comes from ``git_ref``.
    """
    source = project_root / "outputs"
    target = worktree_path / "outputs"
    if not source.exists() or not worktree_path.exists() or target.exists():
        return
    target.symlink_to(source, target_is_directory=True)


@contextmanager
def _job_workdir(
    *,
    project_root: str,
    git_ref: str | None,
    git_worktree_dir: str,
    os_module: Any,
    subprocess_module: Any,
):
    root = Path(project_root).resolve()
    if not git_ref:
        os_module.chdir(str(root))
        yield str(root)
        return

    worktree_path = _build_unique_worktree_path(
        project_root=str(root),
        git_ref=git_ref,
        git_worktree_dir=git_worktree_dir,
    )
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess_module.run(
        [
            "git",
            "-C",
            str(root),
            "worktree",
            "add",
            "--detach",
            str(worktree_path),
            git_ref,
        ],
        check=True,
    )
    _link_worktree_outputs(worktree_path=worktree_path, project_root=root)
    try:
        os_module.chdir(str(worktree_path))
        yield str(worktree_path)
    finally:
        os_module.chdir(str(root))
        subprocess_module.run(
            [
                "git",
                "-C",
                str(root),
                "worktree",
                "remove",
                "--force",
                str(worktree_path),
            ],
            check=True,
        )


def _absolutize_hydra_path_overrides(
    command: list[str], project_root: str
) -> list[str]:
    always_path_keys = {
        "output_dir",
        "eval.output_dir",
        "resume_checkpoint",
    }
    maybe_path_keys = {
        "policy.pretrained_path",
    }

    rewritten: list[str] = []
    for arg in command:
        if "=" not in arg:
            rewritten.append(arg)
            continue
        key, value = arg.split("=", 1)
        if key not in always_path_keys and key not in maybe_path_keys:
            rewritten.append(arg)
            continue
        if value in {"", "null", "None"}:
            rewritten.append(arg)
            continue

        value_path = Path(value)
        if value_path.is_absolute():
            rewritten.append(arg)
            continue

        if key in maybe_path_keys:
            candidate = Path(project_root) / value_path
            if not candidate.exists():
                rewritten.append(arg)
                continue

        abs_value = str((Path(project_root) / value_path).resolve())
        rewritten.append(f"{key}={abs_value}")
    return rewritten


def _quote_multinode_torchrun_arg(arg: str) -> str:
    if arg == f"--node_rank={SLURM_MACHINE_RANK_PLACEHOLDER}":
        return '--node_rank="${SLURM_PROCID}"'
    if arg == f"--master_addr={MASTER_ADDR_PLACEHOLDER}":
        return '--master_addr="${MASTER_ADDR}"'
    return shlex.quote(arg)


def run_training_job(
    command: list[str],
    project_root: str,
    num_nodes: int = 1,
    auto_resume_on_requeue: bool = False,
    git_ref: str | None = None,
    git_worktree_dir: str = "outputs/worktree",
    *,
    os_module: Any = os,
    subprocess_module: Any = subprocess,
) -> None:
    """Execute a training command inside a Slurm job."""
    command = _absolutize_hydra_path_overrides(command, project_root)
    os_module.environ["HYDRA_FULL_ERROR"] = "1"
    if auto_resume_on_requeue:
        has_resume_override = any(
            arg.startswith("resume_checkpoint=") for arg in command
        )
        has_run_mode_override = any(arg.startswith("run_mode=") for arg in command)
        if has_resume_override and not has_run_mode_override:
            command = [*command, "run_mode=resume"]
        if not has_resume_override:
            output_dir_arg = next(
                (arg for arg in command if arg.startswith("output_dir=")), None
            )
            if output_dir_arg is not None:
                output_dir = output_dir_arg.split("=", 1)[1]
                latest_ckpt = Path(output_dir) / "checkpoints" / "latest"
                if latest_ckpt.exists():
                    command = [
                        *command,
                        "run_mode=resume",
                        f"resume_checkpoint={latest_ckpt}",
                    ]
                    print(f"[resume] auto-resuming from {latest_ckpt}")

    with _job_workdir(
        project_root=project_root,
        git_ref=git_ref,
        git_worktree_dir=git_worktree_dir,
        os_module=os_module,
        subprocess_module=subprocess_module,
    ):
        if num_nodes > 1:
            command_for_srun = " ".join(
                _quote_multinode_torchrun_arg(arg) for arg in command
            )
            shell_cmd = (
                'MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"; '
                "export MASTER_ADDR; " + command_for_srun
            )
            srun_cmd = [
                "srun",
                f"--nodes={num_nodes}",
                f"--ntasks={num_nodes}",
                "--ntasks-per-node=1",
                "--kill-on-bad-exit=1",
                "bash",
                "-lc",
                shell_cmd,
            ]
            subprocess_module.run(srun_cmd, check=True)
        else:
            subprocess_module.run(command, check=True)


def run_eval_monitor_job(
    command: list[str],
    project_root: str,
    git_ref: str | None = None,
    git_worktree_dir: str = "outputs/worktree",
    *,
    os_module: Any = os,
    subprocess_module: Any = subprocess,
) -> None:
    """Execute an eval-monitor command inside a Slurm job."""
    with _job_workdir(
        project_root=project_root,
        git_ref=git_ref,
        git_worktree_dir=git_worktree_dir,
        os_module=os_module,
        subprocess_module=subprocess_module,
    ):
        subprocess_module.run(command, check=True)


def run_eval_job(
    command: list[str],
    project_root: str,
    git_ref: str | None = None,
    git_worktree_dir: str = "outputs/worktree",
    *,
    os_module: Any = os,
    subprocess_module: Any = subprocess,
) -> None:
    """Execute a one-off eval command inside a Slurm job."""
    command = _absolutize_hydra_path_overrides(command, project_root)
    with _job_workdir(
        project_root=project_root,
        git_ref=git_ref,
        git_worktree_dir=git_worktree_dir,
        os_module=os_module,
        subprocess_module=subprocess_module,
    ):
        subprocess_module.run(command, check=True)
