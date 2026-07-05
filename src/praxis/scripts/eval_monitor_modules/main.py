# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Main-loop utility helpers for eval monitor."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MonitorState:
    evaluated: set[int]
    fail_counts: dict[int, int]
    failed: set[int]
    current_proc: Any = None
    current_step: int | None = None
    current_tier: str | None = None
    current_launch_ts: float = 0.0
    last_progress_ts: float = 0.0
    cumulative_results: dict[int, dict[str, float]] = field(default_factory=dict)


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


def positive_int(value: str) -> int:
    """argparse type: integer > 0."""
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected int > 0, got {value!r}") from exc
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected int > 0, got {ivalue}")
    return ivalue


def parse_args() -> argparse.Namespace:
    """Parse monitor args.

    Primary eval knobs are ``num_eval_per_task`` and ``num_parallel_env``.
    Policy/env settings are derived from each checkpoint's ``train_config.yaml``,
    and optional extra Hydra overrides can be appended via ``--eval-override``.
    """
    parser = argparse.ArgumentParser(
        description="Monitor top-K checkpoints and evaluate asynchronously.",
    )
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--eval-command", default="praxis-eval")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--num-eval-per-task", type=positive_int, required=True)
    parser.add_argument("--num-parallel-env", type=positive_int, required=True)
    parser.add_argument("--record-episodes-per-task", type=int, default=0)
    parser.add_argument("--stale-timeout-sec", type=int, default=10800)
    parser.add_argument(
        "--eval-timeout-sec",
        type=int,
        default=0,
        help="Kill an eval subprocess if it runs longer than this (0=disabled).",
    )
    parser.add_argument(
        "--eval-override",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Extra Hydra overrides appended to each eval subprocess. "
            "Repeat the flag or pass multiple values after one flag."
        ),
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    args.eval_override = flatten_eval_overrides(args.eval_override)
    return args


def require_step(step: int | None) -> int:
    """Validate that current eval step is populated."""
    if step is None:
        raise RuntimeError(
            "Internal error: eval subprocess exists but current_step is unset."
        )
    return int(step)


def check_inline_eval_conflict(
    train_dir: Path, *, hydra_config_path: str
) -> bool | None:
    """Check if the training config has inline sim eval enabled.

    Returns ``True`` if conflict, ``False`` if safe, ``None`` if config not found
    or parsing fails.
    """
    config_path = train_dir / hydra_config_path
    if not config_path.exists():
        return None
    try:
        import yaml

        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        sim_eval_every = cfg.get("trainer", {}).get("sim_eval_every")
        return sim_eval_every is not None
    except Exception:
        return None


def infer_env_from_dataset(dataset_name: str) -> tuple[str, str]:
    """Return ``(env_type, env_task)`` for a dataset, or ``("", "")`` on failure."""
    from praxis_eval.envs.factory import infer_eval_env_target

    return infer_eval_env_target(dataset_name)


def reset_current_eval_state(
    state: Any,
    *,
    mark_progress: bool = False,
) -> None:
    state.current_proc = None
    state.current_step = None
    state.current_tier = None
    state.current_launch_ts = 0.0
    if mark_progress:
        state.last_progress_ts = time.time()


def cleanup_current_eval(
    state: Any,
    *,
    eval_base,
    keep_results: bool,
    cleanup_eval_dir_fn: Callable[..., None],
    cleanup_staging_dir_fn: Callable[..., None],
    reset_current_eval_state_fn: Callable[..., None],
    kill_process_tree_fn: Callable[[Any], None],
    kill_process: bool = False,
    mark_progress: bool = False,
) -> None:
    step = state.current_step
    proc = state.current_proc
    if kill_process and proc is not None:
        kill_process_tree_fn(proc)
    if step is not None:
        cleanup_eval_dir_fn(eval_base, step, keep_results=keep_results)
        cleanup_staging_dir_fn(eval_base, step)
    reset_current_eval_state_fn(state, mark_progress=mark_progress)


def record_eval_failure(
    state: Any,
    *,
    step: int,
    rc: int,
    max_eval_retries: int,
    logger,
) -> None:
    state.fail_counts[step] = state.fail_counts.get(step, 0) + 1
    attempts = state.fail_counts[step]
    if attempts >= max_eval_retries:
        logger.warning(
            "Eval step %d failed (rc=%d), %d/%d attempts. Skipping.",
            step,
            rc,
            attempts,
            max_eval_retries,
        )
        state.failed.add(step)
    else:
        logger.warning(
            "Eval step %d failed (rc=%d), %d/%d attempts. Will retry.",
            step,
            rc,
            attempts,
            max_eval_retries,
        )


def maybe_preempt_current_eval(
    *,
    state: Any,
    eval_base,
    manifest: list[dict[str, Any]],
    should_preempt_current_eval_fn: Callable[..., bool],
    require_step_fn: Callable[[int | None], int],
    cleanup_current_eval_fn: Callable[..., None],
    logger,
) -> None:
    if state.current_proc is None or not should_preempt_current_eval_fn(
        current_tier=state.current_tier,
        manifest=manifest,
        evaluated=state.evaluated,
        failed=state.failed,
    ):
        return

    step = require_step_fn(state.current_step)
    logger.info(
        "Preempting recent-tier eval (step %d) for unevaluated best-tier candidate.",
        step,
    )
    cleanup_current_eval_fn(
        state,
        eval_base=eval_base,
        keep_results=False,
        kill_process=True,
        mark_progress=True,
    )


def maybe_timeout_current_eval(
    *,
    state: Any,
    eval_base,
    eval_timeout_sec: int,
    require_step_fn: Callable[[int | None], int],
    record_eval_failure_fn: Callable[..., None],
    cleanup_current_eval_fn: Callable[..., None],
    logger,
) -> None:
    """Account timed-out evals as failures before killing their subprocess."""
    if state.current_proc is None or eval_timeout_sec <= 0:
        return

    step = require_step_fn(state.current_step)
    elapsed = time.time() - state.current_launch_ts
    if elapsed < eval_timeout_sec:
        return

    logger.warning(
        "Eval for step %d exceeded timeout (%ds). Killing.",
        step,
        eval_timeout_sec,
    )
    record_eval_failure_fn(state, step=step, rc=-9)
    cleanup_current_eval_fn(
        state,
        eval_base=eval_base,
        keep_results=False,
        kill_process=True,
        mark_progress=True,
    )


def initialize_monitor_state(
    *,
    eval_base: Path,
    checkpoint_dir: Path,
    now: float,
    monitor_state_cls: Callable[..., Any],
    topk_checkpoint_manager_cls,
    cumulative_results_filename: str,
    read_evaluated_steps_fn: Callable[[Path], set[int]],
    read_retained_eval_steps_fn: Callable[[Path], set[int]],
    read_candidate_manifest_fn: Callable[[Path], list[dict[str, Any]] | None],
    has_results_fn: Callable[[Path, int], bool],
    write_evaluated_steps_fn: Callable[[Path, set[int]], None],
    logger,
) -> tuple[Any, Any]:
    state = monitor_state_cls(
        evaluated=read_evaluated_steps_fn(eval_base),
        fail_counts={},
        failed=set(),
        last_progress_ts=now,
    )
    eval_topk = topk_checkpoint_manager_cls(
        k=2,
        mode="max",
        metric_key="success_rate",
        directory=eval_base / "checkpoints",
    )
    state.evaluated |= read_retained_eval_steps_fn(eval_base)

    manifest = read_candidate_manifest_fn(checkpoint_dir)
    if manifest is not None:
        for entry in manifest:
            step = entry["step"]
            if has_results_fn(eval_base, step):
                state.evaluated.add(step)
                logger.info("Recovered existing eval for step %d", step)

    existing_cumulative = eval_base / cumulative_results_filename
    if existing_cumulative.exists():
        try:
            loaded = json.loads(existing_cumulative.read_text())
            for key, metrics in loaded.items():
                if isinstance(metrics, dict):
                    step = int(key)
                    state.evaluated.add(step)
                    state.cumulative_results[step] = {
                        metric_key: float(metric_value)
                        for metric_key, metric_value in metrics.items()
                        if isinstance(metric_value, (int, float))
                    }
            logger.info(
                "Recovered %d cumulative eval results",
                len(state.cumulative_results),
            )
        except (json.JSONDecodeError, OSError, ValueError):
            logger.warning("Failed to load cumulative results; starting fresh.")

    write_evaluated_steps_fn(eval_base, state.evaluated)
    return state, eval_topk


def handle_missing_manifest(
    *,
    training_complete: bool,
    eval_base: Path,
    retained_eval_steps: set[int],
    state: Any,
    stale_timeout_sec: int,
    poll_interval: int,
    cleanup_unneeded_eval_dirs_fn: Callable[..., None],
    stale_watchdog_expired_fn: Callable[..., bool],
    logger,
) -> bool:
    if training_complete:
        cleanup_unneeded_eval_dirs_fn(
            eval_base,
            retained_steps=retained_eval_steps,
            active_step=state.current_step,
        )
        logger.info("Training complete, no manifest. Exiting.")
        return True

    if state.current_proc is None and stale_watchdog_expired_fn(
        last_progress_ts=state.last_progress_ts,
        stale_timeout_sec=stale_timeout_sec,
    ):
        idle_sec = time.time() - state.last_progress_ts
        logger.error(
            "No monitor progress for %ds without manifest/sentinel. "
            "Assuming trainer stalled/crashed and exiting.",
            int(idle_sec),
        )
        sys.exit(2)

    time.sleep(poll_interval)
    return False


def collect_finished_eval(
    *,
    state: Any,
    eval_base: Path,
    eval_topk,
    wb,
    require_step_fn: Callable[[int | None], int],
    write_evaluated_steps_fn: Callable[[Path, set[int]], None],
    read_results_fn: Callable[[Path, int], dict[str, Any] | None],
    write_cumulative_results_fn: Callable[[Path, dict[int, dict[str, float]]], None],
    staging_dir_for_step_fn: Callable[[Path, int], Path],
    record_eval_failure_fn: Callable[..., None],
    cleanup_current_eval_fn: Callable[..., None],
    logger,
) -> None:
    if state.current_proc is None:
        return
    step = require_step_fn(state.current_step)
    rc = state.current_proc.poll()
    if rc is None:
        return

    if rc == 0:
        logger.info("Eval step %d completed.", step)
        state.evaluated.add(step)
        write_evaluated_steps_fn(eval_base, state.evaluated)
        state.last_progress_ts = time.time()
        results = read_results_fn(eval_base, step)
        if results and "overall" in results:
            overall = results["overall"]
            success_rate = float(overall.get("success_rate", 0.0))
            avg_episode_length = float(overall.get("avg_episode_length", 0.0))
            wandb_metrics = {
                "sim_eval/success_rate": success_rate,
                "sim_eval/avg_episode_length": avg_episode_length,
            }
            wb.log(
                step,
                wandb_metrics,
            )
            state.cumulative_results[step] = {
                "success_rate": success_rate,
                "avg_episode_length": avg_episode_length,
            }
            write_cumulative_results_fn(eval_base, state.cumulative_results)

            def _promote_fn(dest: Path, step: int = step) -> None:
                src = staging_dir_for_step_fn(eval_base, step) / "pretrained_model"
                if not src.exists():
                    raise FileNotFoundError(f"Staging checkpoint missing: {src}")
                if dest.exists():
                    shutil.rmtree(dest, ignore_errors=True)
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest / "pretrained_model", dirs_exist_ok=False)

            try:
                eval_topk.consider(
                    step,
                    {"success_rate": success_rate},
                    _promote_fn,
                )
            except FileNotFoundError:
                logger.warning(
                    "Step %d staging missing before promotion; skipping eval top-K.",
                    step,
                )
    else:
        record_eval_failure_fn(state, step=step, rc=rc)

    cleanup_current_eval_fn(
        state,
        eval_base=eval_base,
        keep_results=(rc == 0),
    )


def launch_next_eval(
    *,
    manifest: list[dict[str, Any]],
    state: Any,
    eval_base: Path,
    eval_command: list[str],
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
    eval_overrides: list[str],
    eval_dir_for_step_fn: Callable[[Path, int], Path],
    snapshot_candidate_fn: Callable[[dict[str, Any], Path], Path],
    launch_eval_fn,
    record_eval_failure_fn: Callable[..., None],
    cleanup_current_eval_fn: Callable[..., None],
    logger,
) -> None:
    if state.current_proc is not None:
        return

    for entry in manifest:
        step = entry["step"]
        if step in state.evaluated or step in state.failed:
            continue
        eval_out = eval_dir_for_step_fn(eval_base, step)
        try:
            pretrained_path = snapshot_candidate_fn(entry, eval_base)
            state.current_step = step
            state.current_tier = entry["tier"]
            state.current_proc = launch_eval_fn(
                eval_command=eval_command,
                pretrained_path=pretrained_path,
                eval_output_dir=eval_out,
                num_eval_per_task=num_eval_per_task,
                num_parallel_env=num_parallel_env,
                record_episodes_per_task=record_episodes_per_task,
                extra_eval_overrides=eval_overrides,
            )
            state.current_launch_ts = time.time()
            state.last_progress_ts = state.current_launch_ts
        except Exception as exc:
            logger.exception("Failed to launch eval for step %d: %s", step, exc)
            record_eval_failure_fn(state, step=step, rc=-1)
            cleanup_current_eval_fn(
                state,
                eval_base=eval_base,
                keep_results=False,
            )
        break


def run_monitor_loop(
    *,
    checkpoint_dir: Path,
    sentinel_path: Path,
    eval_base: Path,
    state: Any,
    eval_topk,
    wb,
    eval_command: list[str],
    eval_overrides: list[str],
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
    stale_timeout_sec: int,
    poll_interval: int,
    eval_timeout: int,
    train_dir: Path,
    conflict_verified: bool,
    read_candidate_manifest_fn: Callable[[Path], list[dict[str, Any]] | None],
    read_retained_eval_steps_fn: Callable[[Path], set[int]],
    handle_missing_manifest_fn: Callable[..., bool],
    check_inline_eval_conflict_fn: Callable[[Path], bool | None],
    maybe_preempt_current_eval_fn: Callable[..., None],
    maybe_timeout_current_eval_fn: Callable[..., None],
    collect_finished_eval_fn: Callable[..., None],
    cleanup_unneeded_eval_dirs_fn: Callable[..., None],
    launch_next_eval_fn: Callable[..., None],
    stale_watchdog_expired_fn: Callable[..., bool],
    logger,
) -> None:
    while True:
        manifest = read_candidate_manifest_fn(checkpoint_dir)
        training_complete = sentinel_path.exists()
        retained_eval_steps = read_retained_eval_steps_fn(eval_base)

        if manifest is None:
            if handle_missing_manifest_fn(
                training_complete=training_complete,
                eval_base=eval_base,
                retained_eval_steps=retained_eval_steps,
                state=state,
                stale_timeout_sec=stale_timeout_sec,
                poll_interval=poll_interval,
            ):
                break
            continue

        # Deferred conflict check (monitor started before trainer).
        if not conflict_verified:
            conflict = check_inline_eval_conflict_fn(train_dir)
            if conflict is True:
                logger.error(
                    "trainer.sim_eval_every is set in training config. "
                    "Disable inline sim eval (trainer.sim_eval_every=null) "
                    "before using eval-monitor."
                )
                sys.exit(1)
            if conflict is False:
                conflict_verified = True

        candidate_steps = {e["step"] for e in manifest}

        maybe_preempt_current_eval_fn(
            state=state,
            eval_base=eval_base,
            manifest=manifest,
        )
        maybe_timeout_current_eval_fn(
            state=state,
            eval_base=eval_base,
            eval_timeout_sec=eval_timeout,
        )
        collect_finished_eval_fn(
            state=state,
            eval_base=eval_base,
            eval_topk=eval_topk,
            wb=wb,
        )

        retained_eval_steps = read_retained_eval_steps_fn(eval_base)
        cleanup_unneeded_eval_dirs_fn(
            eval_base,
            retained_steps=retained_eval_steps,
            active_step=state.current_step,
        )

        # Clear failed/retry state for evicted steps.
        for step in list(state.fail_counts):
            if step not in candidate_steps:
                del state.fail_counts[step]
        state.failed -= state.failed - candidate_steps

        launch_next_eval_fn(
            manifest=manifest,
            state=state,
            eval_base=eval_base,
            eval_command=eval_command,
            num_eval_per_task=num_eval_per_task,
            num_parallel_env=num_parallel_env,
            record_episodes_per_task=record_episodes_per_task,
            eval_overrides=eval_overrides,
        )

        if (
            training_complete
            and state.current_proc is None
            and all(
                e["step"] in state.evaluated or e["step"] in state.failed
                for e in manifest
            )
        ):
            logger.info("All candidates evaluated and training done. Exiting.")
            break

        if (
            not training_complete
            and state.current_proc is None
            and stale_watchdog_expired_fn(
                last_progress_ts=state.last_progress_ts,
                stale_timeout_sec=stale_timeout_sec,
            )
        ):
            idle_sec = time.time() - state.last_progress_ts
            logger.error(
                "No monitor progress for %ds without TRAINING_COMPLETE sentinel. "
                "Assuming trainer stalled/crashed and exiting.",
                int(idle_sec),
            )
            sys.exit(2)

        time.sleep(poll_interval)
