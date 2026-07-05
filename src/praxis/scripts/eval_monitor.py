# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Eval monitor: watches candidate checkpoints and evaluates them out-of-process.

Usage::

    praxis-eval-monitor \\
        --train-dir outputs/train/2026-03-09/run_xyz/ \\
        --num-eval-per-task 10 \\
        --num-parallel-env 10

The monitor polls ``best/topk_manifest.json`` and ``recent/recent_manifest.json``
in the checkpoint directory, evaluates deduped candidate checkpoints using
``praxis-eval`` as a subprocess, and optionally logs results to the training wandb run.

Eval config (policy family, env type/task, device) is derived from each
checkpoint's ``pretrained_model/train_config.yaml``.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

from praxis_eval.evaluation.artifacts import resolve_eval_step_dir
from praxis_eval.evaluation.overrides import (
    build_eval_overrides_from_train_config as _build_eval_overrides_from_train_config,
    normalize_eval_overrides,
)

from praxis.scripts.eval_monitor_modules.candidates import (
    read_candidate_manifest as _read_candidate_manifest_impl,
    read_manifest as _read_manifest_impl,
)
from praxis.scripts.eval_monitor_modules.filesystem import (
    cleanup_eval_dir as _cleanup_eval_dir_impl,
    cleanup_staging_dir as _cleanup_staging_dir_impl,
    cleanup_unneeded_eval_dirs as _cleanup_unneeded_eval_dirs_impl,
    has_results as _has_results_impl,
    read_evaluated_steps as _read_evaluated_steps_impl,
    read_results as _read_results_impl,
    read_retained_eval_steps as _read_retained_eval_steps_impl,
    snapshot_candidate as _snapshot_candidate_impl,
    staging_dir_for_step as _staging_dir_for_step_impl,
    write_cumulative_results as _write_cumulative_results_impl,
    write_evaluated_steps as _write_evaluated_steps_impl,
)
from praxis.scripts.eval_monitor_modules.main import (
    MonitorState as _MonitorState,
    check_inline_eval_conflict as _check_inline_eval_conflict_impl,
    cleanup_current_eval as _cleanup_current_eval_impl,
    collect_finished_eval as _collect_finished_eval_impl,
    handle_missing_manifest as _handle_missing_manifest_impl,
    infer_env_from_dataset as _infer_env_from_dataset_impl,
    initialize_monitor_state as _initialize_monitor_state_impl,
    launch_next_eval as _launch_next_eval_impl,
    maybe_preempt_current_eval as _maybe_preempt_current_eval_impl,
    maybe_timeout_current_eval as _maybe_timeout_current_eval_impl,
    parse_args as _parse_args_impl,
    positive_int as _positive_int_impl,
    record_eval_failure as _record_eval_failure_impl,
    require_step as _require_step_impl,
    reset_current_eval_state as _reset_current_eval_state_impl,
    run_monitor_loop as _run_monitor_loop_impl,
)
from praxis.scripts.eval_monitor_modules.reporting import (
    WandBState as _WandBStateImpl,
    log_monitor_startup as _log_monitor_startup_impl,
)
from praxis.scripts.eval_monitor_modules.runner import (
    kill_process_tree as _kill_process_tree_impl,
    launch_eval as _launch_eval_impl,
)
from praxis.scripts.eval_monitor_modules.scheduler import (
    should_preempt_current_eval as _should_preempt_current_eval_impl,
    stale_watchdog_expired as _stale_watchdog_expired_impl,
)

logger = logging.getLogger("eval-monitor")

BEST_MANIFEST_FILENAME = "topk_manifest.json"
RECENT_MANIFEST_FILENAME = "recent_manifest.json"
SENTINEL_FILENAME = "TRAINING_COMPLETE"
WANDB_RUN_INFO_FILENAME = "wandb_run_info.json"
RESULTS_FILENAME = "results.json"
HYDRA_CONFIG_PATH = ".hydra/config.yaml"
CUMULATIVE_RESULTS_FILENAME = "eval_results.json"
EVALUATED_STEPS_FILENAME = "evaluated_steps.json"
_MAX_EVAL_RETRIES = 2

_positive_int = _positive_int_impl
_require_step = _require_step_impl


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_parse_args = _parse_args_impl


# ---------------------------------------------------------------------------
# Conflict check
# ---------------------------------------------------------------------------


_check_inline_eval_conflict = partial(
    _check_inline_eval_conflict_impl,
    hydra_config_path=HYDRA_CONFIG_PATH,
)


# ---------------------------------------------------------------------------
_infer_env_from_dataset = _infer_env_from_dataset_impl


# ---------------------------------------------------------------------------
# Manifest / eval directory helpers
# ---------------------------------------------------------------------------


_read_manifest = _read_manifest_impl


_read_candidate_manifest = partial(
    _read_candidate_manifest_impl,
    best_manifest_filename=BEST_MANIFEST_FILENAME,
    recent_manifest_filename=RECENT_MANIFEST_FILENAME,
)


_should_preempt_current_eval = _should_preempt_current_eval_impl


_staging_dir_for_step = _staging_dir_for_step_impl


_has_results = partial(_has_results_impl, results_filename=RESULTS_FILENAME)


_read_results = partial(_read_results_impl, results_filename=RESULTS_FILENAME)


_cleanup_eval_dir = partial(
    _cleanup_eval_dir_impl,
    results_filename=RESULTS_FILENAME,
    logger=logger,
)


_cleanup_staging_dir = partial(_cleanup_staging_dir_impl, logger=logger)


_read_evaluated_steps = partial(
    _read_evaluated_steps_impl,
    evaluated_steps_filename=EVALUATED_STEPS_FILENAME,
)


_write_evaluated_steps = partial(
    _write_evaluated_steps_impl,
    evaluated_steps_filename=EVALUATED_STEPS_FILENAME,
)


_read_retained_eval_steps = partial(
    _read_retained_eval_steps_impl,
    best_manifest_filename=BEST_MANIFEST_FILENAME,
)


_cleanup_unneeded_eval_dirs = partial(
    _cleanup_unneeded_eval_dirs_impl,
    cleanup_eval_dir_fn=_cleanup_eval_dir,
)


_write_cumulative_results = partial(
    _write_cumulative_results_impl,
    cumulative_results_filename=CUMULATIVE_RESULTS_FILENAME,
)


_snapshot_candidate = partial(_snapshot_candidate_impl, logger=logger)


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------


_kill_process_tree = _kill_process_tree_impl


def _launch_eval(
    eval_command: list[str],
    pretrained_path: Path,
    eval_output_dir: Path,
    num_eval_per_task: int,
    num_parallel_env: int,
    record_episodes_per_task: int,
    extra_eval_overrides: list[str] | None = None,
) -> subprocess.Popen:
    return _launch_eval_impl(
        eval_command=eval_command,
        pretrained_path=pretrained_path,
        eval_output_dir=eval_output_dir,
        num_eval_per_task=num_eval_per_task,
        num_parallel_env=num_parallel_env,
        record_episodes_per_task=record_episodes_per_task,
        extra_eval_overrides=extra_eval_overrides,
        build_eval_overrides_fn=partial(
            _build_eval_overrides_from_train_config,
            infer_env_from_dataset=_infer_env_from_dataset,
        ),
        normalize_eval_overrides_fn=normalize_eval_overrides,
        logger=logger,
    )


_WandBState = partial(
    _WandBStateImpl,
    run_info_filename=WANDB_RUN_INFO_FILENAME,
    logger=logger,
)


# ---------------------------------------------------------------------------
# Main loop helpers
# ---------------------------------------------------------------------------


_log_monitor_startup = partial(_log_monitor_startup_impl, logger=logger)


_stale_watchdog_expired = _stale_watchdog_expired_impl


_reset_current_eval_state = _reset_current_eval_state_impl


_cleanup_current_eval = partial(
    _cleanup_current_eval_impl,
    cleanup_eval_dir_fn=_cleanup_eval_dir,
    cleanup_staging_dir_fn=_cleanup_staging_dir,
    reset_current_eval_state_fn=_reset_current_eval_state,
    kill_process_tree_fn=_kill_process_tree,
)


_record_eval_failure = partial(
    _record_eval_failure_impl,
    max_eval_retries=_MAX_EVAL_RETRIES,
    logger=logger,
)


_maybe_preempt_current_eval = partial(
    _maybe_preempt_current_eval_impl,
    should_preempt_current_eval_fn=_should_preempt_current_eval,
    require_step_fn=_require_step,
    cleanup_current_eval_fn=_cleanup_current_eval,
    logger=logger,
)


_maybe_timeout_current_eval = partial(
    _maybe_timeout_current_eval_impl,
    require_step_fn=_require_step,
    record_eval_failure_fn=_record_eval_failure,
    cleanup_current_eval_fn=_cleanup_current_eval,
    logger=logger,
)


def _initialize_monitor_state(
    *,
    eval_base: Path,
    checkpoint_dir: Path,
    now: float,
) -> tuple[_MonitorState, Any]:
    from praxis.training.core.checkpoint import TopKCheckpointManager

    return _initialize_monitor_state_impl(
        eval_base=eval_base,
        checkpoint_dir=checkpoint_dir,
        now=now,
        monitor_state_cls=_MonitorState,
        topk_checkpoint_manager_cls=TopKCheckpointManager,
        cumulative_results_filename=CUMULATIVE_RESULTS_FILENAME,
        read_evaluated_steps_fn=_read_evaluated_steps,
        read_retained_eval_steps_fn=_read_retained_eval_steps,
        read_candidate_manifest_fn=_read_candidate_manifest,
        has_results_fn=_has_results,
        write_evaluated_steps_fn=_write_evaluated_steps,
        logger=logger,
    )


_handle_missing_manifest = partial(
    _handle_missing_manifest_impl,
    cleanup_unneeded_eval_dirs_fn=_cleanup_unneeded_eval_dirs,
    stale_watchdog_expired_fn=_stale_watchdog_expired,
    logger=logger,
)


_collect_finished_eval = partial(
    _collect_finished_eval_impl,
    require_step_fn=_require_step,
    write_evaluated_steps_fn=_write_evaluated_steps,
    read_results_fn=_read_results,
    write_cumulative_results_fn=_write_cumulative_results,
    staging_dir_for_step_fn=_staging_dir_for_step,
    record_eval_failure_fn=_record_eval_failure,
    cleanup_current_eval_fn=_cleanup_current_eval,
    logger=logger,
)


_launch_next_eval = partial(
    _launch_next_eval_impl,
    eval_dir_for_step_fn=resolve_eval_step_dir,
    snapshot_candidate_fn=_snapshot_candidate,
    launch_eval_fn=_launch_eval,
    record_eval_failure_fn=_record_eval_failure,
    cleanup_current_eval_fn=_cleanup_current_eval,
    logger=logger,
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    train_dir = args.train_dir.resolve()
    checkpoint_dir = (args.checkpoint_dir or train_dir / "checkpoints").resolve()
    sentinel_path = train_dir / SENTINEL_FILENAME
    eval_base = train_dir / "eval"
    eval_command = shlex.split(args.eval_command)
    eval_overrides = list(args.eval_override)
    poll_interval = args.poll_interval
    eval_timeout = args.eval_timeout_sec

    _log_monitor_startup(
        train_dir=train_dir,
        checkpoint_dir=checkpoint_dir,
        poll_interval=poll_interval,
        num_eval_per_task=args.num_eval_per_task,
        num_parallel_env=args.num_parallel_env,
        stale_timeout_sec=args.stale_timeout_sec,
        eval_timeout_sec=eval_timeout,
        eval_overrides=eval_overrides,
    )

    # --- Conflict check: abort if trainer has inline sim eval enabled ---
    conflict = _check_inline_eval_conflict(train_dir)
    if conflict is True:
        logger.error(
            "trainer.sim_eval_every is set in training config. "
            "Disable inline sim eval (trainer.sim_eval_every=null) before using eval-monitor."
        )
        sys.exit(1)
    conflict_verified = conflict is False  # None means config not found yet

    wb = _WandBState(train_dir, disabled=args.no_wandb)
    state, eval_topk = _initialize_monitor_state(
        eval_base=eval_base,
        checkpoint_dir=checkpoint_dir,
        now=time.time(),
    )

    try:
        _run_monitor_loop_impl(
            checkpoint_dir=checkpoint_dir,
            sentinel_path=sentinel_path,
            eval_base=eval_base,
            state=state,
            eval_topk=eval_topk,
            wb=wb,
            eval_command=eval_command,
            eval_overrides=eval_overrides,
            num_eval_per_task=args.num_eval_per_task,
            num_parallel_env=args.num_parallel_env,
            record_episodes_per_task=args.record_episodes_per_task,
            stale_timeout_sec=args.stale_timeout_sec,
            poll_interval=poll_interval,
            eval_timeout=eval_timeout,
            train_dir=train_dir,
            conflict_verified=conflict_verified,
            read_candidate_manifest_fn=_read_candidate_manifest,
            read_retained_eval_steps_fn=_read_retained_eval_steps,
            handle_missing_manifest_fn=_handle_missing_manifest,
            check_inline_eval_conflict_fn=_check_inline_eval_conflict,
            maybe_preempt_current_eval_fn=_maybe_preempt_current_eval,
            maybe_timeout_current_eval_fn=_maybe_timeout_current_eval,
            collect_finished_eval_fn=_collect_finished_eval,
            cleanup_unneeded_eval_dirs_fn=_cleanup_unneeded_eval_dirs,
            launch_next_eval_fn=_launch_next_eval,
            stale_watchdog_expired_fn=_stale_watchdog_expired,
            logger=logger,
        )

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        if state.current_proc is not None:
            logger.info("Cleaning up eval process...")
            _cleanup_current_eval(
                state,
                eval_base=eval_base,
                keep_results=False,
                kill_process=True,
            )
        wb.finish()


if __name__ == "__main__":
    main()
