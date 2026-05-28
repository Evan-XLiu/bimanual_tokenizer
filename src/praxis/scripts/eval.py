"""Evaluation entry point."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import hydra
import praxis_eval
from omegaconf import DictConfig, OmegaConf
from praxis_eval.evaluation.config import (
    env_kwargs_without_type_task,
    env_type_from_cfg,
    optional_env_task,
    optional_env_task_ids,
    require_positive_int,
    resolve_nonnegative_int,
    resolve_optional_positive_timeout_sec,
    resolve_policy_kwargs,
)
from praxis_eval.evaluation.watchdog import (
    EvalPhaseWatchdog,
    resolve_phase_watchdog_threshold_sec,
)
from praxis_eval.policies.remote import RemotePolicy
from praxis_eval.types import Policy

from praxis.core.devices import require_available_device

CONFIG_PATH = str(Path(__file__).parents[1] / "configs")
_LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="eval_policy")
def main(cfg: DictConfig) -> None:
    """Evaluate a policy on simulation environments.

    Two modes:
      - Local (default): loads policy from checkpoint (self-contained).
      - Remote (server_address set): connects to a PolicyServer over gRPC.

    Policy checkpoints must contain ``policy_io.json`` and ``norm_stats.pt``.
    No dataset download is required.
    """
    print(OmegaConf.to_yaml(cfg))
    process_start_monotonic = time.monotonic()

    from praxis.policies.eval_adapter import PolicyEvalAdapter

    eval_cfg = cfg.eval
    watchdog_threshold_sec = resolve_phase_watchdog_threshold_sec(
        OmegaConf.select(eval_cfg, "phase_watchdog_sec", default=None)
    )
    eval_watchdog: EvalPhaseWatchdog | None = None
    if watchdog_threshold_sec is not None:
        eval_watchdog = EvalPhaseWatchdog(
            threshold_sec=float(watchdog_threshold_sec),
            process_start_monotonic=process_start_monotonic,
        )
        eval_watchdog.start()
        _LOGGER.info(
            "WATCHDOG enabled pid=%d wall=%s threshold=%.1fs",
            os.getpid(),
            time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            float(watchdog_threshold_sec),
        )

    def _phase_heartbeat(label: str) -> None:
        if eval_watchdog is not None:
            eval_watchdog.mark_phase(label)
        else:
            _LOGGER.info(
                "PHASE label=%s pid=%d wall=%s mono_elapsed=%.3fs",
                label,
                os.getpid(),
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                time.monotonic() - process_start_monotonic,
            )

    def _progress_heartbeat(label: str) -> None:
        if eval_watchdog is not None:
            eval_watchdog.mark_progress(label)
        else:
            _LOGGER.info(
                "HEARTBEAT label=%s pid=%d wall=%s mono_elapsed=%.3fs",
                label,
                os.getpid(),
                time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                time.monotonic() - process_start_monotonic,
            )

    _phase_heartbeat("eval_main_entry")

    num_eval_per_task = require_positive_int(
        int(eval_cfg.num_eval_per_task), name="eval.num_eval_per_task"
    )
    num_parallel_env = require_positive_int(
        int(eval_cfg.num_parallel_env), name="eval.num_parallel_env"
    )
    eval_mode = str(eval_cfg.mode).strip().lower()
    eval_output_dir = eval_cfg.output_dir
    eval_record_episodes_per_task = int(eval_cfg.record_episodes_per_task)
    eval_debug_verbose = bool(eval_cfg.debug_verbose)
    eval_step_timeout_sec = resolve_optional_positive_timeout_sec(eval_cfg)
    eval_rollout_failure_retries = resolve_nonnegative_int(
        eval_cfg,
        key="rollout_failure_retries",
        label="eval.rollout_failure_retries",
        default=1,
    )
    eval_policy_kwargs = resolve_policy_kwargs(
        eval_cfg,
        label="eval.policy_kwargs",
    )

    if eval_output_dir is None:
        raise ValueError("Missing required `eval.output_dir` (user-specified).")

    # Build policy mode: remote transport or local checkpoint.
    server_address = cfg.server_address
    policy = None
    eval_policy: Policy | None = None

    if eval_mode == "remote_grpc":
        if not server_address:
            raise ValueError(
                "eval.mode=remote_grpc requires `server_address=host:port`."
            )
        from praxis.policies.eval_adapter import EnvironmentActionBoundsPolicyAdapter

        remote_policy = RemotePolicy(str(server_address))
        eval_policy = EnvironmentActionBoundsPolicyAdapter(remote_policy)
        ready, info = remote_policy.client.health_check()
        print(f"Connected to remote policy server: {info} (ready={ready})")
        io = None  # Remote mode: env config must be fully user-specified.
    elif eval_mode == "local_inproc":
        if server_address:
            raise ValueError(
                "server_address is set but eval.mode=local_inproc. "
                "Set eval.mode=remote_grpc to use gRPC evaluation."
            )
        # Local mode: load self-contained policy from checkpoint.
        from praxis.policies.factory import make_policy

        eval_device = require_available_device(cfg.device, purpose="Evaluation")
        _phase_heartbeat("before_make_policy")
        policy = make_policy(cfg.policy)
        _phase_heartbeat("after_make_policy")
        io = policy.io
        if io is None:
            raise RuntimeError("Loaded policy has no PolicyIO. Checkpoint is invalid.")
        eval_policy = PolicyEvalAdapter(
            policy,
            device=eval_device,
            preprocessor=lambda batch: batch,
        )
    else:
        raise ValueError(
            f"Unknown eval.mode: {eval_mode!r}. Available: local_inproc, remote_grpc"
        )

    # Override env config with PolicyIO-derived values when available.
    if io is not None and io.image_features:
        image_height, image_width = io.image_dims
        OmegaConf.update(cfg, "env.observation_height", image_height, force_add=True)
        OmegaConf.update(cfg, "env.observation_width", image_width, force_add=True)
        print(
            f"Env observation dims set from PolicyIO: {image_width}x{image_height}, fps={io.fps}"
        )

    result = None
    try:
        assert eval_policy is not None
        env_type = env_type_from_cfg(cfg.env)
        result = praxis_eval.evaluate(
            env_type,
            policy=eval_policy,
            config=praxis_eval.EvalConfig(
                num_eval_per_task=num_eval_per_task,
                output_dir=eval_output_dir,
                task=optional_env_task(cfg.env),
                task_ids=optional_env_task_ids(cfg.env),
                num_parallel_env=num_parallel_env,
                seed=int(cfg.seed),
                record_episodes_per_task=eval_record_episodes_per_task,
                step_timeout_sec=eval_step_timeout_sec,
                rollout_failure_retries=eval_rollout_failure_retries,
                debug_verbose=eval_debug_verbose,
                policy_kwargs=eval_policy_kwargs,
                env_kwargs=env_kwargs_without_type_task(cfg.env),
                metadata={"mode": eval_mode},
                runtime_hooks=praxis_eval.EvalRuntimeHooks(
                    phase_heartbeat=_phase_heartbeat,
                    progress_heartbeat=_progress_heartbeat,
                ),
            ),
        )
    finally:
        close_policy = getattr(policy, "close", None)
        if callable(close_policy):
            close_policy()
        close_eval_policy = getattr(eval_policy, "close", None)
        if callable(close_eval_policy):
            close_eval_policy()
        if eval_watchdog is not None:
            eval_watchdog.stop()

    if result is None:
        raise RuntimeError("Evaluation failed before producing results.")
    print(f"Overall metrics: {dict(result.overall)}")
    print(f"Saved eval results: {result.artifacts.get('results_path')}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
