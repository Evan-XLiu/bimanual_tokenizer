# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Inline policy evaluation helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import praxis_eval
import torch
from omegaconf import DictConfig, OmegaConf
from praxis_eval.evaluation.config import (
    env_kwargs_without_type_task,
    env_type_from_cfg,
    optional_env_task,
    optional_env_task_ids,
    require_positive_int,
)
from praxis_eval.registry import available_drivers

from praxis.policies.eval_adapter import PolicyEvalAdapter
from praxis.training.core import eval_memory
from praxis.training.core.rank_errors import (
    collect_rank_errors,
    raise_collected_rank_errors,
    raise_if_rank0_error,
    rank0_error_message,
    rank_error_message,
)

EVALUATOR_BACKEND = "praxis_eval"
DONE_MARKER = ".rank0_sim_eval_done"
ERROR_MARKER = ".rank0_sim_eval_error"
RANK_WAIT_TIMEOUT_SEC = 24 * 60 * 60
RANK_WAIT_POLL_SEC = 5.0
SIM_EVAL_MODE = "local_inproc"
LOGGER = logging.getLogger(__name__)


def configured_env_type(env_cfg: DictConfig | dict[str, Any] | None) -> str | None:
    """Return a non-empty configured env type, or None when env is unset."""
    if env_cfg is None:
        return None
    if isinstance(env_cfg, DictConfig):
        env_type = env_cfg.get("type", None)
    else:
        env_type = env_cfg.get("type")
    if env_type is None:
        return None
    env_type_text = str(env_type).strip()
    return env_type_text or None


def resolve_preinit_backend(cfg: DictConfig) -> str | None:
    """Validate config-only sim-eval contracts and resolve the backend early."""
    tc = cfg.trainer
    if tc.sim_eval_every is None:
        return None

    require_positive_int(int(tc.sim_eval_every), name="trainer.sim_eval_every")
    env_cfg_raw = cfg.get("env", None)
    if configured_env_type(env_cfg_raw) is None:
        raise ValueError(
            "trainer.sim_eval_every is set, but cfg.env.type is null or empty. "
            "Inline sim-eval requires an explicit eval env type."
        )
    if str(getattr(tc, "distributed_strategy", "ddp")) == "fsdp":
        raise ValueError(
            "trainer.sim_eval_every is not supported with "
            "trainer.distributed_strategy=fsdp. FSDP policies are sharded "
            "across ranks, while inline sim-eval currently runs a single "
            "rank-local policy. Disable inline sim-eval or export a "
            "pretrained_model checkpoint for one-GPU evaluation."
        )
    return resolve_backend(env_cfg_raw)


def configure_success_rate_topk(
    cfg: DictConfig,
    *,
    sim_eval_backend: str | None,
) -> bool:
    """Set the effective top-K metric before checkpoint managers are built."""
    if sim_eval_backend is None:
        return False

    best_cfg = OmegaConf.select(
        cfg.trainer,
        "checkpoint_candidates.best",
        default=None,
    )
    if best_cfg is None:
        return False
    if str(best_cfg.metric_key) != "val_reconst_loss" or str(best_cfg.mode) != "min":
        return False

    best_cfg.metric_key = "success_rate"
    best_cfg.mode = "max"
    return True


def resolve_backend(env_cfg: DictConfig | dict[str, Any]) -> str:
    """Resolve the inline sim-eval backend for an env family."""
    env_type = env_type_from_cfg(env_cfg)
    available = set(available_drivers())
    if env_type in available:
        return EVALUATOR_BACKEND
    raise ValueError(
        "Inline sim eval does not know how to evaluate env type "
        f"{env_type!r}. Env types with sim-eval support: "
        f"{', '.join(sorted(available)) or '(none)'}"
    )


def clear_rank_markers(eval_out_dir: Path) -> tuple[Path, Path]:
    """Clear stale rank-0 completion markers before a long main-only sim eval."""
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = eval_out_dir / DONE_MARKER
    error_marker = eval_out_dir / ERROR_MARKER
    for marker in (done_marker, error_marker):
        marker.unlink(missing_ok=True)
    return done_marker, error_marker


def wait_for_rank0_marker(
    *,
    done_marker: Path,
    error_marker: Path,
    timeout_sec: float = RANK_WAIT_TIMEOUT_SEC,
    poll_sec: float = RANK_WAIT_POLL_SEC,
) -> None:
    """Wait without entering a distributed collective while rank 0 runs sim eval."""
    start = monotonic()
    while True:
        if done_marker.exists():
            return
        if error_marker.exists():
            message = error_marker.read_text(encoding="utf-8").strip()
            raise RuntimeError(
                "Rank 0 failed during inline sim eval before non-main ranks "
                f"could rejoin: {message}"
            )
        elapsed = monotonic() - start
        if elapsed >= timeout_sec:
            raise TimeoutError(
                "Timed out waiting for rank 0 to finish inline sim eval. "
                f"Waited {elapsed:.1f}s for marker {done_marker}."
            )
        sleep(max(0.1, min(float(poll_sec), float(timeout_sec) - elapsed)))


class InlinePolicyEvalRunner:
    """Runs one trainer-owned inline policy evaluation boundary."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def run(self) -> dict[str, float]:
        trainer = self.trainer
        if trainer.sim_eval_backend is None:
            return {}
        if (
            str(getattr(getattr(trainer, "tc", None), "distributed_strategy", "ddp"))
            == "fsdp"
        ):
            raise RuntimeError(
                "Inline sim-eval is not supported with "
                "trainer.distributed_strategy=fsdp. Export a pretrained_model "
                "checkpoint and run one-GPU evaluation instead."
            )

        eval_out_dir = self._resolve_artifact_dir()
        done_marker, error_marker = self._prepare_rank_markers(eval_out_dir)
        trainer.runtime.wait_for_everyone()

        metrics: dict[str, float] = {}
        body_error: BaseException | None = None
        train_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        optimizer_restore_error: BaseException | None = None
        optimizer_offload: eval_memory.OptimizerStateCpuOffload | None = None
        try:
            if trainer.runtime.is_main_process:
                try:
                    optimizer_offload = self._offload_rank0_optimizer_state_for_eval()
                    metrics = self._run_rank0_eval(
                        eval_out_dir=eval_out_dir,
                    )
                    done_marker.write_text("ok\n", encoding="utf-8")
                except BaseException as exc:
                    error_marker.write_text(
                        f"{type(exc).__name__}: {exc}\n",
                        encoding="utf-8",
                    )
                    raise
            else:
                wait_for_rank0_marker(
                    done_marker=done_marker,
                    error_marker=error_marker,
                )
        except BaseException as exc:
            body_error = exc
        finally:
            train_error = self._restore_train_mode()
            cleanup_error = self._release_cuda_eval_memory()
            optimizer_restore_error = self._restore_rank0_optimizer_state(
                optimizer_offload
            )

        if body_error is not None:
            raise body_error
        self._raise_if_rank_error("Sim-eval train-mode restore", train_error)
        self._raise_if_rank_error("Sim-eval cleanup", cleanup_error)
        self._raise_if_rank0_error(
            "Sim-eval optimizer-state restore",
            optimizer_restore_error,
        )
        trainer.runtime.wait_for_everyone()
        return metrics

    def _resolve_artifact_dir(self) -> Path:
        trainer = self.trainer
        eval_root = (
            trainer.run_context.require_output_dir(purpose="inline sim-eval artifacts")
            / "eval"
        )
        eval_out_dir = eval_root / "by_step" / f"step_{int(trainer.global_step)}"
        eval_out_dir.mkdir(parents=True, exist_ok=True)
        return eval_out_dir

    def _prepare_rank_markers(self, eval_out_dir: Path) -> tuple[Path, Path]:
        trainer = self.trainer
        done_marker = eval_out_dir / DONE_MARKER
        error_marker = eval_out_dir / ERROR_MARKER
        marker_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                done_marker, error_marker = clear_rank_markers(eval_out_dir)
            except BaseException as exc:
                marker_error = (
                    "Inline sim-eval marker setup failed on rank 0: "
                    f"{type(exc).__name__}: {exc}"
                )
        marker_error = trainer.runtime.broadcast_object(marker_error, src=0)
        if marker_error is not None:
            raise RuntimeError(marker_error)
        return done_marker, error_marker

    def _run_rank0_eval(
        self,
        *,
        eval_out_dir: Path,
    ) -> dict[str, float]:
        trainer = self.trainer
        with trainer._rank0_ema_eval_scope(
            backup_device=torch.device("cpu"),
        ) as eval_policy:
            eval_policy.eval()
            self._log_cuda_memory("after EMA apply")
            with torch.inference_mode():
                if trainer.sim_eval_backend != EVALUATOR_BACKEND:
                    raise RuntimeError(
                        f"Unknown sim-eval backend: {trainer.sim_eval_backend!r}"
                    )
                result = self._run_evaluator(eval_policy, eval_out_dir=eval_out_dir)

            if result is None:
                return {}
            overall = dict(result.overall)
            self._log_videos(overall.get("video_paths", []))
            return {
                "success_rate": float(overall.get("success_rate", 0.0)),
                "avg_episode_length": float(overall.get("avg_episode_length", 0.0)),
            }

    def _offload_rank0_optimizer_state_for_eval(
        self,
    ) -> eval_memory.OptimizerStateCpuOffload:
        trainer = self.trainer
        device = getattr(trainer, "device", torch.device("cpu"))
        self._log_cuda_memory("before optimizer-state offload")
        offload = eval_memory.offload_optimizer_state_to_cpu(
            getattr(trainer, "optimizer", None),
        )
        if offload.moved_tensors:
            LOGGER.info(
                "Inline sim-eval offloaded optimizer state from CUDA to CPU: "
                "tensors=%d bytes=%.2f GiB.",
                offload.moved_tensors,
                offload.moved_gib,
            )
        if torch.device(device).type == "cuda":
            eval_memory.release_cuda_eval_memory(device)
        self._log_cuda_memory("after optimizer-state offload")
        return offload

    def _restore_rank0_optimizer_state(
        self,
        offload: eval_memory.OptimizerStateCpuOffload | None,
    ) -> BaseException | None:
        if offload is None:
            return None
        try:
            offload.restore()
            self._log_cuda_memory("after optimizer-state restore")
        except BaseException as exc:
            return exc
        return None

    def _log_cuda_memory(self, label: str) -> None:
        snapshot = eval_memory.cuda_memory_snapshot(
            getattr(self.trainer, "device", torch.device("cpu"))
        )
        if snapshot is None:
            return
        LOGGER.info(
            "Inline sim-eval CUDA memory %s: %s",
            label,
            snapshot.format_gib(),
        )

    def _run_evaluator(
        self,
        eval_policy: Any,
        *,
        eval_out_dir: Path,
    ) -> praxis_eval.EvalResult:
        trainer = self.trainer
        env_type = env_type_from_cfg(trainer.cfg.env)
        adapter = PolicyEvalAdapter(
            eval_policy,
            device=trainer.device,
            preprocessor=trainer.model_preprocessor,
        )
        return praxis_eval.evaluate(
            env_type,
            policy=adapter,
            config=praxis_eval.EvalConfig(
                num_eval_per_task=trainer.sim_eval_num_eval_per_task,
                output_dir=eval_out_dir,
                task=optional_env_task(trainer.cfg.env),
                task_ids=optional_env_task_ids(trainer.cfg.env),
                num_parallel_env=trainer.sim_eval_num_parallel_env,
                seed=int(trainer.cfg.seed),
                record_episodes_per_task=trainer.sim_eval_record_episodes_per_task,
                step_timeout_sec=trainer.sim_eval_step_timeout_sec,
                rollout_failure_retries=trainer.sim_eval_rollout_failure_retries,
                policy_kwargs=trainer.sim_eval_policy_kwargs,
                env_kwargs=env_kwargs_without_type_task(trainer.cfg.env),
                metadata={
                    "step": int(trainer.global_step),
                    "mode": SIM_EVAL_MODE,
                },
                runtime_hooks=praxis_eval.EvalRuntimeHooks(
                    phase_heartbeat=lambda _label: None,
                    progress_heartbeat=lambda _label: None,
                ),
            ),
        )

    def _log_videos(self, video_paths: Any) -> None:
        trainer = self.trainer
        paths = list(video_paths)
        if trainer.wandb_upload_eval_videos and paths:
            trainer.logger.log_videos(
                paths[: trainer.wandb_max_eval_videos],
                step=trainer.global_step,
                key="media/videos",
            )

    def _restore_train_mode(self) -> BaseException | None:
        try:
            self.trainer.model.train()
        except BaseException as exc:
            return exc
        return None

    def _release_cuda_eval_memory(self) -> BaseException | None:
        try:
            eval_memory.release_cuda_eval_memory(
                getattr(self.trainer, "device", torch.device("cpu"))
            )
        except BaseException as exc:
            return exc
        return None

    def _raise_if_rank0_error(
        self,
        context: str,
        exc: BaseException | None,
    ) -> None:
        error = rank0_error_message(context, exc) if exc is not None else None
        raise_if_rank0_error(self.trainer.runtime, error)

    def _raise_if_rank_error(
        self,
        context: str,
        exc: BaseException | None,
    ) -> None:
        local_error = (
            rank_error_message(self.trainer.runtime, context, exc) if exc else None
        )
        raise_collected_rank_errors(
            collect_rank_errors(self.trainer.runtime, local_error)
        )
