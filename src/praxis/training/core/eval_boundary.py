# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Periodic eval/checkpoint boundary orchestration for train loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from praxis.training.core.checkpoint_boundary import CheckpointBoundary
from praxis.training.core.rank_errors import rank0_error_scope
from praxis.training.core.step_logging import StepLogPayload

PeriodicEvalHook = tuple[int | None, Any, bool]


class PeriodicEvalBoundary:
    """Owns eval hook execution and checkpoint-boundary completion ordering."""

    def __init__(self, trainer: Any, *, checkpoint_every: int | None) -> None:
        self.trainer = trainer
        self.checkpoint_boundary = CheckpointBoundary(
            trainer,
            checkpoint_every=checkpoint_every,
        )

    def replay_pending_resume(self, *, hooks: list[PeriodicEvalHook]) -> None:
        trainer = self.trainer
        if not self.checkpoint_boundary.should_replay_on_resume():
            return

        if trainer.runtime.is_main_process:
            print(
                f"Replaying pending checkpoint boundary at step {trainer.global_step}."
            )
        payload = StepLogPayload(trainer)
        resume_checkpoint = getattr(trainer.run_context, "resume_checkpoint", None)
        if resume_checkpoint is None:
            raise RuntimeError("Cannot replay checkpoint phase without a resume path.")
        payload.restore_metrics(
            self.checkpoint_boundary.restore_step_log_payload(
                Path(resume_checkpoint),
            )
        )
        did_eval = self._run_due_eval_hooks(payload, hooks=hooks)
        payload.log_if_due(did_eval=did_eval)
        self._complete(
            checkpoint_due=True,
            did_eval=did_eval,
            checkpoint_dir=Path(resume_checkpoint),
        )

    def run_after_step(
        self,
        payload: StepLogPayload,
        *,
        hooks: list[PeriodicEvalHook],
        checkpoint_due: bool,
        checkpoint_dir: Path | None,
    ) -> None:
        if checkpoint_due and checkpoint_dir is not None:
            self.checkpoint_boundary.write_step_log_payload(
                checkpoint_dir,
                payload.metrics,
            )
        did_eval = self._run_due_eval_hooks(payload, hooks=hooks)
        payload.log_if_due(did_eval=did_eval)
        self._complete(
            checkpoint_due=checkpoint_due,
            did_eval=did_eval,
            checkpoint_dir=checkpoint_dir,
        )

    def mark_train_end_complete(self, checkpoint_dir: Path) -> None:
        self.trainer.runtime.wait_for_everyone()
        self.checkpoint_boundary.mark_complete(checkpoint_dir)

    def _run_due_eval_hooks(
        self,
        payload: StepLogPayload,
        *,
        hooks: list[PeriodicEvalHook],
    ) -> bool:
        trainer = self.trainer
        did_eval = False
        for every, eval_fn, consider_topk in hooks:
            if every and trainer.global_step % every == 0:
                did_eval = True
                publish_context = getattr(
                    trainer,
                    "_publish_training_context_to_model",
                    None,
                )
                if callable(publish_context):
                    publish_context()
                eval_metrics = eval_fn()
                if consider_topk:
                    trainer._update_best_checkpoint_pool(eval_metrics or {})
                with rank0_error_scope(trainer.runtime, "Eval metric payload"):
                    if trainer.runtime.is_main_process and eval_metrics:
                        payload.add_metrics(eval_metrics, default_group="eval")
        return did_eval

    def _complete(
        self,
        *,
        checkpoint_due: bool,
        did_eval: bool,
        checkpoint_dir: Path | None,
    ) -> None:
        trainer = self.trainer
        if did_eval:
            # Eval hooks can trigger rank-scoped side effects such as logging and
            # candidate checkpoint writes. Keep all ranks in lockstep before the
            # next training collective.
            trainer.runtime.wait_for_everyone()

        if checkpoint_due:
            trainer._update_recent_checkpoint_pool()
            # The latest checkpoint is already durable before eval hooks run.
            # Recent checkpoint retention is a separate candidate-pool write.
            trainer.runtime.wait_for_everyone()
            if checkpoint_dir is not None:
                self.checkpoint_boundary.mark_complete(checkpoint_dir)
