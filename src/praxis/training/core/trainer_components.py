"""Extracted services for DefaultTrainer responsibilities."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, cast

import torch
from tqdm import tqdm

from praxis.training.core.checkpoint.io import DdpCheckpointIO, FsdpCheckpointIO
from praxis.training.core.checkpoint.protocols import CheckpointServiceTrainerProtocol
from praxis.training.core.checkpoint.staging import (
    read_checkpoint_wandb_run_info as read_staged_checkpoint_wandb_run_info,
    remove_path as remove_checkpoint_path,
)
from praxis.training.core.distributed import is_distributed_padding_batch
from praxis.training.core.eval_boundary import PeriodicEvalBoundary
from praxis.training.core.rank_errors import (
    collect_rank_errors,
    raise_collected_rank_errors,
    raise_if_rank0_error,
    rank0_error_message,
    rank0_error_scope,
    rank_error_message,
)
from praxis.training.core.step_logging import (
    StepLogPayload,
    accumulate_train_info_metrics,
    reduce_train_info_metrics,
)


class TrainLoopEngine:
    """Owns train-loop execution while using DefaultTrainer facade hooks."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    @staticmethod
    def _latest_checkpoint_dir(trainer: Any) -> Path:
        return Path(trainer.tc.checkpoint_dir) / "latest"

    def run(self) -> None:
        trainer = self.trainer
        if not trainer._should_run_training_loop():
            trainer._run_without_training_loop()
            return

        if trainer.runtime.is_main_process:
            sentinel = trainer.run_context.completion_sentinel_path()
        else:
            sentinel = None
        if sentinel is not None:
            sentinel.unlink(missing_ok=True)

        trainer.model.train()
        data_iter = iter(trainer.dataloader)
        checkpoint_every = (
            trainer.tc.val_every
            if trainer.tc.checkpoint_every is None
            else trainer.tc.checkpoint_every
        )
        eval_boundary = PeriodicEvalBoundary(
            trainer,
            checkpoint_every=checkpoint_every,
        )
        loss_info = torch.zeros(2, device=trainer.device)
        train_info_sums: dict[str, float] = {}
        train_info_weights: dict[str, float] = {}
        pending_optimizer_step_samples = 0

        pbar = tqdm(
            total=trainer.tc.max_steps,
            initial=trainer.global_step,
            desc=trainer._training_description(),
            disable=not trainer.runtime.is_main_process,
        )
        eval_boundary.replay_pending_resume(hooks=trainer._periodic_eval_hooks())
        while trainer.global_step < trainer.tc.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainer.dataloader)
                batch = next(data_iter)

            batch_size = trainer._batch_size_from_batch(batch)
            loss, metrics, did_optimizer_step = trainer.train_step(batch)
            pending_optimizer_step_samples += batch_size * trainer.runtime.num_processes
            loss_info[0] += loss * batch_size
            loss_info[1] += batch_size
            accumulate_train_info_metrics(
                train_info_sums,
                train_info_weights,
                metrics,
                weight=batch_size,
            )

            if not did_optimizer_step:
                continue

            trainer.global_step += 1
            trainer._on_step_complete(samples=pending_optimizer_step_samples)
            pending_optimizer_step_samples = 0
            pbar.update(1)
            step_log_payload = StepLogPayload(trainer)

            if trainer.global_step % trainer.tc.log_every == 0:
                reduced = trainer.runtime.reduce(loss_info, reduction="sum")
                reduced_train_info = reduce_train_info_metrics(
                    trainer.runtime,
                    train_info_sums,
                    train_info_weights,
                )
                with rank0_error_scope(trainer.runtime, "Train metric payload"):
                    if trainer.runtime.is_main_process:
                        avg_loss = (reduced[0] / reduced[1]).item()
                        lr = trainer._get_current_lr()
                        trainer._on_train_log(avg_loss, lr)
                        step_log_payload.add_metrics(
                            trainer._train_log_payload(avg_loss, lr),
                            default_group="train",
                        )
                        step_log_payload.add_train_info_metrics(
                            reduced_train_info,
                        )
                        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                loss_info.zero_()
                train_info_sums.clear()
                train_info_weights.clear()

            checkpoint_due = bool(
                checkpoint_every and trainer.global_step % checkpoint_every == 0
            )
            if checkpoint_due:
                trainer.save_latest_checkpoint()

            eval_boundary.run_after_step(
                step_log_payload,
                hooks=trainer._periodic_eval_hooks(),
                checkpoint_due=checkpoint_due,
                checkpoint_dir=self._latest_checkpoint_dir(trainer),
            )

        trainer.save_latest_checkpoint()
        trainer._update_recent_checkpoint_pool()
        # Match the periodic checkpoint path before train-end teardown.
        eval_boundary.mark_train_end_complete(self._latest_checkpoint_dir(trainer))
        trainer._on_train_end()


class EvalHooksService:
    """Owns eval hooks, validation, and train-end signaling."""

    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer

    def _collect_rank_errors(self, error: str | None) -> list[str]:
        return collect_rank_errors(self.trainer.runtime, error)

    def _rank_error_message(self, context: str, exc: BaseException) -> str:
        return rank_error_message(self.trainer.runtime, context, exc)

    @staticmethod
    def _raise_collected_rank_errors(errors: list[str]) -> None:
        raise_collected_rank_errors(errors)

    @staticmethod
    def _rank0_error_message(context: str, exc: BaseException) -> str:
        return rank0_error_message(context, exc)

    def _raise_if_rank0_error(self, error: str | None) -> None:
        raise_if_rank0_error(self.trainer.runtime, error)

    def periodic_eval_hooks(self) -> list[tuple[int | None, Any, bool]]:
        return [(self.trainer.tc.val_every, self.trainer.validate, True)]

    def eval_progress(self, iterable, *, desc: str, total: int | None = None):
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=False,
            disable=not self.trainer.runtime.is_main_process,
        )

    def on_train_end(self) -> None:
        trainer = self.trainer
        rank0_error = None
        if trainer.runtime.is_main_process:
            try:
                sentinel = trainer.run_context.completion_sentinel_path()
                if sentinel is not None:
                    sentinel.parent.mkdir(parents=True, exist_ok=True)
                    sentinel.write_text("")
                trainer.logger.finish()
                print(f"Training complete. {trainer.global_step} steps.")
            except BaseException as exc:
                rank0_error = self._rank0_error_message("Train-end finalization", exc)
        self._raise_if_rank0_error(rank0_error)
        trainer.runtime.end_training()

    def validate(self) -> dict[str, float]:
        trainer = self.trainer
        if trainer.val_dataloader is None:
            return {}
        trainer.model.eval()
        loss_info = torch.zeros(2, device=trainer.device)
        total_batches = len(trainer.val_dataloader)
        if trainer.max_val_batches:
            total_batches = min(total_batches, int(trainer.max_val_batches))
        val_iter = self.eval_progress(
            enumerate(trainer.val_dataloader),
            desc="Validation",
            total=total_batches,
        )
        with torch.inference_mode():
            for i, batch in val_iter:
                if trainer.max_val_batches and i >= trainer.max_val_batches:
                    break
                is_padding = is_distributed_padding_batch(trainer.val_dataloader, i)
                batch = trainer._move_batch_tensors_to_device(batch, phase="val")
                batch = trainer.preprocess_batch(batch)
                with trainer.runtime.autocast():
                    loss, _ = trainer.model(batch)
                batch_size = trainer._batch_size_from_batch(batch)
                metric_weight = 0 if is_padding else batch_size
                loss_info[0] += loss.detach() * metric_weight
                loss_info[1] += metric_weight
        trainer.model.train()

        loss_info = trainer.runtime.reduce(loss_info, reduction="sum")
        if loss_info[1] == 0:
            return {}
        return {"val_loss": (loss_info[0] / loss_info[1]).item()}

    @contextlib.contextmanager
    def ema_eval_scope(self):
        trainer = self.trainer
        unwrapped = trainer.runtime.unwrap_model(trainer.model)
        ema_applied = False
        ema = trainer.ema
        if ema is None:
            yield unwrapped
            return

        def _restore_ema(*, collect: bool = True) -> list[str]:
            restore_error = None
            if ema_applied or ema.restore_needed:
                try:
                    ema.restore(unwrapped)
                except BaseException as exc:
                    restore_error = self._rank_error_message("EMA eval restore", exc)
            if collect:
                return self._collect_rank_errors(restore_error)
            if restore_error is not None:
                raise RuntimeError(restore_error)
            return []

        apply_error = None
        try:
            ema.apply(unwrapped)
            ema_applied = True
        except BaseException as exc:
            apply_error = self._rank_error_message("EMA eval apply", exc)
        errors = self._collect_rank_errors(apply_error)
        if errors:
            errors.extend(_restore_ema())
            self._raise_collected_rank_errors(errors)
        try:
            yield unwrapped
        except BaseException:
            _restore_ema(collect=False)
            ema_applied = False
            raise
        finally:
            if ema_applied or ema.restore_needed:
                self._raise_collected_rank_errors(_restore_ema())


class CheckpointService:
    """Owns checkpoint save/load and candidate pool update behavior."""

    def __init__(self, trainer: object) -> None:
        # DefaultTrainer is hook-populated; keep the checkpoint service boundary typed.
        self.trainer = cast(CheckpointServiceTrainerProtocol, trainer)

    @staticmethod
    def read_checkpoint_wandb_run_info(
        checkpoint_dir: str | Path,
    ) -> dict[str, str | None] | None:
        return read_staged_checkpoint_wandb_run_info(checkpoint_dir)

    def update_best_checkpoint_pool(self, metrics: dict[str, float]) -> None:
        trainer = self.trainer
        if trainer.topk_manager is None:
            return

        if self._uses_distributed_checkpointing():
            decision: dict[str, Any] | None = None
            if trainer.runtime.is_main_process:
                path = trainer.topk_manager.candidate_path_if_qualified(
                    trainer.global_step,
                    metrics,
                )
                value = metrics.get(trainer.topk_manager.metric_key)
                if path is not None and isinstance(value, (int, float)):
                    decision = {"path": str(path), "value": float(value)}
            decision = trainer.runtime.broadcast_object(decision, src=0)
            if decision is None:
                return
            trainer._save_checkpoint_dir(
                Path(str(decision["path"])),
                apply_ema_weights=getattr(trainer, "ema", None) is not None,
                include_training_state=False,
            )
            rank0_error = None
            if trainer.runtime.is_main_process:
                try:
                    trainer.topk_manager.record_saved_checkpoint(
                        trainer.global_step,
                        float(decision["value"]),
                    )
                except BaseException as exc:
                    rank0_error = self._rank0_error_message(
                        "FSDP best checkpoint manifest update",
                        exc,
                    )
            self._raise_if_rank0_error(rank0_error)
            return

        def _save_fn(path: Path) -> None:
            trainer._save_checkpoint_dir(
                path,
                apply_ema_weights=getattr(trainer, "ema", None) is not None,
                include_training_state=False,
            )

        if trainer.runtime.is_main_process:
            error = None
            try:
                trainer.topk_manager.consider(trainer.global_step, metrics, _save_fn)
            except BaseException as exc:
                error = self._rank0_error_message(
                    "DDP best checkpoint update",
                    exc,
                )
            self._raise_if_rank0_error(error)
        else:
            self._raise_if_rank0_error(None)

    def update_recent_checkpoint_pool(self) -> None:
        trainer = self.trainer
        if trainer.recent_manager is None:
            return

        if self._uses_distributed_checkpointing():
            path = trainer.recent_manager.checkpoint_path(trainer.global_step)
            trainer._save_checkpoint_dir(
                path,
                apply_ema_weights=getattr(trainer, "ema", None) is not None,
                include_training_state=False,
            )
            rank0_error = None
            if trainer.runtime.is_main_process:
                try:
                    trainer.recent_manager.record_saved_checkpoint(trainer.global_step)
                except BaseException as exc:
                    rank0_error = self._rank0_error_message(
                        "FSDP recent checkpoint manifest update",
                        exc,
                    )
            self._raise_if_rank0_error(rank0_error)
            return

        def _save_fn(path: Path) -> None:
            trainer._save_checkpoint_dir(
                path,
                apply_ema_weights=getattr(trainer, "ema", None) is not None,
                include_training_state=False,
            )

        if trainer.runtime.is_main_process:
            error = None
            try:
                trainer.recent_manager.consider(trainer.global_step, _save_fn)
            except BaseException as exc:
                error = self._rank0_error_message(
                    "DDP recent checkpoint update",
                    exc,
                )
            self._raise_if_rank0_error(error)
        else:
            self._raise_if_rank0_error(None)

    def save_checkpoint(self, path: str | Path) -> None:
        trainer = self.trainer
        trainer.runtime.wait_for_everyone()
        if self._uses_distributed_checkpointing() or trainer.runtime.is_main_process:
            error = None
            try:
                trainer._save_checkpoint_dir(Path(path))
            except BaseException as exc:
                if self._uses_distributed_checkpointing():
                    raise
                error = self._rank0_error_message("DDP checkpoint save", exc)
            if not self._uses_distributed_checkpointing():
                self._raise_if_rank0_error(error)
        else:
            self._raise_if_rank0_error(None)
        trainer.runtime.wait_for_everyone()

    def save_latest_checkpoint(self) -> None:
        trainer = self.trainer
        trainer.runtime.wait_for_everyone()
        latest_dir = Path(trainer.tc.checkpoint_dir) / "latest"
        if self._uses_distributed_checkpointing() or trainer.runtime.is_main_process:
            error = None
            try:
                trainer._save_checkpoint_dir(latest_dir)
            except BaseException as exc:
                if self._uses_distributed_checkpointing():
                    raise
                error = self._rank0_error_message("DDP latest checkpoint save", exc)
            if not self._uses_distributed_checkpointing():
                self._raise_if_rank0_error(error)
        else:
            self._raise_if_rank0_error(None)
        trainer.runtime.wait_for_everyone()

    def _uses_distributed_checkpointing(self) -> bool:
        return str(self.trainer.tc.distributed_strategy) == "fsdp"

    def _checkpoint_io(self) -> DdpCheckpointIO | FsdpCheckpointIO:
        if self._uses_distributed_checkpointing():
            return FsdpCheckpointIO(self.trainer)
        return DdpCheckpointIO(self.trainer)

    def _raise_if_rank0_error(self, error: str | None) -> None:
        raise_if_rank0_error(self.trainer.runtime, error)

    def _collect_rank_errors(self, error: str | None) -> list[str]:
        return collect_rank_errors(self.trainer.runtime, error)

    @staticmethod
    def _raise_collected_rank_errors(errors: list[str]) -> None:
        raise_collected_rank_errors(errors)

    @staticmethod
    def _rank0_error_message(context: str, exc: BaseException) -> str:
        return rank0_error_message(context, exc)

    def _rank_error_message(self, context: str, exc: BaseException) -> str:
        return rank_error_message(self.trainer.runtime, context, exc)

    def save_checkpoint_dir(
        self,
        checkpoint_dir: Path,
        *,
        apply_ema_weights: bool = False,
        include_training_state: bool = True,
    ) -> None:
        self._checkpoint_io().save_checkpoint_dir(
            checkpoint_dir,
            apply_ema_weights=apply_ema_weights,
            include_training_state=include_training_state,
        )

    @staticmethod
    def remove_path(path: Path) -> None:
        remove_checkpoint_path(path)

    def load_checkpoint(self, path: str | Path) -> None:
        self._checkpoint_io().load_checkpoint(path)
