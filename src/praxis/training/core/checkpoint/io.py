# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Strategy-specific trainer checkpoint save/load implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.distributed as dist
from lerobot.utils.train_utils import load_training_state, save_training_state
from omegaconf import OmegaConf

from praxis.core.checkpoint_artifacts import NORM_STATS_FILENAME
from praxis.core.types import NormMode
from praxis.training.core.checkpoint.pretrained import (
    filter_checkpoint_incompatible_keys,
    filter_model_export_state_dict,
    load_pretrained_model_weights,
    save_pretrained_model,
    serialize_model_config,
)
from praxis.training.core.checkpoint.protocols import (
    CheckpointEmaProtocol,
    CheckpointTrainerProtocol,
)
from praxis.training.core.checkpoint.staging import (
    publish_staged_directory,
    remove_path,
    stamp_wandb_run_info,
)
from praxis.training.core.rank_errors import (
    collect_rank_errors,
    raise_collected_rank_errors,
    raise_if_rank0_error,
    rank0_error_message,
    rank_error_message,
)


class _CheckpointIOBase:
    """Shared helpers for strategy-specific checkpoint I/O."""

    def __init__(self, trainer: CheckpointTrainerProtocol) -> None:
        self.trainer = trainer

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

    @staticmethod
    def _ema_restore_needed(ema: CheckpointEmaProtocol) -> bool:
        return bool(ema.restore_needed)

    def _serialize_model_config(self, model: torch.nn.Module) -> dict[str, Any] | None:
        return serialize_model_config(
            model,
            serializer=self.trainer._serialize_model_config,
        )


class DdpCheckpointIO(_CheckpointIOBase):
    """Save/load HF-native checkpoints for single-process and DDP training."""

    def save_checkpoint_dir(
        self,
        checkpoint_dir: Path,
        *,
        apply_ema_weights: bool = False,
        include_training_state: bool = True,
    ) -> None:
        trainer = self.trainer
        parent = checkpoint_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging_dir = parent / f".{checkpoint_dir.name}.tmp-{uuid4().hex}"
        backup_dir = parent / f".{checkpoint_dir.name}.prev"
        remove_path(staging_dir)
        remove_path(backup_dir)

        unwrapped = trainer.runtime.unwrap_model(trainer.model)
        ema = getattr(trainer, "ema", None)
        ema_applied = False

        model_dir = staging_dir / "pretrained_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            if apply_ema_weights and ema is not None:
                ema.refresh(unwrapped)
                ema.apply(unwrapped)
                ema_applied = True
            save_pretrained_model(
                unwrapped,
                model_dir,
                config=self._serialize_model_config(unwrapped),
            )
            (model_dir / "train_config.yaml").write_text(OmegaConf.to_yaml(trainer.cfg))
            if include_training_state:
                if trainer.optimizer is not None:
                    save_training_state(
                        checkpoint_dir=staging_dir,
                        train_step=trainer.global_step,
                        optimizer=trainer.optimizer,
                        scheduler=trainer.scheduler,
                    )
                if trainer.ema is not None:
                    trainer.ema.refresh(unwrapped)
                    state_dir = staging_dir / "training_state"
                    state_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(trainer.ema.state_dict(), state_dir / "ema.pt")
                stamp_wandb_run_info(trainer, staging_dir)
        except Exception:
            remove_path(staging_dir)
            raise
        finally:
            if ema is not None and (ema_applied or self._ema_restore_needed(ema)):
                ema.restore(unwrapped)

        publish_staged_directory(
            staging_dir=staging_dir,
            checkpoint_dir=checkpoint_dir,
            backup_dir=backup_dir,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        trainer = self.trainer
        checkpoint_dir = Path(path)
        unwrapped = trainer.runtime.unwrap_model(trainer.model)
        if not checkpoint_dir.is_dir():
            raise NotADirectoryError(
                f"Expected checkpoint directory, got: {checkpoint_dir}"
            )

        model_dir = checkpoint_dir / "pretrained_model"
        state_dir = checkpoint_dir / "training_state"
        missing, unexpected = load_pretrained_model_weights(
            unwrapped,
            model_dir,
            strict=False,
        )
        missing, unexpected = filter_checkpoint_incompatible_keys(
            unwrapped,
            missing=missing,
            unexpected=unexpected,
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Error(s) in loading checkpoint for {unwrapped.__class__.__name__}: "
                f"missing_keys={missing}, unexpected_keys={unexpected}"
            )

        self._load_norm_stats(unwrapped, model_dir / NORM_STATS_FILENAME)

        if trainer.optimizer is not None:
            step, loaded_optimizer, loaded_scheduler = load_training_state(
                checkpoint_dir=checkpoint_dir,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
            )
            trainer.optimizer = loaded_optimizer
            trainer.scheduler = loaded_scheduler
            trainer.global_step = step
        else:
            trainer.global_step = 0

        if trainer.ema is not None:
            ema_path = state_dir / "ema.pt"
            if state_dir.exists() and not ema_path.exists():
                raise RuntimeError(
                    "EMA is configured, but the training-state checkpoint is "
                    f"missing EMA state at {ema_path}."
                )
            if ema_path.exists():
                trainer.ema.load_state_dict(
                    torch.load(ema_path, map_location="cpu", weights_only=True)
                )

    @staticmethod
    def _load_norm_stats(model: torch.nn.Module, path: Path) -> None:
        if not path.exists():
            return
        data = torch.load(path, map_location="cpu", weights_only=True)
        stats = data.get("stats", {})
        modes_raw = data.get("modes", {})
        if not stats:
            return
        modes = {key: NormMode(value) for key, value in modes_raw.items()}
        set_fn = getattr(model, "set_normalization_stats", None)
        if callable(set_fn):
            set_fn(stats, modes=modes)


class FsdpCheckpointIO(_CheckpointIOBase):
    """Save/load DCP checkpoints and HF-native exports for FSDP training."""

    def save_checkpoint_dir(
        self,
        checkpoint_dir: Path,
        *,
        apply_ema_weights: bool = False,
        include_training_state: bool = True,
    ) -> None:
        if include_training_state:
            self._save_dcp_checkpoint_dir(checkpoint_dir)
            return
        self._save_export_checkpoint_dir(
            checkpoint_dir,
            apply_ema_weights=apply_ema_weights,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        self._load_dcp_checkpoint(path)

    def _save_dcp_checkpoint_dir(self, checkpoint_dir: Path) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
        )
        from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

        trainer = self.trainer
        parent = checkpoint_dir.parent
        parent_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except BaseException as exc:
                parent_error = self._rank0_error_message(
                    "FSDP DCP checkpoint parent setup",
                    exc,
                )
        self._raise_if_rank0_error(parent_error)
        trainer.runtime.wait_for_everyone()

        staging_token = uuid4().hex if trainer.runtime.is_main_process else None
        staging_token = trainer.runtime.broadcast_object(staging_token, src=0)
        staging_dir = parent / f".{checkpoint_dir.name}.tmp-{staging_token}"
        backup_dir = parent / f".{checkpoint_dir.name}.prev"
        staging_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                remove_path(staging_dir)
                remove_path(backup_dir)
                (staging_dir / "dcp").mkdir(parents=True, exist_ok=True)
                (staging_dir / "training_state").mkdir(parents=True, exist_ok=True)
            except BaseException as exc:
                remove_path(staging_dir)
                staging_error = self._rank0_error_message(
                    "FSDP DCP checkpoint staging setup",
                    exc,
                )
        self._raise_if_rank0_error(staging_error)
        trainer.runtime.wait_for_everyone()

        if trainer.optimizer is None:
            raise RuntimeError("FSDP checkpointing requires an optimizer.")

        model_state: dict[str, Any] = {}
        optim_state: dict[str, Any] = {}
        rank_error = None
        try:
            model_state, optim_state = get_state_dict(
                trainer.model,
                trainer.optimizer,
                options=StateDictOptions(full_state_dict=False, cpu_offload=False),
            )
        except BaseException as exc:
            rank_error = self._rank_error_message("FSDP DCP state dict", exc)
        errors = self._collect_rank_errors(rank_error)
        if errors:
            if trainer.runtime.is_main_process:
                remove_path(staging_dir)
            self._raise_collected_rank_errors(errors)

        state: dict[str, Any] = {"model": model_state, "optimizer": optim_state}
        ema = getattr(trainer, "ema", None)
        rank_error = None
        try:
            if ema is not None:
                ema.refresh(trainer.runtime.unwrap_model(trainer.model))
                state["ema"] = ema.state_dict()
        except BaseException as exc:
            rank_error = self._rank_error_message("FSDP DCP EMA state", exc)
        errors = self._collect_rank_errors(rank_error)
        if errors:
            if trainer.runtime.is_main_process:
                remove_path(staging_dir)
            self._raise_collected_rank_errors(errors)

        rank_error = None
        try:
            dcp_save(
                state,
                checkpoint_id=staging_dir / "dcp",
                no_dist=not (dist.is_available() and dist.is_initialized()),
            )
        except BaseException as exc:
            rank_error = self._rank_error_message("FSDP DCP save", exc)
        errors = self._collect_rank_errors(rank_error)
        if errors:
            if trainer.runtime.is_main_process:
                remove_path(staging_dir)
            self._raise_collected_rank_errors(errors)

        metadata_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                (staging_dir / "train_config.yaml").write_text(
                    OmegaConf.to_yaml(trainer.cfg)
                )
                torch.save(
                    {
                        "train_step": int(trainer.global_step),
                        "scheduler": None
                        if trainer.scheduler is None
                        else trainer.scheduler.state_dict(),
                    },
                    staging_dir / "training_state" / "trainer.pt",
                )
                stamp_wandb_run_info(trainer, staging_dir)
            except BaseException as exc:
                remove_path(staging_dir)
                metadata_error = self._rank0_error_message(
                    "FSDP DCP metadata write",
                    exc,
                )
        self._raise_if_rank0_error(metadata_error)

        trainer.runtime.wait_for_everyone()
        publish_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                publish_staged_directory(
                    staging_dir=staging_dir,
                    checkpoint_dir=checkpoint_dir,
                    backup_dir=backup_dir,
                )
            except BaseException as exc:
                publish_error = self._rank0_error_message(
                    "FSDP DCP checkpoint publish",
                    exc,
                )
        self._raise_if_rank0_error(publish_error)
        trainer.runtime.wait_for_everyone()

    def _save_export_checkpoint_dir(
        self,
        checkpoint_dir: Path,
        *,
        apply_ema_weights: bool = False,
    ) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        trainer = self.trainer
        parent = checkpoint_dir.parent
        parent_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except BaseException as exc:
                parent_error = self._rank0_error_message(
                    "FSDP pretrained_model parent setup",
                    exc,
                )
        self._raise_if_rank0_error(parent_error)
        trainer.runtime.wait_for_everyone()

        staging_token = uuid4().hex if trainer.runtime.is_main_process else None
        staging_token = trainer.runtime.broadcast_object(staging_token, src=0)
        staging_dir = parent / f".{checkpoint_dir.name}.tmp-{staging_token}"
        backup_dir = parent / f".{checkpoint_dir.name}.prev"
        staging_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                remove_path(staging_dir)
                remove_path(backup_dir)
                (staging_dir / "pretrained_model").mkdir(parents=True, exist_ok=True)
            except BaseException as exc:
                remove_path(staging_dir)
                staging_error = self._rank0_error_message(
                    "FSDP pretrained_model staging setup",
                    exc,
                )
        self._raise_if_rank0_error(staging_error)
        trainer.runtime.wait_for_everyone()

        unwrapped = trainer.runtime.unwrap_model(trainer.model)
        ema = getattr(trainer, "ema", None)
        ema_applied = False

        def _restore_ema() -> list[str]:
            restore_error = None
            if ema is not None and (ema_applied or self._ema_restore_needed(ema)):
                try:
                    ema.restore(unwrapped)
                except BaseException as exc:
                    restore_error = self._rank_error_message(
                        "FSDP pretrained_model EMA restore",
                        exc,
                    )
            return self._collect_rank_errors(restore_error)

        rank_error = None
        try:
            if apply_ema_weights and ema is not None:
                ema.refresh(unwrapped)
                ema.apply(unwrapped)
                ema_applied = True
        except BaseException as exc:
            rank_error = self._rank_error_message(
                "FSDP pretrained_model EMA apply",
                exc,
            )
        errors = self._collect_rank_errors(rank_error)
        if errors:
            errors.extend(_restore_ema())
            if trainer.runtime.is_main_process:
                remove_path(staging_dir)
            self._raise_collected_rank_errors(errors)

        full_state_dict: dict[str, Any] = {}
        rank_error = None
        try:
            full_state_dict = get_model_state_dict(
                trainer.model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        except BaseException as exc:
            rank_error = self._rank_error_message(
                "FSDP pretrained_model state dict",
                exc,
            )
        errors = self._collect_rank_errors(rank_error)
        if errors:
            errors.extend(_restore_ema())
            if trainer.runtime.is_main_process:
                remove_path(staging_dir)
            self._raise_collected_rank_errors(errors)

        rank_error = None
        try:
            full_state_dict = filter_model_export_state_dict(unwrapped, full_state_dict)
        except BaseException as exc:
            rank_error = self._rank_error_message(
                "FSDP pretrained_model state filtering",
                exc,
            )
        errors = self._collect_rank_errors(rank_error)
        if errors:
            errors.extend(_restore_ema())
            if trainer.runtime.is_main_process:
                remove_path(staging_dir)
            self._raise_collected_rank_errors(errors)

        export_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                model_dir = staging_dir / "pretrained_model"
                save_pretrained_model(
                    unwrapped,
                    model_dir,
                    config=self._serialize_model_config(unwrapped),
                    state_dict=full_state_dict,
                )
                (model_dir / "train_config.yaml").write_text(
                    OmegaConf.to_yaml(trainer.cfg)
                )
            except BaseException as exc:
                remove_path(staging_dir)
                export_error = self._rank0_error_message(
                    "FSDP pretrained_model export",
                    exc,
                )

        errors = _restore_ema()
        export_error = trainer.runtime.broadcast_object(export_error, src=0)
        if export_error is not None:
            errors.append(str(export_error))
        if errors:
            if trainer.runtime.is_main_process:
                remove_path(staging_dir)
            self._raise_collected_rank_errors(errors)

        trainer.runtime.wait_for_everyone()
        publish_error: str | None = None
        if trainer.runtime.is_main_process:
            try:
                publish_staged_directory(
                    staging_dir=staging_dir,
                    checkpoint_dir=checkpoint_dir,
                    backup_dir=backup_dir,
                )
            except BaseException as exc:
                publish_error = self._rank0_error_message(
                    "FSDP pretrained_model checkpoint publish",
                    exc,
                )
        self._raise_if_rank0_error(publish_error)
        trainer.runtime.wait_for_everyone()

    def _load_dcp_checkpoint(self, path: str | Path) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
            get_state_dict,
            set_model_state_dict,
            set_state_dict,
        )
        from torch.distributed.checkpoint.state_dict_loader import load as dcp_load

        trainer = self.trainer
        checkpoint_dir = Path(path)
        state_path = checkpoint_dir / "training_state" / "trainer.pt"
        rank_error = None
        local_metadata: dict[str, bool] = {}
        try:
            local_metadata = {
                "checkpoint_dir": checkpoint_dir.is_dir(),
                "dcp_dir": (checkpoint_dir / "dcp").is_dir(),
                "state_path": state_path.exists(),
                "has_optimizer": trainer.optimizer is not None,
            }
            if not local_metadata["checkpoint_dir"]:
                raise NotADirectoryError(
                    f"Expected checkpoint directory, got: {checkpoint_dir}"
                )
            if not local_metadata["dcp_dir"]:
                raise NotADirectoryError(
                    f"Expected FSDP DCP directory, got: {checkpoint_dir / 'dcp'}"
                )
        except BaseException as exc:
            rank_error = self._rank_error_message("FSDP DCP checkpoint metadata", exc)
        errors = self._collect_rank_errors(rank_error)
        if errors:
            self._raise_collected_rank_errors(errors)

        metadata = trainer.runtime.gather_object(local_metadata)
        state_visibility = {bool(item["state_path"]) for item in metadata}
        optimizer_visibility = {bool(item["has_optimizer"]) for item in metadata}
        if len(state_visibility) != 1:
            raise RuntimeError(
                "FSDP DCP checkpoint training_state visibility differs by rank: "
                f"{metadata}"
            )
        if len(optimizer_visibility) != 1:
            raise RuntimeError(
                "FSDP DCP checkpoint optimizer availability differs by rank: "
                f"{metadata}"
            )
        load_training_state = bool(metadata[0]["has_optimizer"]) and bool(
            metadata[0]["state_path"]
        )

        model_state: dict[str, Any] = {}
        optim_state: dict[str, Any] = {}
        state: dict[str, Any] = {}
        rank_error = None
        if load_training_state:
            try:
                assert trainer.optimizer is not None
                model_state, optim_state = get_state_dict(
                    trainer.model,
                    trainer.optimizer,
                    options=StateDictOptions(full_state_dict=False, cpu_offload=False),
                )
                state = {"model": model_state, "optimizer": optim_state}
            except BaseException as exc:
                rank_error = self._rank_error_message(
                    "FSDP DCP load state dict setup",
                    exc,
                )
        else:
            try:
                state = {
                    "model": get_model_state_dict(
                        trainer.model,
                        options=StateDictOptions(
                            full_state_dict=False,
                            cpu_offload=False,
                        ),
                    ),
                }
            except BaseException as exc:
                rank_error = self._rank_error_message(
                    "FSDP DCP load model state setup",
                    exc,
                )
        errors = self._collect_rank_errors(rank_error)
        if errors:
            self._raise_collected_rank_errors(errors)

        ema = getattr(trainer, "ema", None)
        ema_state = None
        rank_error = None
        try:
            if ema is not None:
                ema.refresh(trainer.runtime.unwrap_model(trainer.model))
                ema_state = ema.state_dict()
                state["ema"] = ema_state
        except BaseException as exc:
            rank_error = self._rank_error_message("FSDP DCP EMA state setup", exc)
        errors = self._collect_rank_errors(rank_error)
        if errors:
            self._raise_collected_rank_errors(errors)

        rank_error = None
        try:
            dcp_load(
                state,
                checkpoint_id=checkpoint_dir / "dcp",
                no_dist=not (dist.is_available() and dist.is_initialized()),
            )
        except BaseException as exc:
            rank_error = self._rank_error_message("FSDP DCP load", exc)
        errors = self._collect_rank_errors(rank_error)
        if errors:
            self._raise_collected_rank_errors(errors)

        rank_error = None
        if load_training_state:
            try:
                assert trainer.optimizer is not None
                set_state_dict(
                    trainer.model,
                    trainer.optimizer,
                    model_state_dict=state["model"],
                    optim_state_dict=state["optimizer"],
                    options=StateDictOptions(full_state_dict=False, cpu_offload=False),
                )
            except BaseException as exc:
                rank_error = self._rank_error_message("FSDP DCP state apply", exc)
        else:
            try:
                set_model_state_dict(
                    trainer.model,
                    state["model"],
                    options=StateDictOptions(full_state_dict=False, cpu_offload=False),
                )
            except BaseException as exc:
                rank_error = self._rank_error_message("FSDP DCP model state apply", exc)
        errors = self._collect_rank_errors(rank_error)
        if errors:
            self._raise_collected_rank_errors(errors)

        if load_training_state:
            rank0_error = None
            trainer_state = None
            if trainer.runtime.is_main_process:
                try:
                    trainer_state = torch.load(
                        state_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except BaseException as exc:
                    rank0_error = self._rank0_error_message(
                        "FSDP DCP trainer state load",
                        exc,
                    )
            self._raise_if_rank0_error(rank0_error)
            trainer_state = trainer.runtime.broadcast_object(trainer_state, src=0)
            if not isinstance(trainer_state, dict):
                raise RuntimeError(
                    "FSDP DCP trainer state must be a dict, "
                    f"got {type(trainer_state).__name__}."
                )
            rank_error = None
            try:
                trainer.global_step = int(trainer_state.get("train_step", 0))
                scheduler_state = trainer_state.get("scheduler")
                if scheduler_state is not None and trainer.scheduler is not None:
                    trainer.scheduler.load_state_dict(scheduler_state)
            except BaseException as exc:
                rank_error = self._rank_error_message(
                    "FSDP DCP trainer state apply",
                    exc,
                )
            errors = self._collect_rank_errors(rank_error)
            if errors:
                self._raise_collected_rank_errors(errors)
        else:
            trainer.global_step = 0
        if ema is not None and ema_state is not None:
            rank_error = None
            try:
                ema.load_state_dict(ema_state)
            except BaseException as exc:
                rank_error = self._rank_error_message("FSDP DCP EMA state apply", exc)
            errors = self._collect_rank_errors(rank_error)
            if errors:
                self._raise_collected_rank_errors(errors)
