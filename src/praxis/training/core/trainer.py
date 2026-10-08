# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Hook-based default trainer with PyTorch-native distributed integration."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from praxis.core.config import config_to_dict
from praxis.core.types import PolicyIO
from praxis.training.core.checkpoint import (
    RecentCheckpointManager,
    TopKCheckpointManager,
)
from praxis.training.core.distributed import (
    DistributedRuntimeConfig,
    TrainingRuntime,
    resolve_fsdp_reshard_after_forward,
)
from praxis.training.core.ema import EMAModel
from praxis.training.core.episode_splits import split_train_val_episodes
from praxis.training.core.logging import Logger
from praxis.training.core.lr_schedulers import (
    get_constant_warmup_scheduler,
    get_cosine_warmup_scheduler,
    get_group_cosine_warmup_scheduler,
    get_group_cosine_warmup_scheduler_with_floor,
)
from praxis.training.core.metrics import MetricsTracker
from praxis.training.core.optim import (
    adapt_optimizer_param_groups,
    optimizer_default_lr,
    resolve_trainer_lr_config,
)
from praxis.training.core.run_context import (
    RunDirectoryManager,
    resolve_trainer_run_context,
)
from praxis.training.core.trainer_components import (
    CheckpointService,
    EvalHooksService,
    TrainLoopEngine,
)
from praxis.training.core.training_context import TrainingContext

logger = logging.getLogger(__name__)

BatchPreprocessor = Callable[[dict], dict]
PeriodicEvalHook = tuple[int | None, Callable[[], dict[str, float]], bool]


def _identity_batch_preprocessor(batch: dict) -> dict:
    return batch


def _checkpoint_pool_size(value: Any, *, name: str) -> int:
    """Resolve checkpoint candidate pool size; non-positive values disable it."""
    k = int(value)
    if k <= 0:
        logger.warning("%s=%d disables that checkpoint candidate pool.", name, k)
        return 0
    return k


def resolve_sampler_frame_drops(
    io: PolicyIO,
    *,
    configured_drop_n_first_frames: int | None = None,
    configured_drop_n_last_frames: int | None = None,
) -> tuple[int, int]:
    """Resolve sampler boundary drops with IO-derived minimums.

    Minimums are:
    - first: ``max(0, n_obs_steps - 1)``
    - last: ``max(0, n_action_steps - 1)``

    Optional configured values can increase these minimums.
    """
    min_first = max(0, int(io.n_obs_steps) - 1)
    min_last = max(0, int(io.n_action_steps) - 1)

    if configured_drop_n_first_frames is None:
        final_first = min_first
    else:
        cfg_first = int(configured_drop_n_first_frames)
        if cfg_first < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {cfg_first}.")
        final_first = max(min_first, cfg_first)

    if configured_drop_n_last_frames is None:
        final_last = min_last
    else:
        cfg_last = int(configured_drop_n_last_frames)
        if cfg_last < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {cfg_last}.")
        final_last = max(min_last, cfg_last)

    return final_first, final_last


def resolve_val_batch_size(
    train_batch_size: int,
    configured_val_batch_size: int | None,
) -> int:
    """Resolve validation batch size with null -> train batch size fallback."""
    resolved = (
        train_batch_size
        if configured_val_batch_size is None
        else int(configured_val_batch_size)
    )
    if resolved <= 0:
        raise ValueError(
            f"Validation batch size must be > 0, got {resolved} "
            f"(train_batch_size={train_batch_size}, configured_val_batch_size={configured_val_batch_size})."
        )
    return int(resolved)


def resolve_per_rank_batch_size(
    *,
    configured_batch_size: int,
    global_batch_size: int | None,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Resolve the DataLoader batch size owned by each distributed rank."""
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, got {world_size}.")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be > 0, "
            f"got {gradient_accumulation_steps}."
        )
    if global_batch_size is None:
        resolved = int(configured_batch_size)
        if resolved <= 0:
            raise ValueError(f"trainer.batch_size must be > 0, got {resolved}.")
        return resolved

    global_batch = int(global_batch_size)
    if global_batch <= 0:
        raise ValueError(f"trainer.global_batch_size must be > 0, got {global_batch}.")
    divisor = int(world_size) * int(gradient_accumulation_steps)
    if global_batch % divisor != 0:
        raise ValueError(
            "trainer.global_batch_size must be divisible by "
            "world_size * trainer.gradient_accumulation_steps; got "
            f"global_batch_size={global_batch}, world_size={world_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}."
        )
    resolved = global_batch // divisor
    if resolved <= 0:
        raise ValueError(
            "trainer.global_batch_size resolves to a non-positive per-rank "
            f"batch size: {resolved}."
        )
    return resolved


class DefaultTrainer:
    """Model-agnostic trainer lifecycle with hook-based specialization."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.tc = cfg.trainer
        self.action_key: str | None = None

        self._validate_distributed_config()
        self.runtime = TrainingRuntime(
            cfg=DistributedRuntimeConfig(
                strategy=str(self.tc.distributed_strategy),
                mixed_precision=str(self.tc.mixed_precision),
                gradient_accumulation_steps=int(self.tc.gradient_accumulation_steps),
                find_unused_parameters=bool(self.tc.find_unused_parameters),
                fsdp_min_num_params=int(self.tc.fsdp_min_num_params),
                fsdp_reshard_after_forward=resolve_fsdp_reshard_after_forward(
                    self.tc.fsdp_reshard_after_forward
                ),
                fsdp_cpu_offload=bool(self.tc.fsdp_cpu_offload),
                data_seed=int(cfg.seed),
            ),
            requested_device=cfg.device,
        )
        self.device = self.runtime.device
        global_batch_size = OmegaConf.select(self.tc, "global_batch_size", default=None)
        resolved_global_batch_size = (
            None if global_batch_size is None else int(global_batch_size)
        )
        if resolved_global_batch_size is not None:
            OmegaConf.update(
                self.tc,
                "global_batch_size",
                resolved_global_batch_size,
                merge=False,
            )
        self.tc.batch_size = resolve_per_rank_batch_size(
            configured_batch_size=int(self.tc.batch_size),
            global_batch_size=resolved_global_batch_size,
            world_size=self.runtime.num_processes,
            gradient_accumulation_steps=int(self.tc.gradient_accumulation_steps),
        )
        torch.manual_seed(cfg.seed)
        self.run_context = resolve_trainer_run_context(cfg)
        self._resume_checkpoint = self.run_context.resume_checkpoint
        RunDirectoryManager().prepare(
            context=self.run_context,
            cfg=cfg,
            runtime=self.runtime,
        )
        self.val_batch_size = resolve_val_batch_size(
            self.tc.batch_size, self.tc.val_batch_size
        )

        self._init_data()
        self._init_model_with_download_sync()
        self._validate_model_for_distributed_strategy()
        self.model_preprocessor = self._build_model_preprocessor()
        pre_wrap_optim_params = self._get_optim_params()
        self.run_train_loop = self._resolve_run_train_loop(pre_wrap_optim_params)
        self.ema = None
        if self.tc.use_ema and self._use_ema():
            if str(self.tc.distributed_strategy) != "fsdp":
                self.model = self.model.to(self.device)
            self.ema = EMAModel(
                self.model,
                decay=self.tc.ema_decay,
                shadow_dtype=self._ema_shadow_dtype(),
            )
        self.model = self.runtime.prepare_model(
            self.model,
            wrap_for_training=self.run_train_loop,
        )
        if self.ema is not None:
            self.ema.refresh(self.runtime.unwrap_model(self.model))
        self.dataloader = self.runtime.prepare_dataloader(self.dataloader)
        if self.val_dataloader is not None:
            self.val_dataloader = self.runtime.prepare_dataloader(self.val_dataloader)

        optim_params = self._get_optim_params()

        if self.run_train_loop:
            self.optimizer = self._build_optimizer(optim_params)
            self.scheduler = self._build_scheduler(self.optimizer)
        else:
            self.optimizer = None
            self.scheduler = None
            notice = self._non_trainable_message()
            if notice and self.runtime.is_main_process:
                print(notice)

        is_main = self.runtime.is_main_process
        wandb_config: dict[str, Any] | None = None
        if is_main and self.tc.use_wandb:
            raw_cfg = OmegaConf.to_container(cfg, resolve=False)
            wandb_config = (
                cast(dict[str, Any], raw_cfg) if isinstance(raw_cfg, dict) else None
            )

        wandb_project = str(self.tc.wandb_project or "praxis")
        wandb_entity = self.tc.wandb_entity
        wandb_resume_id: str | None = None
        if is_main and self.tc.use_wandb and self._resume_checkpoint:
            resumed_info = CheckpointService.read_checkpoint_wandb_run_info(
                self._resume_checkpoint
            )
            if resumed_info is not None:
                wandb_resume_id = resumed_info["id"]
                if resumed_info.get("project"):
                    wandb_project = str(resumed_info["project"])
                if resumed_info.get("entity"):
                    wandb_entity = str(resumed_info["entity"])

        self.logger = Logger(
            use_wandb=is_main and self.tc.use_wandb,
            project=wandb_project,
            entity=wandb_entity,
            config=wandb_config,
            run_name=str(cfg.run_name) if (is_main and self.tc.use_wandb) else None,
            run_dir=str(self.run_context.output_dir)
            if (is_main and self.tc.use_wandb and self.run_context.output_dir)
            else None,
            local_log_path=self.tc.local_log_path if is_main else None,
            resume_id=wandb_resume_id,
            wandb_init_timeout=OmegaConf.select(
                self.tc, "wandb_init_timeout", default=None
            ),
        )

        self.train_metrics = self._build_train_metrics()
        self._configure_checkpoint_candidates_before_managers()
        candidate_cfg = self.tc.checkpoint_candidates
        topk_cfg = candidate_cfg.best
        recent_cfg = candidate_cfg.recent
        checkpoint_root = Path(str(self.tc.checkpoint_dir))
        if self._enable_checkpoint_managers():
            manage_candidate_files = self.runtime.is_main_process
            topk_k = _checkpoint_pool_size(
                topk_cfg.k,
                name="checkpoint_candidates.best.k",
            )
            recent_k = _checkpoint_pool_size(
                recent_cfg.k,
                name="checkpoint_candidates.recent.k",
            )
            self.topk_manager: TopKCheckpointManager | None = (
                TopKCheckpointManager(
                    k=topk_k,
                    mode=topk_cfg.mode,
                    metric_key=topk_cfg.metric_key,
                    directory=checkpoint_root / "best",
                    manage_files=manage_candidate_files,
                )
                if topk_k > 0
                else None
            )
            self.recent_manager: RecentCheckpointManager | None = (
                RecentCheckpointManager(
                    k=recent_k,
                    directory=checkpoint_root / "recent",
                    manage_files=manage_candidate_files,
                )
                if recent_k > 0
                else None
            )
        else:
            self.topk_manager = None
            self.recent_manager = None

        self.global_step = 0
        self.max_grad_norm = self.tc.max_grad_norm
        self.max_val_batches = self.tc.max_val_batches
        self.train_loop_engine = TrainLoopEngine(self)
        self.eval_hooks = EvalHooksService(self)
        self.checkpoint_service = CheckpointService(self)

        if self._resume_checkpoint:
            self.load_checkpoint(self._resume_checkpoint)
            if is_main:
                print(
                    f"Resumed from checkpoint: {self._resume_checkpoint} "
                    f"(step {self.global_step})"
                )

    def _validate_distributed_config(self) -> None:
        strategy = str(self.tc.distributed_strategy)
        if strategy not in {"ddp", "fsdp"}:
            raise ValueError(
                "trainer.distributed_strategy must be 'ddp' or 'fsdp', "
                f"got {strategy!r}."
            )

    def _train_loop_engine(self) -> TrainLoopEngine:
        engine = getattr(self, "train_loop_engine", None)
        if engine is None:
            engine = TrainLoopEngine(self)
            self.train_loop_engine = engine
        return cast(TrainLoopEngine, engine)

    def _eval_hooks_service(self) -> EvalHooksService:
        service = getattr(self, "eval_hooks", None)
        if service is None:
            service = EvalHooksService(self)
            self.eval_hooks = service
        return cast(EvalHooksService, service)

    def _checkpoint_service(self) -> CheckpointService:
        service = getattr(self, "checkpoint_service", None)
        if service is None:
            service = CheckpointService(self)
            self.checkpoint_service = service
        return cast(CheckpointService, service)

    # ------------------------------------------------------------------
    # Initialization hooks
    # ------------------------------------------------------------------

    def _init_data(self) -> None:
        raise NotImplementedError

    def _init_model(self) -> None:
        raise NotImplementedError

    def _validate_model_for_distributed_strategy(self) -> None:
        return

    def _configure_checkpoint_candidates_before_managers(self) -> None:
        """Adjust checkpoint-candidate config before managers load manifests."""
        return

    def _init_model_with_download_sync(self) -> None:
        """Serialize first-touch model/HF cache setup across distributed ranks."""
        main_process_first = getattr(self.runtime, "main_process_first", None)
        if callable(main_process_first):
            with cast(contextlib.AbstractContextManager[None], main_process_first()):
                self._init_model()
            return

        is_distributed = bool(getattr(self.runtime, "num_processes", 1) > 1)
        if not is_distributed:
            self._init_model()
            return

        is_main_process = bool(getattr(self.runtime, "is_main_process", True))
        if is_main_process:
            self._init_model()
        self.runtime.wait_for_everyone()

        if not is_main_process:
            self._init_model()
        self.runtime.wait_for_everyone()

    def _dataset_init_context(self):
        """Serialize first-touch dataset/cache setup across distributed ranks."""
        return self.runtime.main_process_first()

    def _build_model_preprocessor(self) -> BatchPreprocessor:
        return _identity_batch_preprocessor

    def _required_tensor_keys(self, phase: str, batch: dict) -> set[str]:
        del phase
        return {k for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _move_batch_tensors_to_device(self, batch: dict, *, phase: str) -> dict:
        required = self._required_tensor_keys(phase, batch)
        missing = sorted(key for key in required if key not in batch)
        if missing:
            raise KeyError(
                f"Required tensor key(s) missing for phase={phase!r}: {missing}"
            )
        for key in sorted(required):
            if not isinstance(batch[key], torch.Tensor):
                raise TypeError(
                    f"Required key {key!r} for phase={phase!r} must be a tensor, "
                    f"got {type(batch[key]).__name__}."
                )
        return {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor) and k in required
            else v
            for k, v in batch.items()
        }

    def _get_optim_params(self) -> list[dict]:
        model = self.runtime.unwrap_model(self.model)
        get_optim_params = getattr(model, "get_optim_params", None)
        if callable(get_optim_params):
            return cast(list[dict], get_optim_params())
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"params": params}] if params else []

    def _publish_training_context_to_model(self) -> None:
        model = self.runtime.unwrap_model(self.model)
        set_training_context = getattr(model, "set_training_context", None)
        if callable(set_training_context):
            set_training_context(TrainingContext(global_step=int(self.global_step)))

    def _resolve_run_train_loop(self, optim_params: list[dict]) -> bool:
        return bool(optim_params)

    def _build_optimizer(self, optim_params: list[dict]) -> torch.optim.Optimizer:
        fused = self.device.type == "cuda"
        if str(self.tc.distributed_strategy) == "fsdp" and bool(
            getattr(self.tc, "fsdp_cpu_offload", False)
        ):
            fused = False
        scalar_lr, family_lrs = resolve_trainer_lr_config(self.tc)
        optimizer_groups = adapt_optimizer_param_groups(
            optim_params,
            scalar_lr=scalar_lr,
            family_lrs=family_lrs,
        )
        return torch.optim.AdamW(
            optimizer_groups,
            lr=optimizer_default_lr(scalar_lr=scalar_lr, family_lrs=family_lrs),
            weight_decay=self.tc.weight_decay,
            betas=tuple(self.tc.betas),
            eps=self.tc.eps,
            fused=fused,
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.tc.lr_scheduler == "constant":
            return get_constant_warmup_scheduler(
                optimizer,
                warmup_steps=self.tc.warmup_steps,
            )
        elif self.tc.lr_scheduler == "cosine":
            by_family = getattr(self.tc, "warmup_steps_by_family", None)
            if by_family:
                family_warmups = {str(k): int(v) for k, v in dict(by_family).items()}
                optim_params = self._get_optim_params()
                group_warmups = [
                    int(family_warmups.get(str(group.get("family", "")), self.tc.warmup_steps))
                    for group in optim_params
                ]
                return get_group_cosine_warmup_scheduler_with_floor(
                    optimizer,
                    warmup_steps=group_warmups,
                    total_steps=self.tc.max_steps,
                    min_lr_by_group=[1.0e-5] * len(optim_params),
                )
            return get_cosine_warmup_scheduler(
                optimizer,
                warmup_steps=self.tc.warmup_steps,
                total_steps=self.tc.max_steps,
                min_lr_ratio=self.tc.min_lr_ratio,
            )
        else:
            raise ValueError(f"Unsupported lr_scheduler: {self.tc.lr_scheduler}")

    def _use_ema(self) -> bool:
        return self.run_train_loop

    def _ema_shadow_dtype(self) -> torch.dtype:
        return torch.float32

    def _enable_checkpoint_managers(self) -> bool:
        return bool(self.run_train_loop)

    def _non_trainable_message(self) -> str | None:
        return None

    def _build_train_metrics(self) -> MetricsTracker:
        return MetricsTracker(
            batch_size=self.tc.batch_size,
            world_size=self.runtime.num_processes,
        )

    def _split_episodes(self, all_episodes: list[int]) -> tuple[list[int], list[int]]:
        split_seed = getattr(self.tc, "split_seed", getattr(self.cfg, "seed", 0))
        train_episodes, val_episodes = split_train_val_episodes(
            all_episodes,
            val_ratio=float(self.tc.val_ratio),
            split_seed=None if split_seed is None else int(split_seed),
        )
        logger.info(
            "Episode split: %d train / %d val from %d episodes "
            "(val_ratio=%s, split_seed=%s).",
            len(train_episodes),
            len(val_episodes),
            len(all_episodes),
            self.tc.val_ratio,
            split_seed,
        )
        return train_episodes, val_episodes

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        self._train_loop_engine().run()

    def _training_description(self) -> str:
        return "Training"

    def _batch_size_from_batch(self, batch: dict) -> int:
        if self.action_key is not None:
            value = batch.get(self.action_key)
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                return int(value.shape[0])
        raise KeyError("Unable to infer batch size: batch contains no tensor entries.")

    def _should_run_training_loop(self) -> bool:
        return self.run_train_loop

    def _run_without_training_loop(self) -> None:
        self.save_latest_checkpoint()
        self._on_train_end()

    def _on_step_complete(self, *, samples: int | None = None) -> None:
        self.train_metrics.step(samples=samples)

    def _get_current_lr(self) -> float:
        if self.optimizer is None or not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0]["lr"])

    def _on_train_log(self, avg_loss: float, lr: float) -> None:
        self.train_metrics.update("train_loss", avg_loss)
        self.train_metrics.update("train_lr", lr)

    def _train_log_payload(self, avg_loss: float, lr: float) -> dict[str, float]:
        return {
            "train_loss": avg_loss,
            "train_lr": lr,
            "train_samples": float(self.train_metrics.samples),
            "train_steps": float(self.train_metrics.steps),
        }

    @staticmethod
    def _namespace_logged_metric_key(key: str, *, default_group: str) -> str:
        if "/" in key:
            return key
        if default_group not in {"train", "eval"}:
            raise ValueError(
                f"default_group must be 'train' or 'eval', got {default_group!r}."
            )
        if key in {"success_rate", "avg_episode_length"}:
            return f"sim_eval/{key}"
        if key.startswith("train_"):
            return f"train/{key[len('train_') :]}"
        if key.startswith("val_"):
            return f"eval/{key[len('val_') :]}"
        return f"{default_group}/{key}"

    def _format_logged_metrics(
        self, metrics: dict[str, float], *, default_group: str
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, value in metrics.items():
            namespaced = self._namespace_logged_metric_key(
                key, default_group=default_group
            )
            if namespaced in out:
                raise RuntimeError(
                    f"Metric key collision after namespacing: {key!r} -> {namespaced!r}."
                )
            out[namespaced] = value
        return out

    def _periodic_eval_hooks(self) -> list[PeriodicEvalHook]:
        return cast(
            list[PeriodicEvalHook], self._eval_hooks_service().periodic_eval_hooks()
        )

    def _eval_progress(self, iterable, *, desc: str, total: int | None = None):
        return self._eval_hooks_service().eval_progress(
            iterable, desc=desc, total=total
        )

    def _on_train_end(self) -> None:
        self._eval_hooks_service().on_train_end()

    # ------------------------------------------------------------------
    # Steps / eval
    # ------------------------------------------------------------------

    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float], bool]:
        batch = self._move_batch_tensors_to_device(batch, phase="train")
        batch = self.preprocess_batch(batch)
        self._publish_training_context_to_model()

        with self.runtime.accumulate(self.model):
            with self.runtime.autocast():
                loss, info = self.model(batch)
            self.runtime.backward(loss)

            if self.optimizer is None or self.scheduler is None:
                raise RuntimeError(
                    "Optimizer/scheduler are required during training loop."
                )

            did_optimizer_step = bool(self.runtime.sync_gradients)
            if did_optimizer_step:
                if self.max_grad_norm > 0:
                    self.runtime.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.runtime.unwrap_model(self.model))
            self.runtime.mark_micro_step_complete()

        return loss.detach(), info, did_optimizer_step

    def validate(self) -> dict[str, float]:
        return self._eval_hooks_service().validate()

    def preprocess_batch(self, batch: dict) -> dict:
        return self.model_preprocessor(batch)

    @contextlib.contextmanager
    def _ema_eval_scope(self):
        """Temporarily apply EMA weights during eval and always restore afterwards."""
        with self._eval_hooks_service().ema_eval_scope() as unwrapped:
            yield unwrapped

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _update_best_checkpoint_pool(self, metrics: dict[str, float]) -> None:
        self._checkpoint_service().update_best_checkpoint_pool(metrics)

    def _update_recent_checkpoint_pool(self) -> None:
        self._checkpoint_service().update_recent_checkpoint_pool()

    def save_checkpoint(self, path: str | Path) -> None:
        self._checkpoint_service().save_checkpoint(path)

    def save_latest_checkpoint(self) -> None:
        self._checkpoint_service().save_latest_checkpoint()

    def _serialize_model_config(self, model: torch.nn.Module) -> dict[str, Any] | None:
        return config_to_dict(getattr(model, "config", None))

    def _save_checkpoint_dir(
        self,
        checkpoint_dir: Path,
        *,
        apply_ema_weights: bool = False,
        include_training_state: bool = True,
    ) -> None:
        self._checkpoint_service().save_checkpoint_dir(
            checkpoint_dir,
            apply_ema_weights=apply_ema_weights,
            include_training_state=include_training_state,
        )

    @staticmethod
    def _remove_path(path: Path) -> None:
        CheckpointService.remove_path(path)

    def load_checkpoint(self, path: str | Path) -> None:
        self._checkpoint_service().load_checkpoint(path)

    def evaluate(self) -> dict[str, float]:
        return self._eval_hooks_service().validate()
