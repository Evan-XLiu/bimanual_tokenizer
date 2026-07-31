# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy trainer: DefaultTrainer specialization for policy learning."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, cast

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig, OmegaConf
from praxis_eval.evaluation.config import (
    require_positive_int,
    resolve_nonnegative_int,
    resolve_optional_positive_timeout_sec,
    resolve_policy_kwargs,
)

from praxis.core.types import PolicyIO
from praxis.data.action_token_cache import ActionTokenCache, CachedActionTokenDataset
from praxis.data.lerobot_adapter import (
    LeRobotDatasetAdapter,
    ensure_lerobot_dataset_available,
)
from praxis.policies.core import BasePolicy
from praxis.policies.factory import make_io_from_metadata, make_policy
from praxis.processing.pipelines import make_policy_pre_post_processors
from praxis.training import inline_policy_eval
from praxis.training.core.distributed import is_distributed_padding_batch
from praxis.training.core.eval_collectors import (
    ReconstLossCollector,
    ReconstructionBatchContext,
    ReconstructionMetricPipeline,
    build_reconstruction_info_annotators,
    build_reconstruction_metric_collectors,
)
from praxis.training.core.eval_memory import release_cuda_eval_memory
from praxis.training.core.rank_errors import (
    raise_if_rank0_error,
    rank0_error_message,
)
from praxis.training.core.trainer import (
    DefaultTrainer,
    resolve_sampler_frame_drops,
)


def _build_reconstruction_obs(
    *,
    policy: Any,
    batch: dict[str, Any],
    fallback_io: PolicyIO | None,
) -> dict[str, Any]:
    """Build reconstruction observations from exact PolicyIO keys."""
    io = getattr(policy, "io", None) or fallback_io
    obs_keys = getattr(io, "obs_keys", None)
    if io is None or obs_keys is None:
        raise RuntimeError(
            "PolicyTrainer.reconstruct requires a PolicyIO on policy.io or trainer.io "
            "to select reconstruction observation keys."
        )

    obs_key_list = list(obs_keys)
    missing = [key for key in obs_key_list if key not in batch]
    if missing:
        available_obs = sorted(
            key
            for key in batch
            if isinstance(key, str) and key.startswith("observation.")
        )
        available_text = available_obs if available_obs else "none"
        raise KeyError(
            "Missing PolicyIO observation key(s) during reconstruction: "
            f"{missing}. Available observation keys in batch: {available_text}."
        )

    obs = {key: batch[key] for key in obs_key_list}
    language_key = getattr(io, "language_key", None)
    if isinstance(language_key, str) and language_key:
        if language_key not in batch:
            raise KeyError(
                "Missing PolicyIO language key during reconstruction: "
                f"{language_key!r}."
            )
        obs[language_key] = batch[language_key]
    return obs


class PolicyTrainer(DefaultTrainer):
    """Extends DefaultTrainer with reconstruction eval and optional sim eval."""

    def __init__(self, cfg: DictConfig) -> None:
        self._resolved_sim_eval_backend = inline_policy_eval.resolve_preinit_backend(
            cfg
        )
        self._sim_eval_topk_metric_overridden = False
        super().__init__(cfg)
        tc = cfg.trainer
        sec = tc.sim_eval

        self.reconst_every = tc.reconst_every
        self.max_reconst_batches = tc.max_reconst_batches

        self.sim_eval_every = (
            None
            if tc.sim_eval_every is None
            else require_positive_int(
                int(tc.sim_eval_every), name="trainer.sim_eval_every"
            )
        )
        sim_eval_start_after_step = OmegaConf.select(
            tc,
            "sim_eval_start_after_step",
            default=None,
        )
        self.sim_eval_start_after_step = (
            None
            if sim_eval_start_after_step is None
            else resolve_nonnegative_int(
                tc,
                key="sim_eval_start_after_step",
                label="trainer.sim_eval_start_after_step",
                default=0,
            )
        )
        self.sim_eval_num_eval_per_task = require_positive_int(
            int(sec.num_eval_per_task), name="trainer.sim_eval.num_eval_per_task"
        )
        self.sim_eval_num_parallel_env = require_positive_int(
            int(sec.num_parallel_env), name="trainer.sim_eval.num_parallel_env"
        )
        self.sim_eval_record_episodes_per_task = int(sec.record_episodes_per_task)
        if self.sim_eval_record_episodes_per_task < 0:
            raise ValueError(
                "trainer.sim_eval.record_episodes_per_task must be >= 0, "
                f"got {self.sim_eval_record_episodes_per_task}"
            )
        self.sim_eval_step_timeout_sec = resolve_optional_positive_timeout_sec(sec)
        self.sim_eval_rollout_failure_retries = resolve_nonnegative_int(
            sec,
            key="rollout_failure_retries",
            label="trainer.sim_eval.rollout_failure_retries",
            default=1,
        )
        self.sim_eval_policy_kwargs = resolve_policy_kwargs(
            sec,
            label="trainer.sim_eval.policy_kwargs",
        )
        self.wandb_upload_eval_videos = bool(tc.wandb_upload_eval_videos)
        self.wandb_max_eval_videos = int(tc.wandb_max_eval_videos)
        self.sim_eval_backend: str | None = None

        env_cfg_raw = cfg.get("env", None)
        if self.sim_eval_every:
            assert env_cfg_raw is not None
            assert self._resolved_sim_eval_backend is not None
            self.sim_eval_backend = self._resolved_sim_eval_backend

        if self._sim_eval_topk_metric_overridden and self.runtime.is_main_process:
            print(
                "Top-K metric set to success_rate (mode=max) because inline "
                "sim-eval is enabled."
            )

    # ------------------------------------------------------------------
    # DefaultTrainer hooks
    # ------------------------------------------------------------------

    def _configure_checkpoint_candidates_before_managers(self) -> None:
        super()._configure_checkpoint_candidates_before_managers()
        self._sim_eval_topk_metric_overridden = (
            inline_policy_eval.configure_success_rate_topk(
                self.cfg,
                sim_eval_backend=self._resolved_sim_eval_backend,
            )
        )

    def _init_data(self) -> None:
        dc = self.cfg.dataset
        ensure_lerobot_dataset_available(
            repo_id=dc.repo_id,
            runtime=self.runtime,
            root=dc.root,
            episodes=dc.episodes,
            download_videos=True,
            bypass_force_cache_sync=bool(getattr(dc, "bypass_force_cache_sync", False)),
        )
        with self._dataset_init_context():
            temp_meta = LeRobotDataset(
                repo_id=dc.repo_id, root=dc.root, episodes=dc.episodes
            )
            observation_keys = getattr(dc, "observation_keys", None)
            self.io = make_io_from_metadata(
                temp_meta.meta.features,
                self.cfg.policy,
                observation_keys=(
                    list(observation_keys) if observation_keys is not None else None
                ),
            )
            if self.io.action_feature is None:
                raise RuntimeError(
                    "Policy trainer requires PolicyIO.action_feature to be defined."
                )
            self.action_key = self.io.action_feature.key
            self.io.fps = temp_meta.fps
            all_episodes = list(dc.episodes or range(temp_meta.meta.total_episodes))
            del temp_meta

            train_episodes, val_episodes = self._split_episodes(all_episodes)

            self.dataset = LeRobotDatasetAdapter(
                repo_id=dc.repo_id,
                io=self.io,
                root=dc.root,
                episodes=train_episodes,
                video_backend=dc.video_backend,
            )
            token_cache_root = getattr(dc, "action_token_cache", None)
            if token_cache_root:
                self.dataset = CachedActionTokenDataset(
                    self.dataset,
                    ActionTokenCache.open(token_cache_root),
                )
            self.io.fps = self.dataset.fps

        tasks_meta = getattr(self.dataset.meta, "tasks", None)
        has_tasks = False
        if tasks_meta is not None:
            if isinstance(tasks_meta, dict) or hasattr(tasks_meta, "__len__"):
                has_tasks = len(tasks_meta) > 0
            else:
                has_tasks = bool(tasks_meta)
        if has_tasks:
            self.io.language_key = "task"

        drop_n_first_frames, drop_n_last_frames = resolve_sampler_frame_drops(
            self.io,
            configured_drop_n_first_frames=self.tc.drop_n_first_frames,
            configured_drop_n_last_frames=self.tc.drop_n_last_frames,
        )

        self.dataloader = self.dataset.make_dataloader(
            batch_size=self.tc.batch_size,
            num_workers=self.tc.num_workers,
            shuffle=True,
            prefetch_factor=self.tc.prefetch_factor,
            use_episode_aware_sampler=self.tc.use_episode_aware_sampler,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            drop_last=True,
        )

        if val_episodes:
            with self._dataset_init_context():
                self.val_dataset = LeRobotDatasetAdapter(
                    repo_id=dc.repo_id,
                    io=self.io,
                    root=dc.root,
                    episodes=val_episodes,
                    video_backend=dc.video_backend,
                )
            val_token_cache_root = getattr(dc, "val_action_token_cache", None)
            if val_token_cache_root:
                self.val_dataset = CachedActionTokenDataset(
                    self.val_dataset,
                    ActionTokenCache.open(val_token_cache_root),
                )
            self.val_dataloader = self.val_dataset.make_dataloader(
                batch_size=self.val_batch_size,
                num_workers=self.tc.num_workers,
                shuffle=False,
                prefetch_factor=self.tc.prefetch_factor,
                use_episode_aware_sampler=self.tc.use_episode_aware_sampler,
                drop_n_first_frames=drop_n_first_frames,
                drop_n_last_frames=drop_n_last_frames,
                drop_last=False,
            )
        else:
            self.val_dataset = None
            self.val_dataloader = None

    def _init_model(self) -> None:
        self.model = make_policy(self.cfg.policy, self.io)
        self.model.set_normalization_stats(
            self.dataset.stats,
            modes=getattr(self.dataset, "normalization_modes", None),
        )

    def _validate_model_for_distributed_strategy(self) -> None:
        strategy = str(self.tc.distributed_strategy)
        self._validate_ki_warmstart_ddp_contract(strategy=strategy)
        if strategy != "fsdp":
            return
        validate_strategy = getattr(self.model, "validate_distributed_strategy", None)
        if callable(validate_strategy):
            validate_strategy(strategy)
            return
        raise ValueError(
            "Policy FSDP currently requires the policy model to validate support "
            "with validate_distributed_strategy('fsdp'). Use "
            "trainer.distributed_strategy=ddp for policies without that hook."
        )

    def _validate_ki_warmstart_ddp_contract(self, *, strategy: str) -> None:
        if strategy != "ddp":
            return
        if bool(getattr(self.tc, "find_unused_parameters", False)):
            return
        warmstart_steps = int(
            getattr(getattr(self.model, "config", None), "ce_warmstart_steps", 0)
        )
        if warmstart_steps <= 0:
            return
        raise ValueError(
            "KI CE warmstart disables the flow/action-expert branch before "
            f"step {warmstart_steps}, which leaves action-expert parameters unused "
            "under DDP. Set trainer.find_unused_parameters=true or disable "
            "policy.ce_warmstart_steps."
        )

    def _build_model_preprocessor(self):
        pre, post = make_policy_pre_post_processors()
        self.policy_postprocessor = post
        return pre

    def _required_tensor_keys(self, phase: str, batch: dict) -> set[str]:
        del phase
        required: set[str] = set(self.io.obs_keys)
        if self.action_key is None:
            raise RuntimeError("action_key must be initialized before batch validation")
        required.add(self.action_key)
        for key in list(required):
            pad_key = f"{key}_is_pad"
            if pad_key in batch:
                required.add(pad_key)
        language_key = self.io.language_key
        if isinstance(language_key, str) and isinstance(
            batch.get(language_key), torch.Tensor
        ):
            required.add(language_key)
        target_key = getattr(
            getattr(self.model, "config", None), "action_token_target_key", None
        )
        if isinstance(target_key, str):
            required.add(target_key)
        length_key = getattr(
            getattr(self.model, "config", None), "action_token_length_key", None
        )
        if isinstance(length_key, str):
            required.add(length_key)
        return required

    def _periodic_eval_hooks(self):
        sim_eval_every = self.sim_eval_every
        if (
            sim_eval_every is not None
            and self.sim_eval_start_after_step is not None
            and int(self.sim_eval_start_after_step) > 0
            and int(getattr(self, "global_step", 0))
            <= int(self.sim_eval_start_after_step)
        ):
            sim_eval_every = None
        return [
            (self.tc.val_every, self.validate, False),
            (self.reconst_every, self.reconstruct, True),
            (sim_eval_every, self.sim_eval, True),
        ]

    def _raise_if_rank0_error(self, context: str, exc: BaseException | None) -> None:
        error = rank0_error_message(context, exc) if exc is not None else None
        raise_if_rank0_error(self.runtime, error)

    @contextlib.contextmanager
    def _rank0_ema_eval_scope(
        self,
        *,
        backup_device: torch.device | str | None = None,
    ) -> Iterator[BasePolicy]:
        """Apply EMA for rank-0-only inline sim eval without distributed collectives."""
        unwrapped = self.runtime.unwrap_model(self.model)
        policy = cast(BasePolicy, unwrapped)
        ema = getattr(self, "ema", None)
        if ema is None:
            yield policy
            return

        ema_applied = False
        try:
            ema.apply(policy, backup_device=backup_device)
            ema_applied = True
        except BaseException:
            if ema.restore_needed:
                ema.restore(policy)
            raise

        try:
            yield policy
        finally:
            if ema_applied or ema.restore_needed:
                ema.restore(policy)

    def _on_train_end(self) -> None:
        super()._on_train_end()

    # ------------------------------------------------------------------
    # Policy-specific eval hooks
    # ------------------------------------------------------------------

    def reconstruct(self) -> dict[str, float]:
        if self.val_dataloader is None:
            return {}
        if self.action_key is None:
            raise RuntimeError("action_key must be initialized before reconstruct")
        action_key = self.action_key

        warned_missing_chunk_api = False
        total_batches = len(self.val_dataloader)
        if self.max_reconst_batches:
            total_batches = min(total_batches, int(self.max_reconst_batches))
        reconst_iter = self._eval_progress(
            enumerate(self.val_dataloader),
            desc="Reconstruction",
            total=total_batches,
        )

        with self._ema_eval_scope() as policy:
            pipeline = ReconstructionMetricPipeline(
                [
                    ReconstLossCollector(device=self.device),
                    *build_reconstruction_metric_collectors(
                        policy=policy,
                        cfg=self.cfg,
                        device=self.device,
                    ),
                ],
                annotators=build_reconstruction_info_annotators(
                    policy=policy,
                    cfg=self.cfg,
                    device=self.device,
                ),
            )
            policy.eval()
            with torch.inference_mode():
                for i, batch in reconst_iter:
                    if self.max_reconst_batches and i >= self.max_reconst_batches:
                        break
                    is_padding = is_distributed_padding_batch(self.val_dataloader, i)
                    batch = self._move_batch_tensors_to_device(
                        batch, phase="reconstruct"
                    )
                    batch = self.preprocess_batch(batch)
                    with self.runtime.autocast():
                        obs = _build_reconstruction_obs(
                            policy=policy,
                            batch=batch,
                            fallback_io=getattr(self, "io", None),
                        )
                        try:
                            pred_action, reconst_info = policy.predict_action_chunk(
                                obs,
                                decode_temperature=0.0,
                                decode_top_k=1,
                            )
                        except NotImplementedError:
                            if (
                                self.runtime.is_main_process
                                and not warned_missing_chunk_api
                            ):
                                print(
                                    "Policy does not implement predict_action_chunk(); "
                                    "skipping reconstruction metric."
                                )
                                warned_missing_chunk_api = True
                            continue

                    gt_action = batch[action_key]
                    if pred_action.shape != gt_action.shape:
                        raise RuntimeError(
                            "Reconstruction shape mismatch: "
                            f"pred_action={tuple(pred_action.shape)} vs gt_action={tuple(gt_action.shape)}"
                        )
                    if is_padding:
                        continue
                    pipeline.update(
                        ReconstructionBatchContext(
                            pred_action=pred_action,
                            gt_action=gt_action,
                            batch=batch,
                            reconst_info=reconst_info,
                            policy=policy,
                            action_key=action_key,
                        )
                    )

        self.model.train()
        release_cuda_eval_memory(self.device)
        pipeline.reduce(self.runtime)
        return pipeline.finalize()

    def sim_eval(self) -> dict[str, float]:
        return inline_policy_eval.InlinePolicyEvalRunner(self).run()
