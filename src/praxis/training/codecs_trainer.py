# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Codec trainer: DefaultTrainer specialization for action codecs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from omegaconf import DictConfig, OmegaConf

from praxis.codecs.factory import make_codec
from praxis.core.config import require_config_dict
from praxis.core.types import FeatureSpec, NormMode, PolicyIO
from praxis.data.lerobot_adapter import (
    LeRobotActionOnlyDatasetAdapter,
    _episode_bounds_from_metadata,
    ensure_lerobot_dataset_available,
    read_lerobot_cache_provenance,
)
from praxis.data.utils import resolve_action_feature_from_metadata, stats_to_tensors
from praxis.data.xtokenizer_canonical_stats import XTokenizerQuantileStatistics
from praxis.data.xtokenizer_vl import VLFeatureManifest, VLFeatureSidecar
from praxis.training.core.dataset_filtering import (
    filter_episode_pool,
    load_action_stats_payload,
    load_excluded_episode_ids,
)
from praxis.training.core.distributed import is_distributed_padding_batch
from praxis.training.core.eval_collectors import (
    ReconstLossCollector,
    ReconstructionBatchContext,
    ReconstructionMetricPipeline,
    build_reconstruction_metric_collectors,
)
from praxis.training.core.trainer import (
    DefaultTrainer,
    resolve_sampler_frame_drops,
)


def _resolve_codec_build_cfg(
    codec_cfg: DictConfig | dict[str, Any],
    *,
    dataset_action_dim: int,
    dataset_action_layout: dict[str, Any] | None = None,
    is_main_process: bool,
) -> dict[str, Any]:
    if isinstance(codec_cfg, DictConfig):
        cfg_dict = OmegaConf.to_container(codec_cfg, resolve=True)
    else:
        cfg_dict = dict(codec_cfg)
    if not isinstance(cfg_dict, dict):
        raise TypeError(f"codec config must resolve to a dict, got {type(cfg_dict)!r}")
    cfg_dict = cast(dict[str, Any], cfg_dict)

    pretrained_path = cfg_dict.get("pretrained_path")
    configured_action_dim = cfg_dict.get("action_dim")
    if not pretrained_path:
        if (
            configured_action_dim is not None
            and int(configured_action_dim) != dataset_action_dim
            and is_main_process
        ):
            print(
                f"[CodecsTrainer] Overriding codec.action_dim={configured_action_dim} "
                f"with dataset action_dim={dataset_action_dim}."
            )
        cfg_dict["action_dim"] = dataset_action_dim
        if (
            cfg_dict.get("kind") == "latents"
            and cfg_dict.get("name") == "xtokenizer"
            and dataset_action_layout is not None
        ):
            cfg_dict["action_layout"] = dict(dataset_action_layout)
    return cfg_dict


def _extract_codec_action_dim(codec: torch.nn.Module) -> int | None:
    action_dim = getattr(codec, "action_dim", None)
    if action_dim is None:
        cfg = getattr(codec, "config", None)
        action_dim = getattr(cfg, "action_dim", None)
    return int(action_dim) if action_dim is not None else None


def _supports_train_loop(codec: torch.nn.Module, optim_params: list[dict]) -> bool:
    del codec
    return bool(optim_params)


def _should_skip_codec_fit(cfg: Any) -> bool:
    trainer_cfg = getattr(cfg, "trainer", None)
    return bool(getattr(trainer_cfg, "skip_codec_fit", False))


def _open_xtokenizer_canonical_stats(
    dataset_cfg: Any,
    codec_cfg: Any,
) -> XTokenizerQuantileStatistics | None:
    """Open the immutable Bridge normalization contract before dataset access."""
    if str(
        getattr(dataset_cfg, "adapter", "")
    ) != "xtokenizer_sidecar" or not _is_xtokenizer_config(codec_cfg):
        return None

    dataset_action_horizon = getattr(dataset_cfg, "action_horizon", None)
    codec_action_horizon = getattr(codec_cfg, "action_horizon", None)
    if any(
        not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
        for horizon in (dataset_action_horizon, codec_action_horizon)
    ):
        raise ValueError(
            "dataset.action_horizon and codec.action_horizon must be positive "
            "integers for X-Tokenizer training."
        )
    dataset_action_horizon = cast(int, dataset_action_horizon)
    codec_action_horizon = cast(int, codec_action_horizon)
    if dataset_action_horizon != codec_action_horizon:
        raise ValueError(
            "X-Tokenizer action horizon mismatch: "
            f"dataset.action_horizon={dataset_action_horizon} != "
            f"codec.action_horizon={codec_action_horizon}."
        )

    artifact_path = getattr(dataset_cfg, "canonical_stats", None)
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError(
            "dataset.canonical_stats must name a canonical X-Tokenizer "
            "statistics artifact."
        )
    action_layout = getattr(dataset_cfg, "action_layout", None)
    if action_layout is None or str(getattr(action_layout, "convention", "")) != (
        "absolute"
    ):
        raise ValueError(
            "dataset.action_layout must define the absolute canonical "
            "X-Tokenizer action layout."
        )
    dof_dims = getattr(action_layout, "dof_dims", None)
    dof_order = getattr(action_layout, "dof_order", None)
    if dof_dims is None or dof_order is None:
        raise ValueError(
            "dataset.action_layout must define dof_dims and explicit dof_order."
        )
    dims = {str(name): int(dim) for name, dim in dof_dims.items()}
    order = tuple(str(name) for name in dof_order)
    action_dim = sum(dims.values())
    if action_dim != 10:
        raise ValueError(
            "dataset.action_layout must define the canonical 10-D Bridge "
            f"X-Tokenizer action layout, got {action_dim}-D."
        )
    try:
        return XTokenizerQuantileStatistics.from_path(
            Path(artifact_path),
            expected_dataset_repo_id=_required_config_string(dataset_cfg, "repo_id"),
            expected_dataset_revision=_required_config_string(
                dataset_cfg,
                "dataset_revision",
            ),
            expected_robot_type=_required_config_string(
                dataset_cfg,
                "robot_type_name",
            ),
            expected_dof_dims=dims,
            expected_dof_order=order,
            expected_action_dim=action_dim,
            expected_stats_horizon=codec_action_horizon,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"canonical X-Tokenizer statistics artifact does not exist: {artifact_path}"
        ) from None


def _resolve_xtokenizer_action_feature(
    raw_feature: FeatureSpec,
    artifact: XTokenizerQuantileStatistics | None,
) -> FeatureSpec:
    """Replace raw action geometry while retaining its LeRobot query key."""
    if artifact is None:
        return raw_feature
    return FeatureSpec(
        key=raw_feature.key,
        type=raw_feature.type,
        shape=(artifact.action_dim,),
        norm_mode=NormMode.MIN_MAX,
    )


def _xtokenizer_sidecar_identity(sidecar: VLFeatureSidecar) -> dict[str, Any]:
    return sidecar.manifest.identity


def _xtokenizer_canonical_stats_identity(
    artifact: XTokenizerQuantileStatistics,
) -> dict[str, Any]:
    return artifact.identity


def _bind_xtokenizer_canonical_stats_to_runtime_config(
    cfg: Any,
    artifact: XTokenizerQuantileStatistics,
    *,
    resume_checkpoint: str | None,
) -> dict[str, Any]:
    """Persist canonical-stat identity and reject changed resume contracts."""
    identity = _xtokenizer_canonical_stats_identity(artifact)
    if resume_checkpoint is not None:
        saved_config_path = (
            Path(resume_checkpoint) / "pretrained_model" / "train_config.yaml"
        )
        try:
            saved_cfg = OmegaConf.load(saved_config_path)
            saved_identity_cfg = OmegaConf.select(
                saved_cfg,
                "dataset.canonical_stats_identity",
                default=None,
            )
            saved_identity = OmegaConf.to_container(
                saved_identity_cfg,
                resolve=True,
            )
        except Exception as exc:
            raise ValueError(
                "Unable to read the saved X-Tokenizer canonical statistics "
                f"identity from {saved_config_path}."
            ) from exc
        if not isinstance(saved_identity, dict) or set(saved_identity) != set(identity):
            raise ValueError(
                "The saved X-Tokenizer canonical statistics identity is missing "
                "or malformed."
            )
        if saved_identity != identity:
            raise ValueError(
                "X-Tokenizer canonical statistics identity mismatch between "
                "the resume checkpoint and the configured artifact."
            )
    OmegaConf.update(
        cfg,
        "dataset.canonical_stats_identity",
        identity,
        force_add=True,
    )
    return identity


def _bind_xtokenizer_sidecar_to_runtime_config(
    cfg: Any,
    sidecar: VLFeatureSidecar,
    *,
    resume_checkpoint: str | None,
) -> dict[str, Any]:
    """Persist sidecar identity and fail closed when resuming another generation."""
    identity = _xtokenizer_sidecar_identity(sidecar)
    if resume_checkpoint is not None:
        saved_config_path = (
            Path(resume_checkpoint) / "pretrained_model" / "train_config.yaml"
        )
        try:
            saved_cfg = OmegaConf.load(saved_config_path)
            saved_identity_cfg = OmegaConf.select(
                saved_cfg,
                "dataset.vl_sidecar_identity",
                default=None,
            )
            saved_identity = OmegaConf.to_container(
                saved_identity_cfg,
                resolve=True,
            )
        except Exception as exc:
            raise ValueError(
                "Unable to read the saved X-Tokenizer sidecar identity from "
                f"{saved_config_path}."
            ) from exc
        if (
            not isinstance(saved_identity, dict)
            or set(saved_identity) != set(identity)
            or not isinstance(saved_identity.get("generation_fingerprint"), str)
            or not saved_identity["generation_fingerprint"]
        ):
            raise ValueError(
                "The saved X-Tokenizer sidecar identity is missing or malformed."
            )
        if saved_identity != identity:
            raise ValueError(
                "X-Tokenizer sidecar identity mismatch between the resume "
                "checkpoint and the configured offline feature generation."
            )
    OmegaConf.update(
        cfg,
        "dataset.vl_sidecar_identity",
        identity,
        force_add=True,
    )
    return identity


def _open_xtokenizer_vl_sidecar(
    dataset_cfg: Any,
    codec_cfg: Any,
    *,
    expected_episode_lengths: dict[int, int] | None = None,
    actual_dataset_revision: str | None = None,
) -> VLFeatureSidecar | None:
    """Open and fully validate X-Tokenizer sidecar metadata during data init."""
    if (
        str(getattr(codec_cfg, "kind", "")) != "latents"
        or str(getattr(codec_cfg, "name", "")) != "xtokenizer"
    ):
        return None
    sidecar_root = getattr(dataset_cfg, "vl_feature_sidecar", None)
    if sidecar_root is None:
        sidecar_root = getattr(dataset_cfg, "vl_sidecar", None)
    if sidecar_root is None:
        return None

    dataset_repo_id = _required_config_string(dataset_cfg, "repo_id")
    manifest = VLFeatureManifest.from_path(Path(str(sidecar_root)) / "manifest.json")
    configured_dataset_revision = _required_config_string(
        dataset_cfg,
        "dataset_revision",
    )
    model_id = _required_config_string(dataset_cfg, "vl_model_id")
    model_revision = _required_config_string(
        dataset_cfg,
        "vl_model_revision",
    )
    if manifest.synthetic:
        dataset_revision = configured_dataset_revision
    else:
        dataset_revision = _require_hub_commit_sha(
            configured_dataset_revision,
            config_path="dataset.dataset_revision",
        )
        model_revision = _require_hub_commit_sha(
            model_revision,
            config_path="dataset.vl_model_revision",
        )
        if actual_dataset_revision is not None:
            actual_revision = _require_hub_commit_sha(
                actual_dataset_revision,
                config_path="loaded dataset revision",
            )
            if actual_revision != dataset_revision:
                raise ValueError(
                    "loaded dataset revision does not match "
                    "dataset.dataset_revision: "
                    f"{actual_revision!r} != {dataset_revision!r}."
                )
    sidecar = VLFeatureSidecar.open(
        Path(str(sidecar_root)),
        dataset_revision=dataset_revision,
        expected_dataset_repo_id=dataset_repo_id,
        expected_model_id=model_id,
        expected_extractor_revision=_required_config_string(
            dataset_cfg,
            "vl_extractor_revision",
        ),
        expected_model_revision=model_revision,
        expected_image_keys=_required_config_string_tuple(
            dataset_cfg,
            "vl_image_keys",
            allow_empty=manifest.synthetic,
        ),
        expected_global_instruction=_required_config_string(
            dataset_cfg,
            "vl_global_instruction",
        ),
        expected_local_subtask=_required_config_string(
            dataset_cfg,
            "vl_local_subtask",
        ),
        expected_feature_dim=int(codec_cfg.vl_feature_dim),
        expected_stride=int(codec_cfg.compression_ratio),
    )
    if expected_episode_lengths is not None:
        sidecar.validate_episode_lengths(expected_episode_lengths)
    return sidecar


def _required_config_string(cfg: Any, key: str) -> str:
    value = getattr(cfg, key, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"dataset.{key} must be a non-empty string when using a VL sidecar."
        )
    return value


def _required_config_string_tuple(
    cfg: Any,
    key: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = getattr(cfg, key, None)
    if value is None or isinstance(value, str):
        raise ValueError(
            f"dataset.{key} must be a non-empty string sequence when using a VL sidecar."
        )
    try:
        values = tuple(value)
    except TypeError:
        raise ValueError(
            f"dataset.{key} must be a non-empty string sequence when using a VL sidecar."
        ) from None
    if (
        (not values and not allow_empty)
        or any(not isinstance(item, str) or not item.strip() for item in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            f"dataset.{key} must contain unique non-empty strings "
            "when using a VL sidecar."
        )
    return cast(tuple[str, ...], values)


def _require_hub_commit_sha(value: str, *, config_path: str) -> str:
    if len(value) != 40:
        raise ValueError(
            f"{config_path} must be an immutable 40-character hexadecimal "
            "Hub commit SHA for a real X-Tokenizer VL sidecar."
        )
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(
            f"{config_path} must be an immutable 40-character hexadecimal "
            "Hub commit SHA for a real X-Tokenizer VL sidecar."
        ) from None
    return value.lower()


def _is_xtokenizer_config(codec_cfg: Any) -> bool:
    return (
        str(getattr(codec_cfg, "kind", "")) == "latents"
        and str(getattr(codec_cfg, "name", "")) == "xtokenizer"
    )


def _resolve_xtokenizer_dataset_revision(
    dataset_cfg: Any,
    codec_cfg: Any,
) -> str | None:
    if not _is_xtokenizer_config(codec_cfg):
        return None
    sidecar_root = getattr(dataset_cfg, "vl_feature_sidecar", None)
    if sidecar_root is None:
        sidecar_root = getattr(dataset_cfg, "vl_sidecar", None)
    if sidecar_root is None:
        return None
    manifest = VLFeatureManifest.from_path(Path(str(sidecar_root)) / "manifest.json")
    if manifest.synthetic:
        return None
    return _require_hub_commit_sha(
        _required_config_string(dataset_cfg, "dataset_revision"),
        config_path="dataset.dataset_revision",
    )


def _resolve_xtokenizer_tail_drop(
    codec_cfg: Any,
    *,
    drop_n_last_frames: int,
    has_sidecar: bool,
) -> int:
    """Reserve the next frame on the episode-level VL sampling grid."""
    if (
        not has_sidecar
        or not _is_xtokenizer_config(codec_cfg)
        or float(getattr(codec_cfg, "prediction_weight", 0.0)) <= 0.0
    ):
        return drop_n_last_frames
    required_tail = int(codec_cfg.action_horizon) + int(codec_cfg.compression_ratio) - 1
    return max(drop_n_last_frames, required_tail)


class CodecsTrainer(DefaultTrainer):
    """Training loop specialization for action codecs."""

    def __init__(self, cfg: DictConfig) -> None:
        self.reconst_every = cfg.trainer.reconst_every
        self.max_reconst_batches = cfg.trainer.max_reconst_batches
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # DefaultTrainer hooks
    # ------------------------------------------------------------------

    def _validate_distributed_config(self) -> None:
        if str(self.tc.distributed_strategy) == "fsdp":
            raise ValueError(
                "Codec training does not support trainer.distributed_strategy=fsdp. "
                "Use ddp for codec training; FSDP is currently reserved for "
                "VLM-based policy training."
            )
        super()._validate_distributed_config()

    def _init_data(self) -> None:
        dc = self.cfg.dataset
        excluded_episodes = load_excluded_episode_ids(
            getattr(dc, "exclude_episodes_file", None)
        )

        self.filtered_action_stats = None
        stats_payload = load_action_stats_payload(
            getattr(dc, "action_stats_file", None)
        )
        if stats_payload is not None:
            self.filtered_action_stats = stats_to_tensors(stats_payload)
        xtokenizer_normalization_statistics = _open_xtokenizer_canonical_stats(
            dc,
            self.cfg.codec,
        )
        self.xtokenizer_normalization_statistics = xtokenizer_normalization_statistics
        if xtokenizer_normalization_statistics is not None:
            _bind_xtokenizer_canonical_stats_to_runtime_config(
                self.cfg,
                xtokenizer_normalization_statistics,
                resume_checkpoint=self._resume_checkpoint,
            )
        configured_revision = _resolve_xtokenizer_dataset_revision(
            dc,
            self.cfg.codec,
        )
        ensure_lerobot_dataset_available(
            repo_id=dc.repo_id,
            runtime=self.runtime,
            root=dc.root,
            episodes=dc.episodes,
            download_videos=False,
            bypass_force_cache_sync=bool(getattr(dc, "bypass_force_cache_sync", False)),
            revision=str(configured_revision)
            if configured_revision is not None
            else None,
        )
        actual_revision: str | None = None
        if configured_revision is not None:
            actual_revision = read_lerobot_cache_provenance(
                root=dc.root,
                repo_id=dc.repo_id,
            )
            if actual_revision != configured_revision:
                raise ValueError(
                    "LeRobot cache provenance revision does not match "
                    "dataset.dataset_revision: "
                    f"{actual_revision!r} != {configured_revision!r}. "
                    "Rerun real prepare_vl_features.py for the configured "
                    "dataset revision."
                )
        temp_dataset_kwargs: dict[str, Any] = {
            "repo_id": dc.repo_id,
            "root": dc.root,
            "episodes": dc.episodes,
        }
        if configured_revision is not None:
            temp_dataset_kwargs["revision"] = str(configured_revision)
        temp_ds = LeRobotDataset(**temp_dataset_kwargs)
        configured_episodes = list(dc.episodes or range(temp_ds.meta.total_episodes))
        all_episodes = filter_episode_pool(configured_episodes, excluded_episodes)
        if excluded_episodes and self.runtime.is_main_process:
            print(
                f"[CodecsTrainer] Excluding {len(excluded_episodes)} episodes; "
                f"training/validation pool contains {len(all_episodes)} episodes."
            )
        meta_features = temp_ds.meta.features
        raw_action_feature = resolve_action_feature_from_metadata(
            meta_features,
            context="dataset metadata for codec training",
        )
        action_feature = _resolve_xtokenizer_action_feature(
            raw_action_feature,
            xtokenizer_normalization_statistics,
        )
        self.action_key = action_feature.key
        action_shape = action_feature.shape
        self.dataset_action_dim = int(action_shape[0])
        self.dataset_action_layout = (
            None
            if xtokenizer_normalization_statistics is None
            else {
                "convention": "absolute",
                "dof_dims": dict(xtokenizer_normalization_statistics.dof_dims),
                "dof_order": list(xtokenizer_normalization_statistics.dof_order),
            }
        )
        action_horizon = int(self.cfg.codec.action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"codec.action_horizon must be > 0, got {action_horizon}.")
        io = PolicyIO(
            obs_features=[],
            action_feature=action_feature,
            n_obs_steps=0,
            n_action_steps=action_horizon,
            fps=temp_ds.fps,
        )
        self.io = io
        episodes_meta = getattr(temp_ds.meta, "episodes", None)
        if episodes_meta is None:
            episode_lengths = None
        else:
            episode_bounds = _episode_bounds_from_metadata(episodes_meta)
            try:
                episode_lengths = {
                    int(episode): (
                        episode_bounds[int(episode)][1]
                        - episode_bounds[int(episode)][0]
                    )
                    for episode in all_episodes
                }
            except KeyError as exc:
                raise ValueError(
                    f"dataset meta.episodes does not contain episode {int(exc.args[0])}."
                ) from None
        if actual_revision is None:
            actual_revision = str(
                getattr(
                    temp_ds.meta,
                    "revision",
                    getattr(temp_ds, "revision", ""),
                )
            )
        self.vl_sidecar = _open_xtokenizer_vl_sidecar(
            dc,
            self.cfg.codec,
            expected_episode_lengths=episode_lengths,
            actual_dataset_revision=actual_revision,
        )
        if self.vl_sidecar is not None and episode_lengths is None:
            raise ValueError(
                "dataset meta.episodes is required to validate VL sidecar bounds."
            )
        if self.vl_sidecar is not None:
            _bind_xtokenizer_sidecar_to_runtime_config(
                self.cfg,
                self.vl_sidecar,
                resume_checkpoint=self._resume_checkpoint,
            )
        del temp_ds

        train_episodes, val_episodes = self._split_episodes(all_episodes)

        is_xtokenizer = _is_xtokenizer_config(self.cfg.codec)
        require_valid_next_vl = bool(
            is_xtokenizer
            and self.vl_sidecar is not None
            and float(getattr(self.cfg.codec, "prediction_weight", 0.0)) > 0.0
        )
        adapter_kwargs: dict[str, Any] = {
            "repo_id": dc.repo_id,
            "io": io,
            "root": dc.root,
            "episodes": train_episodes,
            "video_backend": dc.video_backend,
            "vl_sidecar": self.vl_sidecar,
            "robot_type": int(getattr(dc, "robot_type", 0)),
            "include_xtokenizer_context": is_xtokenizer,
            "require_valid_next_vl": require_valid_next_vl,
        }
        if configured_revision is not None:
            adapter_kwargs["revision"] = str(configured_revision)
        if xtokenizer_normalization_statistics is not None:
            adapter_kwargs.update(
                {
                    "xtokenizer_bridge_dof_dims": (
                        xtokenizer_normalization_statistics.dof_dims
                    ),
                    "xtokenizer_bridge_dof_order": (
                        xtokenizer_normalization_statistics.dof_order
                    ),
                }
            )
        self.dataset = LeRobotActionOnlyDatasetAdapter(
            **adapter_kwargs,
        )
        drop_n_first_frames, drop_n_last_frames = resolve_sampler_frame_drops(
            io,
            configured_drop_n_first_frames=self.tc.drop_n_first_frames,
            configured_drop_n_last_frames=self.tc.drop_n_last_frames,
        )
        drop_n_last_frames = _resolve_xtokenizer_tail_drop(
            self.cfg.codec,
            drop_n_last_frames=drop_n_last_frames,
            has_sidecar=self.vl_sidecar is not None,
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
        # Keep an unsharded main-rank loader for bootstrap-only codec fit().
        # Fitting BPE/other non-gradient codecs on rank 0 must see the
        # dataset stream, not only rank 0's training shard.
        self.fit_dataloader = self.dataloader

        if val_episodes:
            val_adapter_kwargs = dict(adapter_kwargs)
            val_adapter_kwargs["episodes"] = val_episodes
            self.val_dataset = LeRobotActionOnlyDatasetAdapter(**val_adapter_kwargs)
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
        codec_build_cfg = _resolve_codec_build_cfg(
            self.cfg.codec,
            dataset_action_dim=self.dataset_action_dim,
            dataset_action_layout=getattr(self, "dataset_action_layout", None),
            is_main_process=self.runtime.is_main_process,
        )
        self.model = make_codec(codec_build_cfg)
        self.model.set_io(self.io)

        codec_action_dim = _extract_codec_action_dim(self.model)
        if codec_action_dim is not None and codec_action_dim != self.dataset_action_dim:
            pretrained_path = codec_build_cfg.get("pretrained_path")
            where = (
                f"pretrained codec at {pretrained_path}"
                if pretrained_path
                else "codec config"
            )
            raise ValueError(
                f"Codec action_dim mismatch: dataset action_dim={self.dataset_action_dim}, "
                f"but {where} has action_dim={codec_action_dim}. "
                "Use a matching checkpoint or dataset."
            )
        xtokenizer_stats = getattr(
            self,
            "xtokenizer_normalization_statistics",
            None,
        )
        if xtokenizer_stats is None:
            normalization_stats = self.filtered_action_stats or self.dataset.stats
            self.model.set_normalization_stats(
                normalization_stats,
                modes=getattr(self.dataset, "normalization_modes", None),
            )
        else:
            set_xtokenizer_stats = getattr(
                self.model,
                "set_xtokenizer_normalization_stats",
                None,
            )
            if not callable(set_xtokenizer_stats):
                raise TypeError(
                    "The X-Tokenizer paper recipe requires a codec with "
                    "set_xtokenizer_normalization_stats()."
                )
            set_xtokenizer_stats(
                predict_stats=xtokenizer_stats.predict_normalization_stats(
                    self.action_key
                ),
                observation_stats=(
                    xtokenizer_stats.observation_normalization_stats(
                        "observation.state"
                    )
                ),
                canonical_stats_identity=xtokenizer_stats.identity,
                stats_horizon=xtokenizer_stats.stats_horizon,
            )

    def _get_optim_params(self) -> list[dict]:
        model = self.runtime.unwrap_model(self.model)
        get_optim_params = getattr(model, "get_optim_params", None)
        if not callable(get_optim_params):
            return []
        return cast(list[dict], get_optim_params())

    def _resolve_run_train_loop(self, optim_params: list[dict]) -> bool:
        return _supports_train_loop(self.model, optim_params)

    def _non_trainable_message(self) -> str | None:
        return (
            "Codec does not support train-time optimization; "
            "running stats bootstrap and checkpoint export only."
        )

    def _training_description(self) -> str:
        return "Codec Training"

    def _periodic_eval_hooks(self):
        return [
            (self.tc.val_every, self.validate, False),
            (self.reconst_every, self.reconstruct, True),
        ]

    def _required_tensor_keys(self, phase: str, batch: dict) -> set[str]:
        del phase
        if self.action_key is None:
            raise RuntimeError(
                "CodecsTrainer action_key must be resolved before batching."
            )
        required = {self.action_key}
        pad_key = f"{self.action_key}_is_pad"
        if pad_key in batch:
            required.add(pad_key)
        required.update(
            key
            for key in (
                "vl_features",
                "next_vl_features",
                "next_vl_feature_is_pad",
                "vl_feature_is_pad",
                "observation.state",
                "robot_type",
                "action_dof_mask",
            )
            if key in batch
        )
        return required

    def _publish_training_context_to_model(self) -> None:
        super()._publish_training_context_to_model()
        model = self.runtime.unwrap_model(self.model)
        set_training_epoch = getattr(model, "set_training_epoch", None)
        if not callable(set_training_epoch):
            return

        batches_per_epoch = len(self.dataloader)
        if batches_per_epoch <= 0:
            raise ValueError(
                "Codec training dataloader must contain at least one batch."
            )
        batches_seen = getattr(self, "_training_batches_seen", None)
        if batches_seen is None:
            accumulation_steps = max(
                1,
                int(getattr(self.tc, "gradient_accumulation_steps", 1)),
            )
            batches_seen = int(self.global_step) * accumulation_steps
        set_training_epoch(int(batches_seen) // batches_per_epoch)

    def train_step(self, batch: dict) -> tuple[torch.Tensor, dict[str, float], bool]:
        result = super().train_step(batch)
        batches_seen = getattr(self, "_training_batches_seen", None)
        if batches_seen is None:
            accumulation_steps = max(
                1,
                int(getattr(self.tc, "gradient_accumulation_steps", 1)),
            )
            batches_seen = int(self.global_step) * accumulation_steps
        self._training_batches_seen = int(batches_seen) + 1
        return result

    def _run_without_training_loop(self) -> None:
        unwrapped = self.runtime.unwrap_model(self.model)

        def _fit_on_main_rank() -> None:
            if not self.runtime.is_main_process:
                return
            if _should_skip_codec_fit(self.cfg):
                print(
                    "[CodecsTrainer] Skipping codec fit(); bootstrapping stats/checkpoint only."
                )
            elif hasattr(unwrapped, "fit") and callable(unwrapped.fit):
                self._run_fit(unwrapped)

        main_process_first = getattr(self.runtime, "main_process_first", None)
        if callable(main_process_first):
            with cast(AbstractContextManager[None], main_process_first()):
                _fit_on_main_rank()
        else:
            _fit_on_main_rank()
            self.runtime.wait_for_everyone()

        self.save_latest_checkpoint()
        rank0_error = None
        if self.runtime.is_main_process:
            try:
                self.logger.log({"codec/bootstrap_complete": 1.0}, step=0)
                print(
                    "Codec bootstrap complete. Checkpoint saved with dataset normalization stats."
                )
            except BaseException as exc:
                rank0_error = (
                    "Codec bootstrap logging failed on rank 0: "
                    f"{type(exc).__name__}: {exc}"
                )
        broadcast_object = getattr(self.runtime, "broadcast_object", None)
        if callable(broadcast_object):
            rank0_error = broadcast_object(rank0_error, src=0)
        if rank0_error is not None:
            raise RuntimeError(str(rank0_error))
        self._on_train_end()

    def _run_fit(self, codec: torch.nn.Module) -> None:
        """Collect action chunks from the dataloader and call fit()."""
        max_chunks = int(self.cfg.codec.max_fit_chunks)
        if self.runtime.is_main_process:
            print(
                f"[CodecsTrainer] Collecting up to {max_chunks} action chunks for fit()..."
            )
        chunks: list[torch.Tensor] = []
        total = 0
        fit_dataloader = getattr(self, "fit_dataloader", self.dataloader)
        for batch in fit_dataloader:
            actions = batch[self.action_key]
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu()
            chunks.append(actions)
            total += actions.shape[0]
            if total >= max_chunks:
                break
        all_actions = torch.cat(chunks, dim=0)[:max_chunks]
        if self.runtime.is_main_process:
            print(f"[CodecsTrainer] Fitting codec on {all_actions.shape[0]} chunks...")
        fit_fn = getattr(codec, "fit", None)
        if not callable(fit_fn):
            raise TypeError(
                f"Codec {type(codec).__name__} does not expose a callable fit(...) method."
            )
        fit_fn(all_actions)
        if self.runtime.is_main_process:
            print("[CodecsTrainer] fit() complete.")

    def _serialize_model_config(self, model: torch.nn.Module):
        return require_config_dict(model.config)

    # ------------------------------------------------------------------
    # Codec reconstruction eval hook
    # ------------------------------------------------------------------

    def reconstruct(self) -> dict[str, float]:
        if not self.run_train_loop or self.val_dataloader is None:
            return {}
        if self.action_key is None:
            raise RuntimeError("action_key must be initialized before reconstruct")
        action_key = self.action_key

        total_batches = len(self.val_dataloader)
        if self.max_reconst_batches:
            total_batches = min(total_batches, int(self.max_reconst_batches))
        reconst_iter = self._eval_progress(
            enumerate(self.val_dataloader),
            desc="Reconstruction",
            total=total_batches,
        )
        with self._ema_eval_scope() as codec:
            codec.eval()
            collectors = build_reconstruction_metric_collectors(
                policy=codec,
                cfg=getattr(self, "cfg", None),
                device=self.device,
            )
            default_loss_owners = [
                collector
                for collector in collectors
                if bool(
                    getattr(
                        collector,
                        "owns_default_reconstruction_loss",
                        False,
                    )
                )
            ]
            if len(default_loss_owners) > 1:
                owner_names = ", ".join(
                    type(collector).__name__ for collector in default_loss_owners
                )
                raise RuntimeError(
                    "Multiple registered reconstruction metric collectors claim "
                    "ownership of 'val_reconst_loss': "
                    f"{owner_names}."
                )
            if not default_loss_owners:
                collectors.insert(0, ReconstLossCollector(device=self.device))
            metric_pipeline = ReconstructionMetricPipeline(collectors)
            with torch.inference_mode():
                for i, batch in reconst_iter:
                    if self.max_reconst_batches and i >= self.max_reconst_batches:
                        break
                    if is_distributed_padding_batch(self.val_dataloader, i):
                        continue
                    batch = self._move_batch_tensors_to_device(
                        batch, phase="reconstruct"
                    )
                    actions = batch[action_key]
                    with self.runtime.autocast():
                        recon_actions, reconst_info = _reconstruct_actions(
                            codec,
                            actions,
                            batch=batch,
                            action_key=action_key,
                        )
                    metric_pipeline.update(
                        ReconstructionBatchContext(
                            pred_action=recon_actions,
                            gt_action=actions,
                            batch=batch,
                            reconst_info=reconst_info,
                            policy=codec,
                            action_key=action_key,
                        )
                    )
        self.model.train()

        metric_pipeline.reduce(self.runtime)
        return metric_pipeline.finalize()


def _reconstruct_actions(
    codec: torch.nn.Module,
    actions: torch.Tensor,
    *,
    batch: dict[str, Any] | None = None,
    action_key: str = "action",
) -> tuple[torch.Tensor, dict[str, Any]]:
    reconstruct_with_info = getattr(codec, "reconstruct_with_info", None)
    if callable(reconstruct_with_info):
        reconstruction_batch = {action_key: actions} if batch is None else batch
        result = reconstruct_with_info(
            actions,
            batch=reconstruction_batch,
            action_key=action_key,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError(
                "reconstruct_with_info(...) must return "
                "(reconstruction, information dictionary)."
            )
        reconstruction, info = result
        if not isinstance(reconstruction, torch.Tensor):
            raise TypeError(
                "reconstruct_with_info(...) reconstruction must be a tensor."
            )
        if not isinstance(info, dict):
            raise TypeError(
                "reconstruct_with_info(...) must return an information dictionary."
            )
        return reconstruction, cast(dict[str, Any], info)

    autoencode = getattr(codec, "autoencode", None)
    if callable(autoencode):
        return cast(torch.Tensor, autoencode(actions)), {}

    tokenize = getattr(codec, "tokenize", None)
    detokenize = getattr(codec, "detokenize", None)
    if callable(tokenize) and callable(detokenize):
        tokens = tokenize(actions)
        return cast(torch.Tensor, detokenize(tokens)), {}

    raise TypeError(
        f"Codec {type(codec).__name__} must expose autoencode(...) or tokenize(...)/detokenize(...) "
        "for reconstruction evaluation."
    )
