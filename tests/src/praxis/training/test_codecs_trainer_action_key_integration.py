# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Integration test for CodecsTrainer with non-default action key."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from praxis.training.codecs_trainer import CodecsTrainer


@dataclass
class _DummyMeta:
    total_episodes: int
    features: dict


class _DummyLeRobotDataset:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        self.meta = _DummyMeta(
            total_episodes=2,
            features={
                "robot_action": {"dtype": "float32", "shape": [3]},
                "episode_index": {"dtype": "int64", "shape": []},
                "index": {"dtype": "int64", "shape": []},
            },
        )
        self.fps = 20


class _ActionDataset(Dataset):
    def __init__(self, action_key: str, action_horizon: int, action_dim: int) -> None:
        self.action_key = action_key
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        return {
            self.action_key: torch.zeros(
                self.action_horizon, self.action_dim, dtype=torch.float32
            ),
            f"{self.action_key}_is_pad": torch.zeros(
                self.action_horizon, dtype=torch.bool
            ),
        }


class _DummyActionOnlyAdapter:
    def __init__(self, *args, io, **kwargs) -> None:
        del args, kwargs
        self.io = io
        self.fps = 20
        self.batch_sizes_requested: list[int] = []
        self.drop_last_requested: list[bool] = []
        action_key = self.io.action_feature.key
        action_dim = self.io.action_feature.shape[0]
        self.stats = {
            action_key: {
                "min": torch.zeros(action_dim, dtype=torch.float32),
                "max": torch.ones(action_dim, dtype=torch.float32),
            }
        }

    def make_dataloader(
        self,
        *,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        drop_last: bool = True,
        **kwargs,
    ) -> DataLoader:
        del num_workers, shuffle, kwargs
        self.batch_sizes_requested.append(int(batch_size))
        self.drop_last_requested.append(bool(drop_last))
        ds = _ActionDataset(
            action_key=self.io.action_feature.key,
            action_horizon=self.io.n_action_steps,
            action_dim=self.io.action_feature.shape[0],
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=drop_last)


def _build_cfg(tmp_path: Path, codec_cfg: dict | None = None):
    if codec_cfg is None:
        codec_cfg = {
            "name": "bin",
            "kind": "tokens",
            "action_horizon": 2,
            "max_fit_chunks": 32,
            "num_bins": 8,
            "min_val": -1.0,
            "max_val": 1.0,
        }

    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "dataset": {
                "repo_id": "dummy/repo",
                "root": None,
                "episodes": [0, 1],
                "video_backend": None,
            },
            "codec": codec_cfg,
            "trainer": {
                "_trainer_type": "codecs",
                "batch_size": 2,
                "num_workers": 0,
                "prefetch_factor": 2,
                "use_episode_aware_sampler": False,
                "drop_n_first_frames": 0,
                "drop_n_last_frames": 0,
                "lr": 1e-4,
                "weight_decay": 1e-6,
                "betas": [0.95, 0.999],
                "eps": 1e-8,
                "lr_scheduler": "cosine",
                "warmup_steps": 1,
                "min_lr_ratio": 0.0,
                "ema_decay": 0.995,
                "mixed_precision": "no",
                "gradient_accumulation_steps": 1,
                "find_unused_parameters": False,
                "distributed_strategy": "ddp",
                "fsdp_min_num_params": 100000000,
                "fsdp_reshard_after_forward": True,
                "fsdp_cpu_offload": False,
                "max_grad_norm": 1.0,
                "use_wandb": False,
                "wandb_project": "praxis-vla",
                "wandb_entity": None,
                "use_ema": False,
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "local_log_path": str(tmp_path / "train_metrics.jsonl"),
                "output_dir": str(tmp_path),
                "checkpoint_candidates": {
                    "best": {
                        "metric_key": "val_reconst_loss",
                        "mode": "min",
                        "k": 3,
                    },
                    "recent": {
                        "k": 2,
                    },
                },
                "val_ratio": 0.0,
                "val_batch_size": None,
                "max_val_batches": 1,
                "val_every": 1000,
                "checkpoint_every": 1000,
                "reconst_every": None,
                "max_reconst_batches": None,
                "max_steps": 1,
                "log_every": 1000,
            },
        }
    )


@pytest.mark.parametrize(
    "codec_cfg",
    [
        {
            "name": "bin",
            "kind": "tokens",
            "action_horizon": 2,
            "max_fit_chunks": 32,
            "num_bins": 8,
            "min_val": -1.0,
            "max_val": 1.0,
        },
        {
            "name": "acodec",
            "kind": "tokens",
            "action_horizon": 2,
            "latent_horizon": 2,
            "emb_dim": 32,
            "latent_dim": 16,
            "head_dim": 8,
            "encoder_depth": 1,
            "decoder_depth": 1,
            "dropout": 0.0,
            "vq_codebook_size": 32,
            "vq_kmeans_init": False,
        },
    ],
)
@pytest.mark.parametrize("explicit_action", [False, True])
def test_codecs_trainer_nondefault_action_key_checkpoint_reload(
    monkeypatch, tmp_path: Path, codec_cfg: dict, explicit_action: bool
) -> None:
    class MultiActionDataset(_DummyLeRobotDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.meta.features["action"] = {"dtype": "float32", "shape": [16]}

    monkeypatch.setattr(
        "praxis.training.codecs_trainer.LeRobotDataset",
        MultiActionDataset if explicit_action else _DummyLeRobotDataset,
    )
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.LeRobotActionOnlyDatasetAdapter",
        _DummyActionOnlyAdapter,
    )
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.ensure_lerobot_dataset_available",
        lambda **kwargs: None,
    )

    cfg = _build_cfg(tmp_path, codec_cfg=codec_cfg)
    if explicit_action:
        cfg.dataset.action_key = "robot_action"
    trainer = CodecsTrainer(cfg)

    assert trainer.action_key == "robot_action"
    assert trainer.model.action_key == "robot_action"
    assert trainer.model.normalize_action is not None
    assert trainer.model.normalize_action.keys == ["robot_action"]

    custom_stats = {
        "robot_action": {
            "min": torch.full((3,), -2.0, dtype=torch.float32),
            "max": torch.full((3,), 2.0, dtype=torch.float32),
        }
    }
    trainer.model.set_normalization_stats(custom_stats)

    x = torch.ones(1, 2, 3, dtype=torch.float32)
    normalize = trainer.model.normalize_action
    assert normalize is not None
    before = normalize({"robot_action": x})["robot_action"]
    checkpoint_dir = tmp_path / "custom_ckpt"
    trainer.save_checkpoint(checkpoint_dir)

    trainer2 = CodecsTrainer(cfg)
    normalize2 = trainer2.model.normalize_action
    assert normalize2 is not None
    preloaded = normalize2({"robot_action": x})["robot_action"]
    trainer2.load_checkpoint(checkpoint_dir)
    normalize2_reloaded = trainer2.model.normalize_action
    assert normalize2_reloaded is not None
    after = normalize2_reloaded({"robot_action": x})["robot_action"]

    assert not torch.allclose(preloaded, before)
    torch.testing.assert_close(after, before)


def test_codecs_trainer_val_batch_size_defaults_to_train_batch_size(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.LeRobotDataset", _DummyLeRobotDataset
    )
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.LeRobotActionOnlyDatasetAdapter",
        _DummyActionOnlyAdapter,
    )
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.ensure_lerobot_dataset_available",
        lambda **kwargs: None,
    )

    cfg = _build_cfg(tmp_path)
    cfg.trainer.val_ratio = 1.0
    cfg.trainer.batch_size = 3
    cfg.trainer.val_batch_size = None

    trainer = CodecsTrainer(cfg)

    assert trainer.val_batch_size == 3
    assert trainer.val_dataloader is not None
    train_dataset = cast(_DummyActionOnlyAdapter, trainer.dataset)
    val_dataset = cast(_DummyActionOnlyAdapter, trainer.val_dataset)
    assert val_dataset.batch_sizes_requested == [3]
    # Training drops partial batches; the fit/bootstrap loader keeps them.
    assert train_dataset.drop_last_requested == [True, False]
    assert val_dataset.drop_last_requested == [False]


def test_codecs_trainer_val_batch_size_uses_override(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.LeRobotDataset", _DummyLeRobotDataset
    )
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.LeRobotActionOnlyDatasetAdapter",
        _DummyActionOnlyAdapter,
    )
    monkeypatch.setattr(
        "praxis.training.codecs_trainer.ensure_lerobot_dataset_available",
        lambda **kwargs: None,
    )

    cfg = _build_cfg(tmp_path)
    cfg.trainer.val_ratio = 1.0
    cfg.trainer.batch_size = 3
    cfg.trainer.val_batch_size = 1

    trainer = CodecsTrainer(cfg)

    assert trainer.val_batch_size == 1
    assert trainer.val_dataloader is not None
    train_dataset = cast(_DummyActionOnlyAdapter, trainer.dataset)
    val_dataset = cast(_DummyActionOnlyAdapter, trainer.val_dataset)
    assert val_dataset.batch_sizes_requested == [1]
    assert train_dataset.drop_last_requested == [True, False]
    assert val_dataset.drop_last_requested == [False]
