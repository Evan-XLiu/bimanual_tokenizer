from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.tokenizer.evaluate_reconstruction import (
    load_train_split_config,
    resolve_checkpoint_paths,
    resolve_eval_episodes,
    split_validation_episodes,
    supported_keep_ks,
    validate_keep_ks,
)


def _write_train_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "codec": {"name": "boat"},
                "dataset": {
                    "repo_id": "tester/dataset",
                    "root": "/tmp/dataset",
                    "episodes": [2, 4, 6],
                },
                "trainer": {
                    "val_ratio": 0.25,
                    "split_seed": 123,
                    "use_episode_aware_sampler": False,
                    "drop_n_first_frames": 3,
                    "drop_n_last_frames": 5,
                },
            }
        ),
        encoding="utf-8",
    )


def test_resolve_checkpoint_paths_accepts_latest_dir(tmp_path: Path) -> None:
    latest = tmp_path / "checkpoints" / "latest"
    train_config = latest / "pretrained_model" / "train_config.yaml"
    _write_train_config(train_config)

    resolved = resolve_checkpoint_paths(latest)

    assert resolved.checkpoint == latest
    assert resolved.pretrained_model == latest / "pretrained_model"
    assert resolved.train_config == train_config


def test_resolve_checkpoint_paths_accepts_pretrained_model_dir(
    tmp_path: Path,
) -> None:
    pretrained = tmp_path / "checkpoints" / "latest" / "pretrained_model"
    train_config = pretrained / "train_config.yaml"
    _write_train_config(train_config)

    resolved = resolve_checkpoint_paths(pretrained)

    assert resolved.checkpoint == pretrained.parent
    assert resolved.pretrained_model == pretrained
    assert resolved.train_config == train_config


def test_resolve_checkpoint_paths_requires_train_config(tmp_path: Path) -> None:
    latest = tmp_path / "checkpoints" / "latest"
    (latest / "pretrained_model").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="train_config.yaml"):
        resolve_checkpoint_paths(latest)


def test_load_train_split_config_parses_dataset_and_trainer_fields(
    tmp_path: Path,
) -> None:
    train_config = tmp_path / "train_config.yaml"
    _write_train_config(train_config)

    cfg = load_train_split_config(train_config)

    assert cfg.codec_name == "boat"
    assert cfg.dataset_repo_id == "tester/dataset"
    assert cfg.dataset_root == "/tmp/dataset"
    assert cfg.dataset_episodes == [2, 4, 6]
    assert cfg.val_ratio == 0.25
    assert cfg.split_seed == 123
    assert cfg.use_episode_aware_sampler is False
    assert cfg.drop_n_first_frames == 3
    assert cfg.drop_n_last_frames == 5


def test_load_train_split_config_resolves_seed_interpolation(tmp_path: Path) -> None:
    train_config = tmp_path / "train_config.yaml"
    train_config.parent.mkdir(parents=True, exist_ok=True)
    train_config.write_text(
        "\n".join(
            [
                "seed: 42",
                "codec:",
                "  name: boat",
                "dataset:",
                "  repo_id: tester/dataset",
                "trainer:",
                "  val_ratio: 0.25",
                "  split_seed: ${seed}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_train_split_config(train_config)

    assert cfg.split_seed == 42


def test_load_train_split_config_missing_split_seed_marks_legacy_tail(
    tmp_path: Path,
) -> None:
    train_config = tmp_path / "train_config.yaml"
    train_config.parent.mkdir(parents=True, exist_ok=True)
    train_config.write_text(
        yaml.safe_dump(
            {
                "codec": {"name": "boat"},
                "dataset": {"repo_id": "tester/dataset"},
                "trainer": {"val_ratio": 0.25},
            }
        ),
        encoding="utf-8",
    )

    cfg = load_train_split_config(train_config)

    assert cfg.split_seed is None


def test_split_validation_episodes_uses_seeded_exact_count_mask() -> None:
    episodes = list(range(100))

    val_episodes = split_validation_episodes(
        all_episodes=episodes,
        val_ratio=0.1,
        split_seed=123,
    )

    assert len(val_episodes) == 10
    assert val_episodes != episodes[-10:]
    assert val_episodes == split_validation_episodes(
        all_episodes=episodes,
        val_ratio=0.1,
        split_seed=123,
    )
    assert (
        len(
            split_validation_episodes(
                all_episodes=[0, 1, 2],
                val_ratio=0.01,
                split_seed=123,
            )
        )
        == 1
    )


def test_split_validation_episodes_preserves_legacy_tail_when_seed_missing() -> None:
    assert split_validation_episodes(
        all_episodes=[0, 1, 2, 3],
        val_ratio=0.25,
        split_seed=None,
    ) == [3]


def test_resolve_eval_episodes_supports_all_and_val() -> None:
    episodes = [10, 11, 12, 13]

    assert (
        resolve_eval_episodes(
            split="all",
            all_episodes=episodes,
            val_ratio=0.5,
        )
        == episodes
    )
    assert resolve_eval_episodes(
        split="val",
        all_episodes=episodes,
        val_ratio=0.5,
        split_seed=123,
    ) == split_validation_episodes(
        all_episodes=episodes,
        val_ratio=0.5,
        split_seed=123,
    )


class _DummyTokenizer:
    def __init__(
        self,
        *,
        block_schedule: list[int] | None = None,
        latent_horizon: int | None = None,
        supports_prefix_decode: bool = True,
    ) -> None:
        self.block_schedule = block_schedule
        self.latent_horizon = latent_horizon
        self.supports_prefix_decode = supports_prefix_decode


def test_supported_keep_ks_allows_all_prefix_lengths_for_oat() -> None:
    tokenizer: Any = _DummyTokenizer(latent_horizon=4)

    assert supported_keep_ks(tokenizer, tokenizer_name="oat") == [1, 2, 3, 4]


def test_supported_keep_ks_allows_boat_block_schedule_prefixes_only() -> None:
    tokenizer: Any = _DummyTokenizer(block_schedule=[1, 1, 2, 4, 8])

    assert supported_keep_ks(tokenizer, tokenizer_name="boat") == [1, 2, 4, 8, 16]


def test_validate_keep_ks_rejects_boat_values_outside_block_schedule() -> None:
    tokenizer: Any = _DummyTokenizer(block_schedule=[1, 1, 2, 4, 8])

    with pytest.raises(ValueError, match="Invalid keep_k values"):
        validate_keep_ks([1, 3], tokenizer=tokenizer, tokenizer_name="boat")


def test_validate_keep_ks_rejects_tokenizers_without_prefix_decode() -> None:
    tokenizer: Any = _DummyTokenizer(supports_prefix_decode=False)

    with pytest.raises(ValueError, match="does not expose supported prefix decode"):
        validate_keep_ks([1], tokenizer=tokenizer, tokenizer_name="fast")


def test_validate_keep_ks_deduplicates_and_sorts() -> None:
    tokenizer: Any = _DummyTokenizer(block_schedule=[1, 1, 2, 4, 8])

    assert validate_keep_ks([4, 1, 4], tokenizer=tokenizer, tokenizer_name="boat") == [
        1,
        4,
    ]
