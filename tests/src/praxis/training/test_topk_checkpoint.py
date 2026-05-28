"""Tests for praxis.training.core.checkpoint."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest

from praxis.training.core.checkpoint import (
    RecentCheckpointManager,
    TopKCheckpointManager,
)
from praxis.training.core.trainer import _checkpoint_pool_size


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


def _dummy_save_fn(path: Path) -> None:
    """Simulate saving a checkpoint by creating a checkpoint directory."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker.txt").write_text("fake-ckpt\n")


class TestTopKCheckpointManager:
    """Unit tests for ranking, eviction, and manifest."""

    def test_min_mode_keeps_smallest(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=2, mode="min", metric_key="loss", directory=tmp_dir
        )

        assert mgr.consider(1, {"loss": 0.5}, _dummy_save_fn)
        assert mgr.consider(2, {"loss": 0.3}, _dummy_save_fn)
        assert mgr.consider(3, {"loss": 0.4}, _dummy_save_fn)  # displaces 0.5

        assert mgr.best is not None
        assert mgr.best.value == 0.3
        assert len(mgr._entries) == 2
        values = [e.value for e in mgr._entries]
        assert values == [0.3, 0.4]

    def test_max_mode_keeps_largest(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=2, mode="max", metric_key="score", directory=tmp_dir
        )

        assert mgr.consider(1, {"score": 0.7}, _dummy_save_fn)
        assert mgr.consider(2, {"score": 0.9}, _dummy_save_fn)
        assert mgr.consider(3, {"score": 0.8}, _dummy_save_fn)  # displaces 0.7

        assert mgr.best is not None
        assert mgr.best.value == 0.9
        values = [e.value for e in mgr._entries]
        assert values == [0.9, 0.8]

    def test_replacement_deletes_worst_file(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=1, mode="min", metric_key="loss", directory=tmp_dir
        )

        mgr.consider(1, {"loss": 0.5}, _dummy_save_fn)
        first_path = mgr._entries[0].path
        assert first_path.exists()

        mgr.consider(2, {"loss": 0.3}, _dummy_save_fn)
        assert not first_path.exists(), "displaced checkpoint should be deleted"
        assert mgr._entries[0].path.exists()

    def test_missing_key_skips(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=3, mode="min", metric_key="loss", directory=tmp_dir
        )

        assert not mgr.consider(1, {"accuracy": 0.9}, _dummy_save_fn)
        assert len(mgr._entries) == 0

    def test_nan_metric_skips(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=3, mode="min", metric_key="loss", directory=tmp_dir
        )

        assert not mgr.consider(1, {"loss": float("nan")}, _dummy_save_fn)
        assert not mgr.consider(2, {"loss": math.inf}, _dummy_save_fn)
        assert len(mgr._entries) == 0

    def test_k_equals_one(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=1, mode="max", metric_key="score", directory=tmp_dir
        )

        mgr.consider(1, {"score": 0.5}, _dummy_save_fn)
        mgr.consider(2, {"score": 0.9}, _dummy_save_fn)
        mgr.consider(3, {"score": 0.7}, _dummy_save_fn)  # worse than 0.9, skipped

        assert len(mgr._entries) == 1
        assert mgr.best is not None
        assert mgr.best.value == 0.9
        assert mgr.best.step == 2

    def test_worse_candidate_rejected(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=2, mode="min", metric_key="loss", directory=tmp_dir
        )

        mgr.consider(1, {"loss": 0.1}, _dummy_save_fn)
        mgr.consider(2, {"loss": 0.2}, _dummy_save_fn)
        assert not mgr.consider(3, {"loss": 0.5}, _dummy_save_fn)
        assert len(mgr._entries) == 2

    def test_candidate_path_if_qualified_does_not_mutate(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=1, mode="min", metric_key="loss", directory=tmp_dir
        )

        path = mgr.candidate_path_if_qualified(1, {"loss": 0.5})

        assert path == tmp_dir / "step_1"
        assert mgr._entries == []

    def test_record_saved_checkpoint_updates_manifest(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=1, mode="min", metric_key="loss", directory=tmp_dir
        )
        _dummy_save_fn(tmp_dir / "step_1")

        mgr.record_saved_checkpoint(1, 0.5)

        assert mgr.best is not None
        assert mgr.best.step == 1
        manifest = json.loads((tmp_dir / "topk_manifest.json").read_text())
        assert manifest == [
            {
                "step": 1,
                "metric_key": "loss",
                "mode": "min",
                "value": 0.5,
                "file": "step_1",
            }
        ]

    def test_tie_keeps_earlier_step(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=1, mode="min", metric_key="loss", directory=tmp_dir
        )

        mgr.consider(1, {"loss": 0.5}, _dummy_save_fn)
        # Same value — should NOT displace (earlier step wins)
        assert not mgr.consider(2, {"loss": 0.5}, _dummy_save_fn)
        assert mgr.best is not None
        assert mgr.best.step == 1

    def test_manifest_content(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=3, mode="min", metric_key="loss", directory=tmp_dir
        )

        mgr.consider(1, {"loss": 0.5}, _dummy_save_fn)
        mgr.consider(2, {"loss": 0.3}, _dummy_save_fn)
        mgr.consider(3, {"loss": 0.4}, _dummy_save_fn)

        manifest_path = tmp_dir / "topk_manifest.json"
        assert manifest_path.exists()

        entries = json.loads(manifest_path.read_text())
        assert len(entries) == 3
        # Best first
        assert entries[0]["value"] == 0.3
        assert entries[0]["step"] == 2
        assert entries[0]["metric_key"] == "loss"
        assert entries[0]["mode"] == "min"
        assert entries[1]["value"] == 0.4
        assert entries[2]["value"] == 0.5

    def test_checkpoint_name_uses_step_only(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=1, mode="max", metric_key="success_rate", directory=tmp_dir
        )

        mgr.consider(10, {"success_rate": 0.85}, _dummy_save_fn)
        assert mgr._entries[0].path.name == "step_10"

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            TopKCheckpointManager(
                k=0, mode="min", metric_key="loss", directory="/tmp/x"
            )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode must be"):
            TopKCheckpointManager(
                k=1, mode="avg", metric_key="loss", directory="/tmp/x"
            )

    def test_best_is_none_when_empty(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=3, mode="min", metric_key="loss", directory=tmp_dir
        )
        assert mgr.best is None

    def test_many_candidates_keeps_only_k(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=3, mode="min", metric_key="loss", directory=tmp_dir
        )

        for i in range(10):
            mgr.consider(i, {"loss": float(10 - i)}, _dummy_save_fn)

        assert len(mgr._entries) == 3
        values = [e.value for e in mgr._entries]
        assert values == [1.0, 2.0, 3.0]

        # Only 3 checkpoint directories should remain
        ckpt_dirs = [p for p in tmp_dir.glob("step_*") if p.is_dir()]
        assert len(ckpt_dirs) == 3

    def test_participant_mode_does_not_manage_files(self, tmp_dir: Path):
        mgr = TopKCheckpointManager(
            k=1,
            mode="min",
            metric_key="loss",
            directory=tmp_dir,
            manage_files=False,
        )

        mgr.consider(1, {"loss": 0.5}, _dummy_save_fn)
        first_path = mgr._entries[0].path
        mgr.consider(2, {"loss": 0.3}, _dummy_save_fn)

        assert first_path.exists()
        assert mgr._entries[0].step == 2
        assert not (tmp_dir / "topk_manifest.json").exists()

    def test_manifest_recovery_requires_matching_metric_and_mode(
        self,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        _dummy_save_fn(tmp_dir / "step_1")
        _dummy_save_fn(tmp_dir / "step_2")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "topk_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "step": 1,
                        "metric_key": "loss",
                        "mode": "min",
                        "value": 0.1,
                        "file": "step_1",
                    },
                    {
                        "step": 2,
                        "metric_key": "loss",
                        "mode": "max",
                        "value": 0.9,
                        "file": "step_2",
                    },
                ]
            )
        )

        caplog.set_level(
            logging.WARNING,
            logger="praxis.training.core.checkpoint.managers",
        )
        mgr = TopKCheckpointManager(
            k=2,
            mode="min",
            metric_key="loss",
            directory=tmp_dir,
        )

        assert [entry.step for entry in mgr._entries] == [1]
        assert "Discarded 1 top-K manifest entries" in caplog.text

    def test_stale_reconst_manifest_is_not_reused_for_sim_eval_success_rate(
        self,
        tmp_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ):
        _dummy_save_fn(tmp_dir / "step_1")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "topk_manifest.json").write_text(
            json.dumps(
                [
                    {
                        "step": 1,
                        "metric_key": "val_reconst_loss",
                        "mode": "min",
                        "value": 99.0,
                        "file": "step_1",
                    }
                ]
            )
        )

        caplog.set_level(
            logging.WARNING,
            logger="praxis.training.core.checkpoint.managers",
        )
        mgr = TopKCheckpointManager(
            k=1,
            mode="max",
            metric_key="success_rate",
            directory=tmp_dir,
        )

        assert mgr._entries == []
        assert "Discarded 1 top-K manifest entries" in caplog.text
        assert mgr.consider(2, {"success_rate": 0.0}, _dummy_save_fn)
        assert mgr.best is not None
        assert mgr.best.step == 2


class TestRecentCheckpointManager:
    def test_keeps_most_recent_steps(self, tmp_dir: Path):
        mgr = RecentCheckpointManager(k=2, directory=tmp_dir)

        mgr.consider(1, _dummy_save_fn)
        mgr.consider(2, _dummy_save_fn)
        mgr.consider(3, _dummy_save_fn)

        steps = [e.step for e in mgr._entries]
        assert steps == [3, 2]
        assert not (tmp_dir / "step_1").exists()

    def test_reconsider_same_step_dedups_manifest(self, tmp_dir: Path):
        mgr = RecentCheckpointManager(k=2, directory=tmp_dir)

        mgr.consider(2, _dummy_save_fn)
        mgr.consider(2, _dummy_save_fn)

        manifest = json.loads((tmp_dir / "recent_manifest.json").read_text())
        assert manifest == [{"step": 2, "file": "step_2"}]

    def test_manifest_recovery(self, tmp_dir: Path):
        mgr = RecentCheckpointManager(k=2, directory=tmp_dir)
        mgr.consider(4, _dummy_save_fn)
        mgr.consider(6, _dummy_save_fn)

        recovered = RecentCheckpointManager(k=2, directory=tmp_dir)
        steps = [e.step for e in recovered._entries]
        assert steps == [6, 4]

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError, match="k must be positive"):
            RecentCheckpointManager(k=0, directory="/tmp/x")

    def test_participant_mode_does_not_manage_files(self, tmp_dir: Path):
        mgr = RecentCheckpointManager(k=1, directory=tmp_dir, manage_files=False)

        mgr.consider(1, _dummy_save_fn)
        first_path = mgr._entries[0].path
        mgr.consider(2, _dummy_save_fn)

        assert first_path.exists()
        assert [entry.step for entry in mgr._entries] == [2]
        assert not (tmp_dir / "recent_manifest.json").exists()


@pytest.mark.parametrize("k", [0, -1])
def test_checkpoint_pool_size_warns_and_disables_non_positive_values(
    caplog: pytest.LogCaptureFixture,
    k: int,
) -> None:
    assert _checkpoint_pool_size(k, name="checkpoint_candidates.recent.k") == 0
    assert (
        f"checkpoint_candidates.recent.k={k} disables that checkpoint candidate pool."
        in caplog.text
    )
