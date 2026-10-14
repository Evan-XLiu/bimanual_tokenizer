# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for training logger local output and fallback behavior."""

from __future__ import annotations

import json
import sys
import types

import pytest

from praxis.training.core.logging import Logger


class TestLogger:
    def test_local_jsonl_logging(self, tmp_path):
        log_path = tmp_path / "train_metrics.jsonl"
        logger = Logger(use_wandb=False, local_log_path=str(log_path))
        logger.log({"loss": 1.25, "lr": 1e-4}, step=3)
        logger.finish()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["step"] == 3
        assert record["loss"] == 1.25
        assert record["lr"] == 1e-4

    def test_wandb_init_failure_falls_back_to_local(self, tmp_path, monkeypatch):
        fake_wandb = types.SimpleNamespace(
            init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("wandb down")),
        )
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        log_path = tmp_path / "train_metrics.jsonl"
        logger = Logger(use_wandb=True, local_log_path=str(log_path))
        assert logger.use_wandb is False
        logger.log({"loss": 0.5}, step=1)
        logger.finish()

        record = json.loads(log_path.read_text().strip())
        assert record["step"] == 1
        assert record["loss"] == 0.5

    def test_required_wandb_init_failure_stops_training(self, monkeypatch):
        fake_wandb = types.SimpleNamespace(
            init=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad entity")),
        )
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        with pytest.raises(RuntimeError, match="Required W&B initialization failed: bad entity"):
            Logger(use_wandb=True, wandb_required=True)

    def test_log_videos_noop_without_wandb(self, tmp_path):
        logger = Logger(use_wandb=False, local_log_path=str(tmp_path / "m.jsonl"))
        logger.log_videos([], step=1)

    def test_wandb_init_uses_configured_run_name_and_dir(self, tmp_path, monkeypatch):
        captured: dict[str, object] = {}

        def _fake_init(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(id="fake_run_id")

        fake_wandb = types.SimpleNamespace(
            init=_fake_init,
            log=lambda *args, **kwargs: None,
            finish=lambda: None,
            Video=lambda *args, **kwargs: None,
        )
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        log_path = tmp_path / "train_metrics.jsonl"
        run_dir = tmp_path / "wandb_dir"
        logger = Logger(
            use_wandb=True,
            project="proj",
            entity="team",
            config={"seed": 42},
            run_name="my_train_run",
            run_dir=str(run_dir),
            local_log_path=str(log_path),
        )
        logger.finish()

        assert logger.use_wandb is True
        assert captured["project"] == "proj"
        assert captured["entity"] == "team"
        assert captured["name"] == "my_train_run"
        assert captured["dir"] == str(run_dir)
        assert run_dir.exists()

        # Verify wandb run info sidecar for eval monitors.
        info_path = run_dir / "wandb_run_info.json"
        assert info_path.exists()
        info = json.loads(info_path.read_text())
        assert info["id"] == "fake_run_id"
        assert info["project"] == "proj"
        assert info["entity"] == "team"

    def test_wandb_init_threads_resume_id(self, tmp_path, monkeypatch):
        captured: dict[str, object] = {}

        def _fake_init(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(id="prior_run_id")

        fake_wandb = types.SimpleNamespace(
            init=_fake_init,
            log=lambda *args, **kwargs: None,
            finish=lambda: None,
            Video=lambda *args, **kwargs: None,
        )
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        logger = Logger(
            use_wandb=True,
            project="proj",
            entity="team",
            run_dir=str(tmp_path / "wandb_dir"),
            resume_id="prior_run_id",
        )
        logger.finish()

        assert captured["id"] == "prior_run_id"
        assert captured["resume"] == "allow"

    def test_wandb_init_threads_timeout_setting(self, tmp_path, monkeypatch):
        captured: dict[str, object] = {}

        class _FakeSettings:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        def _fake_init(**kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(id="run_with_timeout")

        fake_wandb = types.SimpleNamespace(
            init=_fake_init,
            log=lambda *args, **kwargs: None,
            finish=lambda: None,
            Video=lambda *args, **kwargs: None,
            Settings=_FakeSettings,
        )
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        logger = Logger(
            use_wandb=True,
            run_dir=str(tmp_path / "wandb_dir"),
            wandb_init_timeout=300,
        )
        logger.finish()

        assert captured["settings"].kwargs == {"init_timeout": 300.0}
