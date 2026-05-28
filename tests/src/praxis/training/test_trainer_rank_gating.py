"""Tests for main-rank checkpoint gating in DefaultTrainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from praxis.training.core import (
    run_context as run_context_module,
    trainer as trainer_module,
)
from praxis.training.core.run_context import (
    RunDirectoryManager,
    TrainerRunContext,
    resolve_trainer_run_context,
)
from praxis.training.core.trainer import DefaultTrainer


class _FakeRuntime:
    def __init__(
        self,
        *,
        is_main_process: bool,
        num_processes: int = 1,
        events: list[str] | None = None,
    ) -> None:
        self.is_main_process = is_main_process
        self.num_processes = num_processes
        self.events = events
        self.wait_calls = 0

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1
        if self.events is not None:
            self.events.append("wait")

    @staticmethod
    def broadcast_object(obj: Any, *, src: int = 0) -> Any:
        del src
        return obj


def _make_stub_trainer(
    *, is_main_process: bool, checkpoint_dir: Path, distributed_strategy: str = "ddp"
) -> DefaultTrainer:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    cast(Any, trainer).runtime = _FakeRuntime(is_main_process=is_main_process)
    trainer.tc = OmegaConf.create(
        {
            "checkpoint_dir": str(checkpoint_dir),
            "distributed_strategy": distributed_strategy,
        }
    )
    return trainer


def _trainer_runtime(trainer: DefaultTrainer) -> _FakeRuntime:
    return cast(_FakeRuntime, trainer.runtime)


def _run_dir_cfg(
    output_dir: Path,
    *,
    run_mode: str = "overwrite",
    resume_checkpoint: str | None = None,
):
    return OmegaConf.create(
        {
            "output_dir": str(output_dir),
            "run_name": "test-run",
            "run_mode": run_mode,
            "resume_checkpoint": resume_checkpoint,
        }
    )


def _prepare_run_directory(cfg, runtime: _FakeRuntime) -> TrainerRunContext:
    context = resolve_trainer_run_context(cfg)
    RunDirectoryManager().prepare(
        context=context,
        cfg=cfg,
        runtime=runtime,
    )
    return context


class _MinimalDefaultTrainer(DefaultTrainer):
    def _init_data(self) -> None:
        self.dataloader = DataLoader(
            TensorDataset(torch.zeros(1, 1)),
            batch_size=1,
        )
        self.val_dataloader = None

    def _init_model(self) -> None:
        self.model = torch.nn.Identity()


def _minimal_default_trainer_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "run_name": "minimal",
            "trainer": {
                "batch_size": 1,
                "val_batch_size": None,
                "mixed_precision": "no",
                "gradient_accumulation_steps": 1,
                "find_unused_parameters": False,
                "distributed_strategy": "ddp",
                "fsdp_min_num_params": 100000000,
                "fsdp_reshard_after_forward": True,
                "fsdp_cpu_offload": False,
                "lr": 1e-4,
                "weight_decay": 0.0,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "lr_scheduler": "constant",
                "warmup_steps": 0,
                "max_steps": 1,
                "min_lr_ratio": 0.0,
                "max_grad_norm": 1.0,
                "max_val_batches": None,
                "use_ema": False,
                "ema_decay": 0.0,
                "use_wandb": True,
                "wandb_project": "praxis-test",
                "wandb_entity": None,
                "local_log_path": str(tmp_path / "metrics.jsonl"),
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "checkpoint_candidates": {
                    "best": {"metric_key": "val_loss", "mode": "min", "k": 1},
                    "recent": {"k": 1},
                },
            },
        }
    )


def test_resolve_trainer_run_context_fresh_with_output_dir(tmp_path: Path) -> None:
    cfg = _run_dir_cfg(tmp_path / "run")

    context = resolve_trainer_run_context(cfg)

    assert context.output_dir == tmp_path / "run"
    assert context.run_mode == "overwrite"
    assert context.resume_checkpoint is None
    assert context.has_output_dir
    assert context.owns_output_dir
    assert context.completion_sentinel_path() == tmp_path / "run" / "TRAINING_COMPLETE"


def test_resolve_trainer_run_context_resume_does_not_own_output_dir(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "latest"
    cfg = _run_dir_cfg(
        tmp_path / "run",
        run_mode="resume",
        resume_checkpoint=str(checkpoint),
    )

    context = resolve_trainer_run_context(cfg)

    assert context.output_dir == tmp_path / "run"
    assert context.run_mode == "resume"
    assert context.resume_checkpoint == str(checkpoint)
    assert not context.owns_output_dir


def test_resolve_trainer_run_context_resume_requires_checkpoint(
    tmp_path: Path,
) -> None:
    cfg = _run_dir_cfg(tmp_path / "run", run_mode="resume")

    with pytest.raises(ValueError, match="run_mode=resume requires resume_checkpoint"):
        resolve_trainer_run_context(cfg)


def test_resolve_trainer_run_context_overwrite_rejects_resume_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "latest"
    cfg = _run_dir_cfg(tmp_path / "run", resume_checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="run_mode=resume"):
        resolve_trainer_run_context(cfg)


def test_resolve_trainer_run_context_rejects_unknown_mode(tmp_path: Path) -> None:
    cfg = _run_dir_cfg(tmp_path / "run", run_mode="append")

    with pytest.raises(ValueError, match="Unsupported run_mode"):
        resolve_trainer_run_context(cfg)


def test_run_context_require_output_dir_raises_without_output_dir() -> None:
    context = resolve_trainer_run_context(OmegaConf.create({"run_name": "minimal"}))

    with pytest.raises(RuntimeError, match="required for inline sim-eval artifacts"):
        context.require_output_dir(purpose="inline sim-eval artifacts")


def test_default_trainer_init_missing_root_output_dir_sets_wandb_run_dir_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _CapturingLogger:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(trainer_module, "Logger", _CapturingLogger)

    trainer = _MinimalDefaultTrainer(_minimal_default_trainer_cfg(tmp_path))

    assert trainer.run_context.output_dir is None
    assert captured["run_dir"] is None


def test_default_trainer_prepares_no_train_loop_model_without_training_wrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {}

    class _NullLogger:
        def __init__(self, **_kwargs) -> None:
            return

    original_prepare_model = trainer_module.TrainingRuntime.prepare_model

    def _capture_prepare_model(self, model, *, wrap_for_training: bool = True):
        captured["wrap_for_training"] = wrap_for_training
        return original_prepare_model(
            self,
            model,
            wrap_for_training=wrap_for_training,
        )

    monkeypatch.setattr(trainer_module, "Logger", _NullLogger)
    monkeypatch.setattr(
        trainer_module.TrainingRuntime,
        "prepare_model",
        _capture_prepare_model,
    )

    trainer = _MinimalDefaultTrainer(_minimal_default_trainer_cfg(tmp_path))

    assert trainer.run_train_loop is False
    assert captured["wrap_for_training"] is False


def test_default_trainer_init_prepares_fresh_output_dir_before_logger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("stale\n")
    captured: dict[str, Any] = {}

    class _CapturingLogger:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            assert not (output_dir / "stale.txt").exists()
            assert (output_dir / "train_config.yaml").exists()

    monkeypatch.setattr(trainer_module, "Logger", _CapturingLogger)
    cfg = _minimal_default_trainer_cfg(tmp_path)
    cfg.output_dir = str(output_dir)

    trainer = _MinimalDefaultTrainer(cfg)

    assert trainer.run_context.output_dir == output_dir
    assert captured["run_dir"] == str(output_dir)


def test_save_checkpoint_only_main_process_writes(tmp_path: Path) -> None:
    trainer = _make_stub_trainer(
        is_main_process=True, checkpoint_dir=tmp_path / "ckpts"
    )
    calls: list[Path] = []

    trainer._save_checkpoint_dir = lambda path, **_: calls.append(Path(path))  # type: ignore[method-assign]
    target = tmp_path / "manual" / "step_1"

    trainer.save_checkpoint(target)

    assert _trainer_runtime(trainer).wait_calls == 2
    assert calls == [target]


def test_save_checkpoint_non_main_process_skips_write(tmp_path: Path) -> None:
    trainer = _make_stub_trainer(
        is_main_process=False, checkpoint_dir=tmp_path / "ckpts"
    )
    calls: list[Path] = []

    trainer._save_checkpoint_dir = lambda path, **_: calls.append(Path(path))  # type: ignore[method-assign]

    trainer.save_checkpoint(tmp_path / "manual" / "step_1")

    assert _trainer_runtime(trainer).wait_calls == 2
    assert calls == []


def test_save_checkpoint_fsdp_writes_on_all_ranks(tmp_path: Path) -> None:
    trainer = _make_stub_trainer(
        is_main_process=False,
        checkpoint_dir=tmp_path / "ckpts",
        distributed_strategy="fsdp",
    )
    calls: list[Path] = []

    trainer._save_checkpoint_dir = lambda path, **_: calls.append(Path(path))  # type: ignore[method-assign]
    target = tmp_path / "manual" / "step_1"

    trainer.save_checkpoint(target)

    assert _trainer_runtime(trainer).wait_calls == 2
    assert calls == [target]


def test_save_latest_checkpoint_only_main_process_writes(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = _make_stub_trainer(is_main_process=True, checkpoint_dir=checkpoint_dir)
    calls: list[Path] = []

    trainer._save_checkpoint_dir = lambda path, **_: calls.append(Path(path))  # type: ignore[method-assign]

    trainer.save_latest_checkpoint()

    assert _trainer_runtime(trainer).wait_calls == 2
    assert calls == [checkpoint_dir / "latest"]


def test_save_latest_checkpoint_non_main_process_skips_write(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = _make_stub_trainer(is_main_process=False, checkpoint_dir=checkpoint_dir)
    calls: list[Path] = []

    trainer._save_checkpoint_dir = lambda path, **_: calls.append(Path(path))  # type: ignore[method-assign]

    trainer.save_latest_checkpoint()

    assert _trainer_runtime(trainer).wait_calls == 2
    assert calls == []


def test_save_latest_checkpoint_fsdp_writes_on_all_ranks(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = _make_stub_trainer(
        is_main_process=False,
        checkpoint_dir=checkpoint_dir,
        distributed_strategy="fsdp",
    )
    calls: list[Path] = []

    trainer._save_checkpoint_dir = lambda path, **_: calls.append(Path(path))  # type: ignore[method-assign]

    trainer.save_latest_checkpoint()

    assert _trainer_runtime(trainer).wait_calls == 2
    assert calls == [checkpoint_dir / "latest"]


def test_init_model_with_download_sync_main_rank_runs_before_barriers() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    events: list[str] = []
    cast(Any, trainer).runtime = _FakeRuntime(
        is_main_process=True, num_processes=4, events=events
    )
    trainer._init_model = lambda: events.append("init")  # type: ignore[method-assign]

    trainer._init_model_with_download_sync()

    assert events == ["init", "wait", "wait"]
    assert _trainer_runtime(trainer).wait_calls == 2


def test_init_model_with_download_sync_non_main_rank_waits_before_init() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    events: list[str] = []
    cast(Any, trainer).runtime = _FakeRuntime(
        is_main_process=False, num_processes=4, events=events
    )
    trainer._init_model = lambda: events.append("init")  # type: ignore[method-assign]

    trainer._init_model_with_download_sync()

    assert events == ["wait", "init", "wait"]
    assert _trainer_runtime(trainer).wait_calls == 2


def test_init_model_with_download_sync_single_process_has_no_barrier() -> None:
    trainer = DefaultTrainer.__new__(DefaultTrainer)
    events: list[str] = []
    cast(Any, trainer).runtime = _FakeRuntime(
        is_main_process=True, num_processes=1, events=events
    )
    trainer._init_model = lambda: events.append("init")  # type: ignore[method-assign]

    trainer._init_model_with_download_sync()

    assert events == ["init"]
    assert _trainer_runtime(trainer).wait_calls == 0


def test_prepare_output_dir_for_fresh_run_overwrites_existing_dir(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_file = output_dir / "stale.txt"
    stale_file.write_text("stale\n")
    runtime = _FakeRuntime(is_main_process=True)
    cfg = _run_dir_cfg(output_dir)

    context = _prepare_run_directory(cfg, runtime)

    assert context.owns_output_dir
    assert output_dir.is_dir()
    assert not stale_file.exists()
    assert (output_dir / "train_config.yaml").read_text() == OmegaConf.to_yaml(cfg)
    assert runtime.wait_calls == 2


def test_prepare_output_dir_moves_stale_dir_aside_when_rmtree_is_flaky(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_file = output_dir / "stale.txt"
    stale_file.write_text("stale\n")
    real_rmtree = run_context_module.shutil.rmtree
    original_failures = 0

    def flaky_rmtree(path: str | Path) -> None:
        nonlocal original_failures
        target = Path(path)
        if target == output_dir and original_failures < 3:
            original_failures += 1
            raise OSError(39, "Directory not empty")
        real_rmtree(path)

    monkeypatch.setattr(run_context_module.shutil, "rmtree", flaky_rmtree)
    runtime = _FakeRuntime(is_main_process=True)
    cfg = _run_dir_cfg(output_dir)

    _prepare_run_directory(cfg, runtime)

    assert original_failures == 3
    assert output_dir.is_dir()
    assert not stale_file.exists()
    assert list((tmp_path / ".trainer_trash").iterdir()) == []
    assert runtime.wait_calls == 2


def test_prepare_output_dir_for_resume_preserves_existing_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_file = output_dir / "train_metrics.jsonl"
    stale_file.write_text("keep\n")
    snapshot = output_dir / "train_config.yaml"
    snapshot.write_text("old snapshot\n")
    runtime = _FakeRuntime(is_main_process=True)
    cfg = _run_dir_cfg(
        output_dir,
        run_mode="resume",
        resume_checkpoint=str(output_dir / "checkpoints" / "latest"),
    )

    context = _prepare_run_directory(cfg, runtime)

    assert not context.owns_output_dir
    assert stale_file.read_text() == "keep\n"
    assert snapshot.read_text() == "old snapshot\n"
    assert runtime.wait_calls == 0


def test_prepare_output_dir_non_main_waits_without_writing(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    runtime = _FakeRuntime(is_main_process=False)
    cfg = _run_dir_cfg(output_dir)

    _prepare_run_directory(cfg, runtime)

    assert not output_dir.exists()
    assert runtime.wait_calls == 2


def test_prepare_output_dir_replaces_existing_file(tmp_path: Path) -> None:
    output_dir = tmp_path / "run"
    output_dir.write_text("stale file\n")
    runtime = _FakeRuntime(is_main_process=True)
    cfg = _run_dir_cfg(output_dir)

    _prepare_run_directory(cfg, runtime)

    assert output_dir.is_dir()
    assert (output_dir / "train_config.yaml").exists()


def test_prepare_output_dir_replaces_existing_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output_dir = tmp_path / "run"
    output_dir.symlink_to(target, target_is_directory=True)
    runtime = _FakeRuntime(is_main_process=True)
    cfg = _run_dir_cfg(output_dir)

    _prepare_run_directory(cfg, runtime)

    assert output_dir.is_dir()
    assert not output_dir.is_symlink()
    assert target.is_dir()


def test_prepare_output_dir_refuses_filesystem_root() -> None:
    cfg = _run_dir_cfg(Path("/"))
    runtime = _FakeRuntime(is_main_process=True)

    with pytest.raises(ValueError, match="Refusing to overwrite filesystem root"):
        _prepare_run_directory(cfg, runtime)


def test_run_directory_manager_noops_without_output_dir() -> None:
    cfg = OmegaConf.create({"run_name": "minimal-unit-test-config"})
    runtime = _FakeRuntime(is_main_process=True)

    context = _prepare_run_directory(cfg, runtime)

    assert context.output_dir is None
    assert runtime.wait_calls == 0
