# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from praxis.policies.bar.core.vlm.base import VlmActionTokenPolicyBase
from praxis.training import inline_policy_eval
from praxis.training.core.run_context import TrainerRunContext
from praxis.training.policy_trainer import PolicyTrainer


class _StatsPolicy:
    def __init__(self) -> None:
        self.stats_calls: list[tuple[dict, dict | None]] = []

    def set_normalization_stats(self, stats, modes=None) -> None:
        self.stats_calls.append((stats, modes))


class _NonVlmPolicy(nn.Module):
    pass


def test_init_model_sets_stats(monkeypatch) -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    policy = _StatsPolicy()
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create({"policy": {"name": "bar_boat_pali"}})
    stub.io = object()
    stub.dataset = SimpleNamespace(stats={"x": 1}, normalization_modes=None)

    monkeypatch.setattr(
        "praxis.training.policy_trainer.make_policy",
        lambda cfg, io: policy,
    )

    trainer._init_model()

    assert stub.model is policy
    assert policy.stats_calls == [({"x": 1}, None)]


def test_policy_fsdp_rejects_non_vlm_policy() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create({"trainer": {"distributed_strategy": "fsdp"}})
    stub.tc = stub.cfg.trainer
    stub.model = _NonVlmPolicy()

    with pytest.raises(ValueError, match="validate support"):
        trainer._validate_model_for_distributed_strategy()


def test_policy_trainer_rejects_ddp_ki_warmstart_without_unused_param_detection() -> (
    None
):
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.tc = SimpleNamespace(
        distributed_strategy="ddp",
        find_unused_parameters=False,
    )
    stub.model = SimpleNamespace(config=SimpleNamespace(ce_warmstart_steps=20000))

    with pytest.raises(ValueError, match="find_unused_parameters=true"):
        trainer._validate_model_for_distributed_strategy()


def test_policy_fsdp_allows_vlm_policy_with_wrap_hook() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create({"trainer": {"distributed_strategy": "fsdp"}})
    stub.tc = stub.cfg.trainer
    policy = cast(
        VlmActionTokenPolicyBase,
        VlmActionTokenPolicyBase.__new__(VlmActionTokenPolicyBase),
    )
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(use_lora=False, freeze_action_tokenizer=True)
    policy.vlm = nn.Module()
    policy.vlm.layers = nn.ModuleList([nn.Linear(2, 2)])
    stub.model = policy

    trainer._validate_model_for_distributed_strategy()


def test_resolve_sim_eval_backend_prefers_pool_then_custom_driver(monkeypatch) -> None:
    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.available_drivers",
        lambda: ("libero", "robocasa", "mshab", "simpler"),
    )
    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.env_type_from_cfg",
        lambda env_cfg: str(env_cfg["type"]).strip().lower(),
    )

    assert (
        inline_policy_eval.resolve_backend({"type": "libero"})
        == inline_policy_eval.EVALUATOR_BACKEND
    )
    assert (
        inline_policy_eval.resolve_backend({"type": "robocasa"})
        == inline_policy_eval.EVALUATOR_BACKEND
    )
    assert (
        inline_policy_eval.resolve_backend({"type": "simpler"})
        == inline_policy_eval.EVALUATOR_BACKEND
    )
    assert (
        inline_policy_eval.resolve_backend({"type": "mshab"})
        == inline_policy_eval.EVALUATOR_BACKEND
    )


class _DummyRuntime:
    def __init__(self, *, is_main_process: bool = True) -> None:
        self.is_main_process = is_main_process
        self.global_rank = 0 if is_main_process else 1
        self.wait_calls = 0
        self.broadcast_inputs: list[tuple[object, int]] = []
        self.gather_inputs: list[object] = []

    def wait_for_everyone(self) -> None:
        self.wait_calls += 1

    def broadcast_object(self, obj, *, src: int = 0):
        self.broadcast_inputs.append((obj, src))
        return obj

    def gather_object(self, obj) -> list[object]:
        self.gather_inputs.append(obj)
        return [obj]

    def unwrap_model(self, model):
        return model


class _DummyLogger:
    def __init__(self) -> None:
        self.videos: list[tuple[list[str], int, str]] = []

    def log_videos(self, paths: list[str], *, step: int, key: str) -> None:
        self.videos.append((paths, step, key))


class _ModeTrackingPolicy:
    def __init__(self) -> None:
        self.training = True
        self.eval_calls = 0
        self.train_calls = 0

    def eval(self) -> None:
        self.training = False
        self.eval_calls += 1

    def train(self, mode: bool = True) -> _ModeTrackingPolicy:
        self.training = bool(mode)
        self.train_calls += 1
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def reset(self) -> None:
        return None

    def predict_action(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        del kwargs
        state = obs.get("observation.state")
        batch_size = int(state.shape[0]) if isinstance(state, torch.Tensor) else 1
        return torch.zeros((batch_size, 2), dtype=torch.float32)

    def consume_inference_info(self) -> list[dict[str, Any]]:
        return []


class _DummyTopKManager:
    def __init__(
        self, *, metric_key: str = "val_reconst_loss", mode: str = "min"
    ) -> None:
        self.metric_key = metric_key
        self.mode = mode


def _minimal_policy_trainer_cfg(
    *, env: dict[str, Any] | None, sim_eval_every: int | None = 10000
) -> DictConfig:
    return OmegaConf.create(
        {
            "trainer": {
                "val_every": 1000,
                "reconst_every": 1000,
                "max_reconst_batches": 1,
                "sim_eval_every": sim_eval_every,
                "sim_eval_start_after_step": 0,
                "distributed_strategy": "ddp",
                "sim_eval": {
                    "num_eval_per_task": 10,
                    "num_parallel_env": 40,
                    "record_episodes_per_task": 0,
                    "step_timeout_sec": None,
                    "policy_kwargs": {},
                },
                "wandb_upload_eval_videos": False,
                "wandb_max_eval_videos": 0,
                "checkpoint_candidates": {
                    "best": {
                        "metric_key": "val_reconst_loss",
                        "mode": "min",
                    }
                },
            },
            "policy": {},
            "env": env,
        }
    )


def _stub_default_trainer_init(
    monkeypatch: pytest.MonkeyPatch, *, topk_manager: _DummyTopKManager
) -> None:
    def fake_init(self, cfg) -> None:
        self.cfg = cfg
        self.tc = cfg.trainer
        self.runtime = _DummyRuntime()
        self._configure_checkpoint_candidates_before_managers()
        best_cfg = cfg.trainer.checkpoint_candidates.best
        topk_manager.metric_key = str(best_cfg.metric_key)
        topk_manager.mode = str(best_cfg.mode)
        self.topk_manager = topk_manager

    monkeypatch.setattr(
        "praxis.training.policy_trainer.DefaultTrainer.__init__",
        fake_init,
    )


def test_policy_trainer_requires_env_when_sim_eval_interval_is_configured(
    monkeypatch,
) -> None:
    topk = _DummyTopKManager()
    _stub_default_trainer_init(monkeypatch, topk_manager=topk)

    with pytest.raises(ValueError, match="cfg.env.type is null or empty"):
        PolicyTrainer(_minimal_policy_trainer_cfg(env={"type": None, "task": None}))

    assert topk.metric_key == "val_reconst_loss"
    assert topk.mode == "min"


def test_policy_trainer_pre_init_validation_rejects_bad_sim_eval_config(
    monkeypatch,
) -> None:
    def fail_base_init(self, cfg) -> None:
        del self, cfg
        raise AssertionError("PolicyTrainer config validation should run first")

    monkeypatch.setattr(
        "praxis.training.policy_trainer.DefaultTrainer.__init__",
        fail_base_init,
    )

    with pytest.raises(ValueError, match="cfg.env.type is null or empty"):
        PolicyTrainer(_minimal_policy_trainer_cfg(env={"type": None, "task": None}))


def test_policy_trainer_rejects_fsdp_inline_sim_eval(monkeypatch) -> None:
    topk = _DummyTopKManager()
    _stub_default_trainer_init(monkeypatch, topk_manager=topk)
    cfg = _minimal_policy_trainer_cfg(env={"type": "simpler", "task": "bridge"})
    cfg.trainer.distributed_strategy = "fsdp"

    with pytest.raises(ValueError, match="sim_eval_every is not supported"):
        PolicyTrainer(cfg)

    assert topk.metric_key == "val_reconst_loss"
    assert topk.mode == "min"


def test_policy_trainer_pre_init_validation_rejects_fsdp_sim_eval(
    monkeypatch,
) -> None:
    def fail_base_init(self, cfg) -> None:
        del self, cfg
        raise AssertionError("PolicyTrainer config validation should run first")

    monkeypatch.setattr(
        "praxis.training.policy_trainer.DefaultTrainer.__init__",
        fail_base_init,
    )
    cfg = _minimal_policy_trainer_cfg(env={"type": "simpler", "task": "bridge"})
    cfg.trainer.distributed_strategy = "fsdp"

    with pytest.raises(ValueError, match="sim_eval_every is not supported"):
        PolicyTrainer(cfg)


def test_policy_trainer_allows_empty_env_when_sim_eval_is_disabled(monkeypatch) -> None:
    topk = _DummyTopKManager()
    _stub_default_trainer_init(monkeypatch, topk_manager=topk)

    trainer = PolicyTrainer(
        _minimal_policy_trainer_cfg(
            env={"type": None, "task": None}, sim_eval_every=None
        )
    )

    assert trainer.sim_eval_backend is None
    assert topk.metric_key == "val_reconst_loss"
    assert topk.mode == "min"
    assert trainer._periodic_eval_hooks()[-1][0] is None


def test_policy_trainer_uses_success_rate_topk_when_sim_eval_backend_exists(
    monkeypatch,
) -> None:
    topk = _DummyTopKManager()
    _stub_default_trainer_init(monkeypatch, topk_manager=topk)
    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.resolve_backend",
        lambda _env_cfg: inline_policy_eval.EVALUATOR_BACKEND,
    )

    trainer = PolicyTrainer(
        _minimal_policy_trainer_cfg(env={"type": "simpler", "task": "bridge"})
    )

    assert trainer.sim_eval_backend == inline_policy_eval.EVALUATOR_BACKEND
    assert topk.metric_key == "success_rate"
    assert topk.mode == "max"
    assert trainer._periodic_eval_hooks()[-1][0] == 10000


def test_policy_trainer_skips_sim_eval_until_start_step(monkeypatch) -> None:
    topk = _DummyTopKManager()
    _stub_default_trainer_init(monkeypatch, topk_manager=topk)
    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.resolve_backend",
        lambda _env_cfg: inline_policy_eval.EVALUATOR_BACKEND,
    )
    cfg = _minimal_policy_trainer_cfg(env={"type": "simpler", "task": "bridge"})
    cfg.trainer.sim_eval_start_after_step = 2

    trainer = PolicyTrainer(cfg)
    trainer.global_step = 2
    assert trainer._periodic_eval_hooks()[-1][0] is None

    trainer.global_step = 3
    assert trainer._periodic_eval_hooks()[-1][0] == 10000


def test_policy_trainer_resolves_sim_eval_runtime_kwargs(monkeypatch) -> None:
    topk = _DummyTopKManager()
    _stub_default_trainer_init(monkeypatch, topk_manager=topk)
    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.resolve_backend",
        lambda _env_cfg: inline_policy_eval.EVALUATOR_BACKEND,
    )
    cfg = _minimal_policy_trainer_cfg(env={"type": "simpler", "task": "bridge"})
    cfg.trainer.sim_eval.step_timeout_sec = 12.5
    cfg.trainer.sim_eval.policy_kwargs = {
        "decode_keep_k": 8,
        "debug_force_stop_k": None,
    }

    trainer = PolicyTrainer(cfg)

    assert trainer.sim_eval_step_timeout_sec == 12.5
    assert trainer.sim_eval_policy_kwargs == {"decode_keep_k": 8}


def test_rank0_sim_eval_ema_scope_does_not_enter_collectives() -> None:
    class _NoGatherRuntime(_DummyRuntime):
        def gather_object(self, obj):
            del obj
            raise AssertionError("rank-0-only sim eval must not gather EMA state")

    class _EMA:
        def __init__(self) -> None:
            self.backup: dict[str, object] = {}
            self.apply_calls = 0
            self.restore_calls = 0

        @property
        def restore_needed(self) -> bool:
            return bool(self.backup)

        def apply(self, model, *, backup_device=None) -> None:
            assert backup_device is None
            assert model is policy
            self.backup = {"weight": object()}
            self.apply_calls += 1

        def restore(self, model) -> None:
            assert model is policy
            self.backup.clear()
            self.restore_calls += 1

    policy = _ModeTrackingPolicy()
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.runtime = _NoGatherRuntime()
    stub.model = policy
    stub.ema = _EMA()

    with trainer._rank0_ema_eval_scope() as eval_policy:
        assert eval_policy is policy

    assert stub.ema.apply_calls == 1
    assert stub.ema.restore_calls == 1


def test_sim_eval_driver_backend_runs_eval_driver(
    monkeypatch,
    tmp_path: Path,
) -> None:
    policy = _ModeTrackingPolicy()
    runtime = _DummyRuntime()
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create(
        {
            "env": {"type": "simpler", "python_bin": "/tmp/simpler-python"},
            "output_dir": str(tmp_path / "wrong_cfg_output_dir"),
            "seed": 123,
        }
    )
    stub.run_context = TrainerRunContext(output_dir=tmp_path, resume_checkpoint=None)
    stub.global_step = 5
    stub.sim_eval_backend = inline_policy_eval.EVALUATOR_BACKEND
    stub.sim_eval_num_eval_per_task = 1
    stub.sim_eval_num_parallel_env = 1
    stub.sim_eval_record_episodes_per_task = 0
    stub.sim_eval_step_timeout_sec = 12.5
    stub.sim_eval_rollout_failure_retries = 1
    stub.sim_eval_policy_kwargs = {"decode_keep_k": 8}
    stub.device = torch.device("cpu")
    stub.model_preprocessor = object()
    stub.runtime = runtime
    stub.wandb_upload_eval_videos = False
    stub.wandb_max_eval_videos = 0
    stub.logger = _DummyLogger()
    stub.model = policy
    stub.ema = None

    captured: dict[str, Any] = {}
    inference_depth = 0
    inference_entries = 0
    cleanup_devices: list[torch.device] = []

    class _InferenceModeTracker:
        def __enter__(self) -> None:
            nonlocal inference_depth, inference_entries
            inference_depth += 1
            inference_entries += 1

        def __exit__(self, exc_type, exc, tb) -> None:
            nonlocal inference_depth
            del exc_type, exc, tb
            inference_depth -= 1

    monkeypatch.setattr(
        torch,
        "inference_mode",
        lambda: _InferenceModeTracker(),
    )
    monkeypatch.setattr(
        "praxis.training.core.eval_memory.release_cuda_eval_memory",
        lambda device: cleanup_devices.append(torch.device(device)),
    )

    def fake_evaluate(env, *, policy, config):
        assert inference_depth == 1
        captured["env"] = env
        captured["policy"] = policy
        captured["config"] = config
        return inline_policy_eval.praxis_eval.EvalResult(
            overall={
                "success_rate": 0.25,
                "avg_episode_length": 12.0,
                "video_paths": [],
            },
            per_task={},
        )

    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.praxis_eval.evaluate",
        fake_evaluate,
    )

    metrics = trainer.sim_eval()

    assert metrics == {"success_rate": 0.25, "avg_episode_length": 12.0}
    assert policy.eval_calls == 1
    assert policy.train_calls == 1
    assert policy.training is True
    assert inference_entries == 1
    assert cleanup_devices == [torch.device("cpu")]
    assert runtime.wait_calls == 2
    config = captured["config"]
    assert captured["env"] == "simpler"
    assert captured["policy"].policy is policy
    assert config.step_timeout_sec == 12.5
    assert dict(config.policy_kwargs) == {"decode_keep_k": 8}
    assert Path(config.output_dir) == tmp_path / "eval" / "by_step" / "step_5"
    assert config.metadata["step"] == 5
    assert (
        tmp_path / "eval" / "by_step" / "step_5" / inline_policy_eval.DONE_MARKER
    ).exists()


def test_sim_eval_offloads_optimizer_state_and_uses_cpu_ema_backup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    policy = _ModeTrackingPolicy()
    runtime = _DummyRuntime()
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create(
        {
            "env": {"type": "simpler", "python_bin": "/tmp/simpler-python"},
            "seed": 123,
        }
    )
    stub.run_context = TrainerRunContext(output_dir=tmp_path, resume_checkpoint=None)
    stub.global_step = 5
    stub.sim_eval_backend = inline_policy_eval.EVALUATOR_BACKEND
    stub.sim_eval_num_eval_per_task = 1
    stub.sim_eval_num_parallel_env = 1
    stub.sim_eval_record_episodes_per_task = 0
    stub.sim_eval_step_timeout_sec = None
    stub.sim_eval_rollout_failure_retries = 1
    stub.sim_eval_policy_kwargs = {}
    stub.device = torch.device("cpu")
    stub.model_preprocessor = object()
    stub.runtime = runtime
    stub.wandb_upload_eval_videos = False
    stub.wandb_max_eval_videos = 0
    stub.logger = _DummyLogger()
    stub.model = policy
    stub.optimizer = object()

    events: list[str] = []

    class _Offload:
        moved_tensors = 2
        moved_bytes = 4096

        @property
        def moved_gib(self) -> float:
            return self.moved_bytes / (1024**3)

        def restore(self) -> None:
            events.append("restore_optimizer")

    class _EMA:
        def __init__(self) -> None:
            self.backup: dict[str, object] = {}

        @property
        def restore_needed(self) -> bool:
            return bool(self.backup)

        def apply(self, model, *, backup_device=None) -> None:
            assert model is policy
            assert backup_device == torch.device("cpu")
            self.backup = {"weight": object()}
            events.append("apply_ema")

        def restore(self, model) -> None:
            assert model is policy
            self.backup.clear()
            events.append("restore_ema")

    stub.ema = _EMA()

    def fake_offload_optimizer_state(optimizer):
        assert optimizer is stub.optimizer
        events.append("offload_optimizer")
        return _Offload()

    monkeypatch.setattr(
        "praxis.training.core.eval_memory.offload_optimizer_state_to_cpu",
        fake_offload_optimizer_state,
    )
    monkeypatch.setattr(
        "praxis.training.core.eval_memory.release_cuda_eval_memory",
        lambda device: None,
    )

    def fake_evaluate(*_args, **_kwargs):
        events.append("eval")
        return inline_policy_eval.praxis_eval.EvalResult(
            overall={
                "success_rate": 0.25,
                "avg_episode_length": 12.0,
                "video_paths": [],
            },
            per_task={},
        )

    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.praxis_eval.evaluate",
        fake_evaluate,
    )

    assert trainer.sim_eval() == {"success_rate": 0.25, "avg_episode_length": 12.0}
    assert events == [
        "offload_optimizer",
        "apply_ema",
        "eval",
        "restore_ema",
        "restore_optimizer",
    ]


def test_sim_eval_cleanup_error_is_gathered_before_final_barrier(
    monkeypatch,
    tmp_path: Path,
) -> None:
    policy = _ModeTrackingPolicy()
    runtime = _DummyRuntime()
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create(
        {
            "env": {"type": "simpler", "python_bin": "/tmp/simpler-python"},
            "seed": 123,
        }
    )
    stub.run_context = TrainerRunContext(output_dir=tmp_path, resume_checkpoint=None)
    stub.global_step = 5
    stub.sim_eval_backend = inline_policy_eval.EVALUATOR_BACKEND
    stub.sim_eval_num_eval_per_task = 1
    stub.sim_eval_num_parallel_env = 1
    stub.sim_eval_record_episodes_per_task = 0
    stub.sim_eval_step_timeout_sec = None
    stub.sim_eval_rollout_failure_retries = 1
    stub.sim_eval_policy_kwargs = {}
    stub.device = torch.device("cpu")
    stub.model_preprocessor = object()
    stub.runtime = runtime
    stub.wandb_upload_eval_videos = False
    stub.wandb_max_eval_videos = 0
    stub.logger = _DummyLogger()
    stub.model = policy
    stub.ema = None

    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.praxis_eval.evaluate",
        lambda *_args, **_kwargs: inline_policy_eval.praxis_eval.EvalResult(
            overall={
                "success_rate": 0.25,
                "avg_episode_length": 12.0,
                "video_paths": [],
            },
            per_task={},
        ),
    )

    def fail_cleanup(device) -> None:
        del device
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        "praxis.training.core.eval_memory.release_cuda_eval_memory",
        fail_cleanup,
    )

    with pytest.raises(RuntimeError, match="Sim-eval cleanup failed on rank 0"):
        trainer.sim_eval()

    assert runtime.wait_calls == 1
    assert "cleanup failed" in str(runtime.gather_inputs[-1])
    assert all(payload is None for payload, _src in runtime.broadcast_inputs[1:])


def test_sim_eval_driver_backend_non_main_waits_on_rank0_marker(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime = _DummyRuntime(is_main_process=False)
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create(
        {
            "env": {"type": "simpler", "python_bin": "/tmp/simpler-python"},
            "output_dir": str(tmp_path),
            "seed": 123,
        }
    )
    stub.run_context = TrainerRunContext(output_dir=tmp_path, resume_checkpoint=None)
    stub.global_step = 5
    stub.sim_eval_backend = inline_policy_eval.EVALUATOR_BACKEND
    stub.runtime = runtime
    stub.model = _ModeTrackingPolicy()

    captured: dict[str, Path] = {}

    def fake_wait_for_marker(*, done_marker, error_marker):
        captured["done_marker"] = done_marker
        captured["error_marker"] = error_marker

    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.wait_for_rank0_marker",
        fake_wait_for_marker,
    )

    def fail_evaluate(*_args, **_kwargs):
        raise AssertionError("non-main ranks must not run inline sim eval")

    monkeypatch.setattr(
        "praxis.training.inline_policy_eval.praxis_eval.evaluate",
        fail_evaluate,
    )

    assert trainer.sim_eval() == {}
    assert runtime.wait_calls == 2
    assert (
        captured["done_marker"]
        == tmp_path / "eval" / "by_step" / "step_5" / inline_policy_eval.DONE_MARKER
    )


def test_sim_eval_requires_output_dir_before_rank_marker_wait() -> None:
    runtime = _DummyRuntime(is_main_process=False)
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.cfg = OmegaConf.create(
        {"env": {"type": "simpler", "python_bin": "/tmp/simpler-python"}, "seed": 123}
    )
    stub.run_context = TrainerRunContext(output_dir=None, resume_checkpoint=None)
    stub.global_step = 5
    stub.sim_eval_backend = inline_policy_eval.EVALUATOR_BACKEND
    stub.runtime = runtime

    with pytest.raises(RuntimeError, match="required for inline sim-eval artifacts"):
        trainer.sim_eval()

    assert runtime.wait_calls == 0


def test_sim_eval_rejects_fsdp_even_when_called_directly() -> None:
    trainer = PolicyTrainer.__new__(PolicyTrainer)
    stub = cast(Any, trainer)
    stub.sim_eval_backend = inline_policy_eval.EVALUATOR_BACKEND
    stub.tc = SimpleNamespace(distributed_strategy="fsdp")

    with pytest.raises(RuntimeError, match="Inline sim-eval is not supported"):
        trainer.sim_eval()
