# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from praxis.policies.eval_adapter import EnvironmentActionBoundsPolicyAdapter
from praxis.scripts import eval as eval_script


def _eval_cfg(*, mode: str, server_address: str | None = None):
    return OmegaConf.create(
        {
            "seed": 42,
            "device": "cuda",
            "server_address": server_address,
            "policy": {"name": "dummy"},
            "env": {
                "type": "libero",
                "task": "libero_10",
                "task_ids": None,
                "python_bin": None,
            },
            "eval": {
                "mode": mode,
                "num_eval_per_task": 1,
                "num_parallel_env": 1,
                "output_dir": "eval/out",
                "record_episodes_per_task": 0,
                "step_timeout_sec": 0,
                "rollout_failure_retries": 1,
                "phase_watchdog_sec": None,
                "debug_verbose": False,
                "policy_kwargs": {
                    "decode_keep_k": None,
                    "decode_temperature": None,
                    "decode_top_k": None,
                },
            },
        }
    )


class _FakeRemotePolicy:
    def __init__(self, address: str) -> None:
        self.address = address
        self.client = SimpleNamespace(health_check=lambda: (True, {"address": address}))
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeLocalPolicy:
    training = True
    io = SimpleNamespace(image_features=[])

    def __init__(self) -> None:
        self.closed = False
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def eval(self) -> None:
        self.training = False

    def close(self) -> None:
        self.closed = True


def _install_fake_evaluator(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    policies: list[object] = []

    def fake_evaluate(_env_type, *, policy, config):
        policies.append(policy)
        assert config.output_dir == "eval/out"
        return SimpleNamespace(overall={"success_rate": 1.0}, artifacts={})

    monkeypatch.setattr(eval_script.praxis_eval, "evaluate", fake_evaluate)
    return policies


def test_remote_eval_wraps_policy_without_local_device_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies = _install_fake_evaluator(monkeypatch)
    monkeypatch.setattr(eval_script, "RemotePolicy", _FakeRemotePolicy)
    monkeypatch.setattr(
        eval_script,
        "require_available_device",
        lambda *_args, **_kwargs: pytest.fail(
            "remote eval must not validate local CUDA"
        ),
    )

    eval_script.main.__wrapped__(
        _eval_cfg(mode="remote_grpc", server_address="127.0.0.1:50051")
    )

    assert len(policies) == 1
    assert isinstance(policies[0], EnvironmentActionBoundsPolicyAdapter)
    assert policies[0].policy.address == "127.0.0.1:50051"


def test_local_eval_validates_device_before_policy_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policies = _install_fake_evaluator(monkeypatch)
    calls: list[tuple[object, str]] = []

    def fake_require_available_device(requested_device, *, purpose: str):
        calls.append((requested_device, purpose))
        return "cpu"

    monkeypatch.setattr(
        eval_script, "require_available_device", fake_require_available_device
    )
    monkeypatch.setattr(
        "praxis.policies.factory.make_policy",
        lambda _cfg: _FakeLocalPolicy(),
    )

    eval_script.main.__wrapped__(_eval_cfg(mode="local_inproc"))

    assert calls == [("cuda", "Evaluation")]
    assert len(policies) == 1


def test_local_eval_rejects_server_address() -> None:
    with pytest.raises(ValueError, match="eval.mode=remote_grpc"):
        eval_script.main.__wrapped__(
            _eval_cfg(mode="local_inproc", server_address="127.0.0.1:50051")
        )
