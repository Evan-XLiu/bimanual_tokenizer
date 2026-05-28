"""Parity-focused tests for Praxis gRPC policy serving."""

from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import torch
from praxis_eval.contracts import ActionSpec
from praxis_eval.policies.remote import RemotePolicy
from praxis_remote import PolicyClient
from praxis_remote.serialization import PolicyKwargValue, build_observation

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.core import BasePolicy
from praxis.policies.eval_adapter import (
    EnvironmentActionBoundsPolicyAdapter,
    PolicyEvalAdapter,
)
from praxis.serving.server import PolicyServer


class DummyParityPolicy(BasePolicy):
    """Deterministic policy that exercises task, images, kwargs, and reset state."""

    def __init__(self) -> None:
        super().__init__()
        self.set_io(
            PolicyIO(
                obs_features=[
                    FeatureSpec(
                        key="observation.state",
                        type=FeatureType.STATE,
                        shape=(4,),
                        norm_mode=NormMode.NONE,
                    ),
                    FeatureSpec(
                        key="observation.images.primary",
                        type=FeatureType.IMAGE,
                        shape=(3, 2, 2),
                        norm_mode=NormMode.NONE,
                    ),
                    FeatureSpec(
                        key="observation.images.wrist",
                        type=FeatureType.IMAGE,
                        shape=(3, 2, 2),
                        norm_mode=NormMode.NONE,
                    ),
                ],
                action_feature=FeatureSpec(
                    key="action",
                    type=FeatureType.ACTION,
                    shape=(8,),
                    norm_mode=NormMode.NONE,
                ),
                n_obs_steps=1,
                n_action_steps=1,
                fps=10,
                language_key="task",
            )
        )
        self.predict_calls = 0
        self.reset_calls = 0
        self._step_index = 0

    def forward(self, batch):
        return torch.tensor(0.0), {}

    def predict_action(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self.predict_calls += 1

        decode_keep_k = kwargs.get("decode_keep_k")
        decode_temperature = kwargs.get("decode_temperature")
        decode_strict = kwargs.get("decode_strict")

        if decode_keep_k is not None and not isinstance(decode_keep_k, int):
            raise TypeError("decode_keep_k must remain an int across remote transport.")
        if decode_temperature is not None and not isinstance(decode_temperature, float):
            raise TypeError(
                "decode_temperature must remain a float across remote transport."
            )
        if decode_strict is not None and not isinstance(decode_strict, bool):
            raise TypeError("decode_strict must remain a bool across remote transport.")

        state = _as_batched_tensor(obs["observation.state"])
        primary = _as_batched_tensor(obs["observation.images.primary"])
        wrist = _as_batched_tensor(obs["observation.images.wrist"])
        tasks = _as_batched_tasks(obs.get("task"), batch_size=int(state.shape[0]))

        action = torch.zeros(
            (state.shape[0], 8), dtype=torch.float32, device=state.device
        )
        action[:, 0] = state.sum(dim=1)
        action[:, 1] = primary.reshape(primary.shape[0], -1).sum(dim=1)
        action[:, 2] = wrist.reshape(wrist.shape[0], -1).sum(dim=1)
        action[:, 3] = torch.tensor(
            [float(len(task)) for task in tasks],
            dtype=torch.float32,
            device=state.device,
        )
        action[:, 4] = float(decode_keep_k or 0)
        action[:, 5] = float(decode_temperature or 0.0)
        action[:, 6] = 1.0 if decode_strict else 0.0
        action[:, 7] = float(self._step_index)

        self._step_index += 1
        return action.squeeze(0) if int(action.shape[0]) == 1 else action

    def reset(self) -> None:
        super().reset()
        self.reset_calls += 1
        self._step_index = 0


class _OverlapDetectingPolicy(DummyParityPolicy):
    """Stateful test policy that records overlapping predict/reset entry."""

    def __init__(self) -> None:
        super().__init__()
        self.first_predict_entered = threading.Event()
        self.overlap_detected = False
        self._active_call: str | None = None
        self._num_entries = 0
        self._state_guard = threading.Lock()

    def _enter_policy_call(self, call_name: str) -> None:
        with self._state_guard:
            if self._active_call is not None:
                self.overlap_detected = True
            self._active_call = call_name
            self._num_entries += 1
            if self._num_entries == 1:
                self.first_predict_entered.set()
        time.sleep(0.1)

    def _exit_policy_call(self) -> None:
        with self._state_guard:
            self._active_call = None

    def predict_action(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        self._enter_policy_call("predict")
        try:
            return super().predict_action(obs, **kwargs)
        finally:
            self._exit_policy_call()

    def reset(self) -> None:
        self._enter_policy_call("reset")
        try:
            super().reset()
        finally:
            self._exit_policy_call()


class _ChunkQueuePolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.set_io(
            PolicyIO(
                obs_features=[
                    FeatureSpec(
                        key="observation.state",
                        type=FeatureType.STATE,
                        shape=(4,),
                        norm_mode=NormMode.NONE,
                    ),
                ],
                action_feature=FeatureSpec(
                    key="action",
                    type=FeatureType.ACTION,
                    shape=(1,),
                    norm_mode=NormMode.NONE,
                ),
                n_obs_steps=1,
                n_action_steps=2,
                fps=10,
                language_key=None,
            )
        )
        self.config = type("ChunkConfig", (), {"n_obs_steps": 1, "n_action_steps": 2})()
        self._configure_action_queue(n_action_steps=2)

    def forward(self, batch):
        del batch
        return torch.tensor(0.0), {}

    def predict_action_chunk(self, obs: dict[str, torch.Tensor], **kwargs):
        del kwargs
        state = obs["observation.state"]
        base = state[:, -1, :1]
        return torch.stack((base, base + 100.0), dim=1), {}

    def predict_action(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return self._predict_action_from_chunk_queue(
            obs,
            predict_action_chunk=self.predict_action_chunk,
            **kwargs,
        )


def _as_batched_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.dim() in {1, 3}:
        return value.unsqueeze(0)
    return value


def _as_batched_tasks(value: Any, *, batch_size: int) -> list[str]:
    if value is None:
        return [""] * batch_size
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _build_observation(example_index: int) -> dict[str, np.ndarray | str]:
    return {
        "observation.state": np.full((4,), example_index + 1, dtype=np.float32),
        "observation.images.primary": np.full(
            (3, 2, 2), 0.1 * (example_index + 1), dtype=np.float32
        ),
        "observation.images.wrist": np.full(
            (3, 2, 2), 0.2 * (example_index + 1), dtype=np.float32
        ),
        "task": f"task-{example_index}",
    }


def _policy_kwargs() -> dict[str, Any]:
    return {
        "decode_keep_k": 3,
        "decode_temperature": 0.25,
        "decode_strict": True,
    }


class _LocalParityClient:
    """Test-only local shim matching the remote client surface."""

    def __init__(self, policy: BasePolicy, device: str = "cpu") -> None:
        self.policy = policy
        self.adapter = PolicyEvalAdapter(policy=policy, device=device)

    @torch.inference_mode()
    def predict_action_from_parts(
        self,
        state: np.ndarray | None = None,
        images: dict[str, np.ndarray] | None = None,
        task_description: str | None = None,
        *,
        policy_kwargs: dict[str, PolicyKwargValue] | None = None,
    ) -> np.ndarray:
        observation = build_observation(
            state=state,
            images=images,
            task_description=task_description,
        )
        return self.predict_action([observation], policy_kwargs=policy_kwargs)

    @torch.inference_mode()
    def predict_action(
        self,
        observations: list[dict[str, np.ndarray | str]],
        *,
        policy_kwargs: dict[str, PolicyKwargValue] | None = None,
    ) -> np.ndarray:
        return self.adapter.act(observations, policy_kwargs=policy_kwargs)

    def reset(self) -> bool:
        self.adapter.reset()
        return True


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server_ready(client: PolicyClient) -> None:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        ready, _ = client.health_check()
        if ready:
            return
        time.sleep(0.05)
    pytest.fail("Timed out waiting for gRPC policy server to become ready.")


def _remote_predict_once(port: int) -> np.ndarray:
    client = PolicyClient(host="127.0.0.1", port=port)
    try:
        return client.predict_action(
            [_build_observation(0)],
            policy_kwargs=_policy_kwargs(),
        )
    finally:
        client.close()


def _remote_reset_once(port: int) -> bool:
    client = PolicyClient(host="127.0.0.1", port=port)
    try:
        return bool(client.reset())
    finally:
        client.close()


@pytest.fixture
def serving_stack():
    remote_policy = DummyParityPolicy()
    local_policy = DummyParityPolicy()

    port = _get_free_port()
    server = PolicyServer(
        policy=remote_policy,
        device="cpu",
        host="127.0.0.1",
        port=port,
        max_workers=2,
    )
    server.serve(block=False)

    client = PolicyClient(host="127.0.0.1", port=port)
    _wait_for_server_ready(client)

    local_client = _LocalParityClient(local_policy, device="cpu")
    yield {
        "server": server,
        "remote_policy": remote_policy,
        "local_policy": local_policy,
        "client": client,
        "local_client": local_client,
    }

    client.close()
    server.stop(grace=0)


class TestPolicyClients:
    def test_health_check(self, serving_stack):
        ready, info = serving_stack["client"].health_check()
        assert ready is True
        assert "DummyParityPolicy" in info

    def test_single_example_remote_matches_local(self, serving_stack):
        remote_client: PolicyClient = serving_stack["client"]
        local_client: _LocalParityClient = serving_stack["local_client"]

        images = {
            "observation.images.primary": np.full((3, 2, 2), 0.1, dtype=np.float32),
            "observation.images.wrist": np.full((3, 2, 2), 0.2, dtype=np.float32),
        }
        state = np.ones((4,), dtype=np.float32)

        remote_action = remote_client.predict_action_from_parts(
            state=state,
            images=images,
            task_description="task-0",
            policy_kwargs=_policy_kwargs(),
        )
        local_action = local_client.predict_action_from_parts(
            state=state,
            images=images,
            task_description="task-0",
            policy_kwargs=_policy_kwargs(),
        )

        np.testing.assert_allclose(remote_action, local_action)
        np.testing.assert_allclose(
            remote_action,
            np.array(
                [[4.0, 1.2, 2.4, 6.0, 3.0, 0.25, 1.0, 0.0]],
                dtype=np.float32,
            ),
        )

    def test_batched_predict_action_matches_local_and_is_single_policy_call(
        self, serving_stack
    ):
        remote_client: PolicyClient = serving_stack["client"]
        local_client: _LocalParityClient = serving_stack["local_client"]
        remote_policy: DummyParityPolicy = serving_stack["remote_policy"]

        observations = [_build_observation(0), _build_observation(1)]

        remote_action = remote_client.predict_action(
            observations,
            policy_kwargs=_policy_kwargs(),
        )
        local_action = local_client.predict_action(
            observations,
            policy_kwargs=_policy_kwargs(),
        )

        np.testing.assert_allclose(remote_action, local_action)
        np.testing.assert_allclose(
            remote_action,
            np.array(
                [
                    [4.0, 1.2, 2.4, 6.0, 3.0, 0.25, 1.0, 0.0],
                    [8.0, 2.4, 4.8, 6.0, 3.0, 0.25, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        assert remote_policy.predict_calls == 1

    def test_reset_semantics_match_local(self, serving_stack):
        remote_client: PolicyClient = serving_stack["client"]
        local_client: _LocalParityClient = serving_stack["local_client"]
        remote_policy: DummyParityPolicy = serving_stack["remote_policy"]

        observation = _build_observation(0)
        first_remote = remote_client.predict_action(
            [observation],
            policy_kwargs=_policy_kwargs(),
        )
        first_local = local_client.predict_action(
            [observation],
            policy_kwargs=_policy_kwargs(),
        )
        second_remote = remote_client.predict_action(
            [observation],
            policy_kwargs=_policy_kwargs(),
        )
        second_local = local_client.predict_action(
            [observation],
            policy_kwargs=_policy_kwargs(),
        )

        assert remote_client.reset() is True
        assert local_client.reset() is True

        after_reset_remote = remote_client.predict_action(
            [observation],
            policy_kwargs=_policy_kwargs(),
        )
        after_reset_local = local_client.predict_action(
            [observation],
            policy_kwargs=_policy_kwargs(),
        )

        np.testing.assert_allclose(first_remote, first_local)
        np.testing.assert_allclose(second_remote, second_local)
        np.testing.assert_allclose(after_reset_remote, after_reset_local)
        assert float(first_remote[0, -1]) == pytest.approx(0.0)
        assert float(second_remote[0, -1]) == pytest.approx(1.0)
        assert float(after_reset_remote[0, -1]) == pytest.approx(0.0)
        assert remote_policy.reset_calls == 1

    def test_episode_ids_are_accepted_for_vector_eval(self, serving_stack):
        remote_client: PolicyClient = serving_stack["client"]
        observations = [_build_observation(0), _build_observation(1)]

        action = remote_client.predict_action(
            observations,
            policy_kwargs=_policy_kwargs(),
            episode_ids=["slot-0", "slot-1"],
        )
        assert tuple(action.shape) == (2, 8)

        assert remote_client.reset(episode_ids=["slot-0"]) is True

    def test_policy_adapter_uses_eval_mode(self):
        policy = _ChunkQueuePolicy()
        policy.train()

        PolicyEvalAdapter(policy=policy, device="cpu")

        assert policy.training is False

    def test_episode_ids_preserve_chunk_state_and_partial_reset(self):
        adapter = PolicyEvalAdapter(policy=_ChunkQueuePolicy(), device="cpu")

        first = adapter.act(
            [_build_observation(0), _build_observation(1)],
            episode_ids=["lane-a", "lane-b"],
        )
        np.testing.assert_allclose(first, np.asarray([[1.0], [2.0]], dtype=np.float32))

        adapter.reset(episode_ids=["lane-a"])
        second = adapter.act(
            [_build_observation(2), _build_observation(3)],
            episode_ids=["lane-a", "lane-b"],
        )

        np.testing.assert_allclose(
            second,
            np.asarray([[3.0], [102.0]], dtype=np.float32),
        )

    def test_concurrent_predict_calls_are_serialized_against_shared_policy(self):
        policy = _OverlapDetectingPolicy()
        port = _get_free_port()
        server = PolicyServer(
            policy=policy,
            device="cpu",
            host="127.0.0.1",
            port=port,
            max_workers=2,
        )
        server.serve(block=False)
        health_client = PolicyClient(host="127.0.0.1", port=port)
        try:
            _wait_for_server_ready(health_client)
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(_remote_predict_once, port)
                assert policy.first_predict_entered.wait(timeout=2.0)
                second = executor.submit(_remote_predict_once, port)
                first.result(timeout=5.0)
                second.result(timeout=5.0)

            assert policy.overlap_detected is False
            assert policy.predict_calls == 2
        finally:
            health_client.close()
            server.stop(grace=0)

    def test_concurrent_predict_and_reset_are_serialized_against_shared_policy(self):
        policy = _OverlapDetectingPolicy()
        port = _get_free_port()
        server = PolicyServer(
            policy=policy,
            device="cpu",
            host="127.0.0.1",
            port=port,
            max_workers=2,
        )
        server.serve(block=False)
        health_client = PolicyClient(host="127.0.0.1", port=port)
        try:
            _wait_for_server_ready(health_client)
            with ThreadPoolExecutor(max_workers=2) as executor:
                predict = executor.submit(_remote_predict_once, port)
                assert policy.first_predict_entered.wait(timeout=2.0)
                reset = executor.submit(_remote_reset_once, port)
                predict.result(timeout=5.0)
                assert reset.result(timeout=5.0) is True

            assert policy.overlap_detected is False
            assert policy.predict_calls == 1
            assert policy.reset_calls == 1
        finally:
            health_client.close()
            server.stop(grace=0)


class TestPolicyAdapters:
    def test_remote_eval_policy_matches_local_eval_policy_on_batched_input(
        self, serving_stack
    ):
        remote_policy: DummyParityPolicy = serving_stack["remote_policy"]
        local_policy: DummyParityPolicy = serving_stack["local_policy"]
        remote_eval_policy = RemotePolicy(
            f"127.0.0.1:{serving_stack['server'].port}",
        )
        local_eval_policy = PolicyEvalAdapter(
            policy=local_policy,
            device="cpu",
        )
        observations = [_build_observation(0), _build_observation(1)]

        try:
            remote_action = remote_eval_policy.act(
                observations,
                policy_kwargs=_policy_kwargs(),
            )
            local_action = local_eval_policy.act(
                observations,
                policy_kwargs=_policy_kwargs(),
            )
        finally:
            remote_eval_policy.close()

        np.testing.assert_allclose(remote_action, local_action)
        assert tuple(remote_action.shape) == (2, 8)
        assert remote_policy.predict_calls == 1
        assert local_policy.predict_calls == 1

    def test_remote_eval_policy_reset_forwards_to_client(self, serving_stack):
        remote_policy: DummyParityPolicy = serving_stack["remote_policy"]
        remote_eval_policy = RemotePolicy(f"127.0.0.1:{serving_stack['server'].port}")

        try:
            _ = remote_eval_policy.act([_build_observation(0)])
            remote_eval_policy.reset()
            _ = remote_eval_policy.act([_build_observation(0)])
        finally:
            remote_eval_policy.close()

        assert remote_policy.reset_calls == 1

    def test_local_eval_policy_validates_action_spec(self):
        local_eval_policy = PolicyEvalAdapter(
            policy=DummyParityPolicy(),
            device="cpu",
        )

        with pytest.raises(ValueError, match="Batched action shape"):
            local_eval_policy.act(
                [_build_observation(0)],
                action_spec=ActionSpec(shape=(7,), dtype="float32"),
            )

    def test_local_eval_policy_leaves_action_bounds_to_eval_env(self):
        local_eval_policy = PolicyEvalAdapter(
            policy=DummyParityPolicy(),
            device="cpu",
        )

        action = local_eval_policy.act(
            [_build_observation(0)],
            action_spec=ActionSpec(
                shape=(8,),
                dtype="float32",
                minimum=-1.0,
                maximum=1.0,
            ),
        )

        assert tuple(action.shape) == (1, 8)
        assert np.max(action) > 1.0

    def test_environment_action_bounds_adapter_matches_local_bound_semantics(self):
        class _BoundCheckingPolicy:
            def __init__(self) -> None:
                self.seen_action_spec: ActionSpec | None = None

            def act(
                self,
                observations,
                *,
                action_spec=None,
                policy_kwargs=None,
                episode_ids=None,
            ):
                del observations, policy_kwargs, episode_ids
                self.seen_action_spec = action_spec
                return np.asarray([[2.0] * 8], dtype=np.float32)

        remote_policy = _BoundCheckingPolicy()
        adapter = EnvironmentActionBoundsPolicyAdapter(remote_policy)

        action = adapter.act(
            [_build_observation(0)],
            action_spec=ActionSpec(
                shape=(8,),
                dtype="float32",
                minimum=-1.0,
                maximum=1.0,
            ),
        )

        assert tuple(action.shape) == (1, 8)
        assert remote_policy.seen_action_spec is not None
        assert remote_policy.seen_action_spec.minimum is None
        assert remote_policy.seen_action_spec.maximum is None
