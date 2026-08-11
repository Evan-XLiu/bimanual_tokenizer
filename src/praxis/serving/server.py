# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Policy server backed by the standalone praxis-remote transport."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from praxis_remote import PolicyServer as RemotePolicyServer
from praxis_remote.serialization import ObservationValue, PolicyKwargValue

from praxis.policies.core import BasePolicy
from praxis.policies.eval_adapter import PolicyEvalAdapter
from praxis.serving.model_identity import build_policy_serving_identity


class _PolicyServerHandler:
    """Adapt a ``BasePolicy`` to the generic praxis-remote handler API."""

    def __init__(
        self,
        policy: BasePolicy,
        device: str = "cuda",
        model_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.adapter = PolicyEvalAdapter(policy, device=device)
        self.policy = self.adapter.policy
        self.device = torch.device(device)
        self.policy.eval()
        self._model_metadata = {
            "schema_version": 1,
            "policy_class": self.policy.__class__.__name__,
            "device": str(self.device),
            **dict(model_metadata or {}),
        }

    def predict_action(
        self,
        observations: Sequence[Mapping[str, ObservationValue]],
        *,
        policy_kwargs: Mapping[str, PolicyKwargValue] | None = None,
        episode_ids: Sequence[str] | None = None,
    ) -> np.ndarray:
        return self.adapter.act(
            observations,
            policy_kwargs=policy_kwargs,
            episode_ids=episode_ids,
        )

    def reset(self, episode_ids: Sequence[str] | None = None) -> None:
        self.adapter.reset(episode_ids=episode_ids)

    def model_info(self) -> str:
        return json.dumps(self._model_metadata, sort_keys=True)


class PolicyServer:
    """Manages the gRPC server lifecycle for a Praxis ``BasePolicy``."""

    def __init__(
        self,
        policy: BasePolicy,
        device: str = "cuda",
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 4,
        checkpoint_path: str | Path | None = None,
        serving_identity: Mapping[str, Any] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        model_metadata: dict[str, Any] = {}
        if checkpoint_path is not None and serving_identity is not None:
            raise ValueError("Pass checkpoint_path or serving_identity, not both.")
        if checkpoint_path is not None:
            model_metadata["serving_artifact"] = build_policy_serving_identity(
                checkpoint_path
            )
        elif serving_identity is not None:
            model_metadata["serving_artifact"] = dict(serving_identity)
        if source_metadata is not None:
            model_metadata["source"] = dict(source_metadata)
        self.handler = _PolicyServerHandler(policy, device, model_metadata)
        self.server = RemotePolicyServer(
            self.handler,
            host=host,
            port=port,
            max_workers=max_workers,
        )

    def serve(self, block: bool = True) -> None:
        """Start the server.

        Args:
            block: If True (default), blocks until server is stopped.
        """
        self.server.serve(block=block)

    def stop(self, grace: float = 0) -> None:
        """Stop the server."""
        self.server.stop(grace=grace)
