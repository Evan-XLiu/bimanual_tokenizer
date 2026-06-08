# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for policy normalization persistence."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import Tensor

from praxis.core.types import FeatureSpec, FeatureType, NormMode, PolicyIO
from praxis.policies.core import BasePolicy


def _toy_policy_io(action_key: str = "action") -> PolicyIO:
    return PolicyIO(
        obs_features=[
            FeatureSpec(
                key="observation.state",
                type=FeatureType.STATE,
                shape=(2,),
                norm_mode=NormMode.MIN_MAX,
            )
        ],
        action_feature=FeatureSpec(
            key=action_key,
            type=FeatureType.ACTION,
            shape=(3,),
            norm_mode=NormMode.MIN_MAX,
        ),
        n_obs_steps=1,
        n_action_steps=1,
        fps=10,
    )


class _ToyPolicy(BasePolicy):
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        del batch
        z = torch.tensor(0.0)
        return z, {"loss": float(z.item())}

    def predict_action(self, obs: dict[str, Tensor]) -> Tensor:
        del obs
        return torch.zeros(3)


def _save_norm_stats(policy: BasePolicy, path: Path) -> None:
    stats: dict[str, dict[str, Tensor]] = {}
    modes: dict[str, str] = {}
    if policy.normalize is not None:
        for key in policy.normalize.keys:
            if key in policy.normalize.stats:
                stats[key] = policy.normalize.stats[key]
            if key in policy.normalize.modes:
                modes[key] = policy.normalize.modes[key].value
    if policy.unnormalize_action is not None:
        for key in policy.unnormalize_action.keys:
            if key not in stats and key in policy.unnormalize_action.stats:
                stats[key] = policy.unnormalize_action.stats[key]
            if key not in modes and key in policy.unnormalize_action.modes:
                modes[key] = policy.unnormalize_action.modes[key].value
    torch.save({"stats": stats, "modes": modes}, path)


def _load_norm_stats(
    path: Path,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, NormMode]]:
    data = torch.load(path, map_location="cpu", weights_only=True)
    stats = data["stats"]
    modes = {k: NormMode(v) for k, v in data["modes"].items()}
    return stats, modes


class TestPolicyNormPersistence:
    def test_policy_norm_save_load_via_file(self):
        policy = _ToyPolicy()
        policy.set_io(_toy_policy_io())
        stats = {
            "observation.state": {
                "min": torch.tensor([-1.0, -2.0]),
                "max": torch.tensor([1.0, 2.0]),
            },
            "action": {
                "min": torch.tensor([0.0, -1.0, 0.5]),
                "max": torch.tensor([10.0, 1.0, 5.0]),
            },
        }
        policy.set_normalization_stats(stats)

        sample = {
            "observation.state": torch.tensor([[0.25, -0.25]], dtype=torch.float32)
        }
        out_ref = policy.normalize(sample)["observation.state"]

        with tempfile.TemporaryDirectory() as tmpdir:
            norm_path = Path(tmpdir) / "norm_stats.pt"
            _save_norm_stats(policy, norm_path)
            loaded_stats, loaded_modes = _load_norm_stats(norm_path)

        loaded = _ToyPolicy()
        loaded.set_io(_toy_policy_io())
        loaded.set_normalization_stats(loaded_stats, modes=loaded_modes)

        out_loaded = loaded.normalize(sample)["observation.state"]
        torch.testing.assert_close(out_ref, out_loaded)

    def test_policy_norm_save_load_with_custom_action_key(self):
        policy = _ToyPolicy()
        policy.set_io(_toy_policy_io(action_key="act"))
        stats = {
            "observation.state": {
                "min": torch.tensor([-1.0, -2.0]),
                "max": torch.tensor([1.0, 2.0]),
            },
            "act": {
                "min": torch.tensor([0.0, -1.0, 0.5]),
                "max": torch.tensor([10.0, 1.0, 5.0]),
            },
        }
        policy.set_normalization_stats(stats)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm_path = Path(tmpdir) / "norm_stats.pt"
            _save_norm_stats(policy, norm_path)
            loaded_stats, loaded_modes = _load_norm_stats(norm_path)

        loaded = _ToyPolicy()
        loaded.set_io(_toy_policy_io(action_key="act"))
        loaded.set_normalization_stats(loaded_stats, modes=loaded_modes)
        out = loaded.unnormalize_action({"act": torch.zeros(1, 3)})["act"]
        assert out.shape == (1, 3)

    def test_policy_honors_explicit_modes(self):
        policy = _ToyPolicy()
        policy.set_io(_toy_policy_io())
        stats = {
            "observation.state": {
                "min": torch.tensor([-1.0, -2.0]),
                "max": torch.tensor([1.0, 2.0]),
                "mean": torch.tensor([0.5, -0.5]),
                "std": torch.tensor([0.25, 0.5]),
            },
            "action": {
                "min": torch.tensor([0.0, 0.0, 0.0]),
                "max": torch.tensor([10.0, 10.0, 10.0]),
                "mean": torch.tensor([5.0, 4.0, 3.0]),
                "std": torch.tensor([1.0, 2.0, 4.0]),
            },
        }
        modes = {
            "observation.state": NormMode.MEAN_STD,
            "action": NormMode.MEAN_STD,
        }
        policy.set_normalization_stats(stats, modes=modes)

        obs = {"observation.state": torch.tensor([[1.0, 0.0]], dtype=torch.float32)}
        normalized_obs = policy.normalize(obs)["observation.state"]
        expected_obs = (
            obs["observation.state"] - stats["observation.state"]["mean"]
        ) / stats["observation.state"]["std"]
        torch.testing.assert_close(normalized_obs, expected_obs, atol=1e-5, rtol=1e-5)

        with tempfile.TemporaryDirectory() as tmpdir:
            norm_path = Path(tmpdir) / "norm_stats.pt"
            _save_norm_stats(policy, norm_path)
            loaded_stats, loaded_modes = _load_norm_stats(norm_path)

        loaded = _ToyPolicy()
        loaded.set_io(_toy_policy_io())
        loaded.set_normalization_stats(loaded_stats, modes=loaded_modes)

        norm_loaded = loaded.normalize(obs)["observation.state"]
        torch.testing.assert_close(norm_loaded, normalized_obs, atol=1e-5, rtol=1e-5)
