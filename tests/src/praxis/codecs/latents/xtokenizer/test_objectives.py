# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for X-Tokenizer objective functions."""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from praxis.codecs.latents.xtokenizer.objectives import (
    _gather_distributed_pairs,
    masked_action_l1,
    rotation_geodesic_loss,
    symmetric_global_local_infonce,
)
from praxis.core.types import NormMode


def _distributed_infonce_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        basis = torch.eye(3)
        if rank == 0:
            values = torch.stack((basis[0], torch.zeros(3))).reshape(2, 1, 3)
            valid_mask = torch.tensor([[True], [False]])
        else:
            values = torch.stack((basis[1], basis[2])).reshape(2, 1, 3)
            valid_mask = torch.ones(2, 1, dtype=torch.bool)
        action_features = values.clone().requires_grad_(True)
        vl_features = values.clone().requires_grad_(True)
        logit_scale = torch.tensor(math.log(10.0), requires_grad=True)
        gathered_action, gathered_vl = _gather_distributed_pairs(
            action_features[valid_mask],
            vl_features[valid_mask],
        )

        global_loss, local_loss = symmetric_global_local_infonce(
            action_features,
            vl_features,
            valid_mask,
            logit_scale,
        )
        (global_loss + local_loss).backward()
        torch.save(
            {
                "global_loss": global_loss,
                "local_loss": local_loss,
                "action_grad": action_features.grad,
                "vl_grad": vl_features.grad,
                "scale_grad": logit_scale.grad,
                "gathered_action": gathered_action,
                "gathered_vl": gathered_vl,
            },
            Path(output_dir) / f"infonce-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_masked_action_l1_excludes_padded_values_before_arithmetic() -> None:
    reconstruction = torch.tensor(
        [[[1.0, 3.0], [float("nan"), float("inf")]]],
        requires_grad=True,
    )
    target = torch.tensor([[[0.0, 1.0], [float("nan"), float("-inf")]]])
    valid_mask = torch.tensor([[True, False]])

    loss = masked_action_l1(reconstruction, target, valid_mask)

    assert loss.item() == pytest.approx(1.5)
    loss.backward()
    assert reconstruction.grad is not None
    assert torch.equal(
        reconstruction.grad,
        torch.tensor([[[0.5, 0.5], [0.0, 0.0]]]),
    )


def test_masked_action_l1_applies_per_dof_validity() -> None:
    reconstruction = torch.tensor([[[1.0, 1000.0], [3.0, 2000.0]]])
    target = torch.zeros_like(reconstruction)
    dof_valid = torch.tensor([[[True, False], [True, False]]])

    loss = masked_action_l1(reconstruction, target, dof_valid)

    assert loss.item() == pytest.approx(2.0)


def test_rotation_geodesic_denormalizes_rotation_and_respects_mask() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    mean = torch.arange(10, dtype=torch.float32) / 10.0
    std = torch.linspace(0.5, 1.4, 10)

    target_physical = torch.zeros(1, 2, 10)
    reconstruction_physical = torch.zeros_like(target_physical)
    target_physical[:, :, 3:9] = identity
    reconstruction_physical[:, 0, 3:9] = z_90
    reconstruction_physical[:, 1, 3:9] = float("nan")
    target_physical[:, 1, 3:9] = float("nan")
    target = (target_physical - mean) / std
    reconstruction = (reconstruction_physical - mean) / std

    loss = rotation_geodesic_loss(
        reconstruction,
        target,
        {"mean": mean, "std": std},
        torch.tensor([[True, False]]),
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.5, abs=2e-4)


def test_rotation_geodesic_requires_all_rotation_dofs_to_be_valid() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    target = identity.reshape(1, 1, 6)
    reconstruction = z_90.reshape(1, 1, 6)
    dof_valid = torch.tensor([[[True, True, True, True, True, False]]])

    loss = rotation_geodesic_loss(reconstruction, target, {}, dof_valid)

    assert loss.item() == 0.0


def test_rotation_geodesic_uses_selected_min_max_stats_when_all_pairs_exist() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    minimum = -torch.ones(10)
    maximum = torch.ones(10)
    mean = torch.zeros(10)
    mean[3:9] = identity
    std = torch.full((10,), 0.1)

    target = torch.zeros(1, 1, 10)
    reconstruction = torch.zeros_like(target)
    target[..., 3:9] = identity
    reconstruction[..., 3:9] = z_90

    loss = rotation_geodesic_loss(
        reconstruction,
        target,
        {
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
        },
        torch.ones(1, 1, dtype=torch.bool),
        normalization_mode=NormMode.MIN_MAX,
    )

    assert loss.item() == pytest.approx(0.5, abs=2e-4)


def test_rotation_geodesic_uses_explicit_named_rotation_slice() -> None:
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    z_90 = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    target = torch.zeros(1, 1, 10)
    reconstruction = torch.zeros_like(target)
    target[..., 4:10] = identity
    reconstruction[..., 4:10] = z_90
    reconstruction[..., 3] = 9.0

    loss = rotation_geodesic_loss(
        reconstruction,
        target,
        {},
        torch.ones(1, 1, dtype=torch.bool),
        rotation_slice=slice(4, 10),
    )

    assert loss.item() == pytest.approx(0.5, abs=2e-4)


def test_rotation_geodesic_is_finite_for_degenerate_6d_inputs() -> None:
    reconstruction = torch.zeros(2, 3, 6, requires_grad=True)
    target = torch.zeros_like(reconstruction)

    loss = rotation_geodesic_loss(
        reconstruction,
        target,
        {},
        torch.ones(2, 3, dtype=torch.bool),
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert reconstruction.grad is not None
    assert torch.isfinite(reconstruction.grad).all()


def test_rotation_geodesic_computes_in_fp32_for_low_precision_inputs() -> None:
    identity = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.bfloat16,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = rotation_geodesic_loss(
            identity,
            identity,
            {},
            torch.ones(1, 1, dtype=torch.bool),
        )

    assert loss.dtype == torch.float32
    assert loss.item() == 0.0


def test_symmetric_global_local_infonce_masks_invalid_pairs_before_math() -> None:
    action_features = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [float("nan"), float("nan"), float("nan")]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        requires_grad=True,
    )
    vl_features = action_features.detach().clone()
    valid_mask = torch.tensor([[True, False], [True, True]])
    logit_scale = torch.tensor(math.log(10.0), requires_grad=True)

    global_loss, local_loss = symmetric_global_local_infonce(
        action_features,
        vl_features,
        valid_mask,
        logit_scale,
    )

    assert global_loss.ndim == 0
    assert local_loss.ndim == 0
    assert torch.isfinite(global_loss)
    assert torch.isfinite(local_loss)
    (global_loss + local_loss).backward()
    assert action_features.grad is not None
    assert torch.isfinite(action_features.grad).all()
    assert logit_scale.grad is not None
    assert torch.isfinite(logit_scale.grad)


def test_symmetric_global_local_infonce_returns_differentiable_zero_without_pairs() -> (
    None
):
    action_features = torch.randn(2, 3, 4, requires_grad=True)
    vl_features = torch.randn(2, 3, 4)
    valid_mask = torch.zeros(2, 3, dtype=torch.bool)
    logit_scale = torch.tensor(math.log(10.0), requires_grad=True)

    global_loss, local_loss = symmetric_global_local_infonce(
        action_features,
        vl_features,
        valid_mask,
        logit_scale,
    )

    assert global_loss.item() == 0.0
    assert local_loss.item() == 0.0
    (global_loss + local_loss).backward()
    assert action_features.grad is not None
    assert torch.equal(action_features.grad, torch.zeros_like(action_features))


def test_symmetric_infonce_uses_paper_logit_scale() -> None:
    features = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])

    global_loss, local_loss = symmetric_global_local_infonce(
        features,
        features,
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor(math.log(10.0)),
    )

    expected = math.log1p(math.exp(-10.0))
    assert global_loss.item() == pytest.approx(expected, abs=1e-7)
    assert local_loss.item() == pytest.approx(expected, abs=1e-7)


def test_symmetric_infonce_clamps_logit_scale_at_100() -> None:
    theta = math.pi / 4.0 - 0.005
    action_features = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    vl_features = torch.tensor(
        [
            [[math.cos(theta), math.sin(theta)]],
            [[math.sin(theta), math.cos(theta)]],
        ]
    )

    global_loss, local_loss = symmetric_global_local_infonce(
        action_features,
        vl_features,
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor(math.log(1000.0)),
    )

    similarity_margin = math.cos(theta) - math.sin(theta)
    expected = math.log1p(math.exp(-100.0 * similarity_margin))
    assert global_loss.item() == pytest.approx(expected, abs=2e-6)
    assert local_loss.item() == pytest.approx(expected, abs=2e-6)


def test_distributed_infonce_uses_global_variable_count_pairs_and_gradients(
    tmp_path: Path,
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch gloo distributed support is unavailable.")

    world_size = 2
    spawn(
        _distributed_infonce_worker,
        args=(world_size, str(tmp_path / "infonce-init"), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"infonce-rank-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]
    expected = math.log1p(2.0 * math.exp(-10.0))
    for state in states:
        assert state["gathered_action"].shape == (3, 3)
        torch.testing.assert_close(state["gathered_action"], torch.eye(3))
        torch.testing.assert_close(state["gathered_vl"], torch.eye(3))
        assert state["global_loss"].item() == pytest.approx(expected, abs=1e-7)
        assert state["local_loss"].item() == pytest.approx(expected, abs=1e-7)
        assert torch.isfinite(state["action_grad"]).all()
        assert torch.isfinite(state["vl_grad"]).all()
        assert bool((state["action_grad"] != 0).any())
        assert bool((state["vl_grad"] != 0).any())
        assert torch.isfinite(state["scale_grad"])
