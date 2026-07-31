# SPDX-FileCopyrightText: 2026 Xiaoshen Han
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the X-Tokenizer semantic residual quantizer."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn

from praxis.codecs.latents.xtokenizer.quantizer import (
    SemanticResidualQuantizer,
)


def _set_codebooks(
    quantizer: SemanticResidualQuantizer,
    codebooks: torch.Tensor,
) -> None:
    with torch.no_grad():
        quantizer.codebooks.copy_(codebooks)
        quantizer.ema_sums.copy_(codebooks * quantizer.ema_counts.unsqueeze(-1))


def _distributed_quantizer_worker(
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
        quantizer = SemanticResidualQuantizer(
            dim=2,
            codebook_size=3,
            num_quantizers=1,
            decay=0.5,
            dead_code_threshold=2,
            kmeans_init=False,
        )
        with torch.no_grad():
            quantizer.codebooks.zero_()
            quantizer.ema_counts.copy_(torch.tensor([[10.0, 0.0, 0.0]]))
            quantizer.ema_sums.zero_()
        quantizer.train()
        if rank == 0:
            rank_latents = torch.zeros(1, 2, 2)
            valid_mask = torch.zeros(1, 2, dtype=torch.bool)
        else:
            rank_latents = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
            valid_mask = torch.ones(1, 2, dtype=torch.bool)
        torch.manual_seed(100 + rank)

        quantizer(rank_latents, valid_mask)

        torch.save(
            {
                "codebooks": quantizer.codebooks,
                "ema_counts": quantizer.ema_counts,
                "ema_sums": quantizer.ema_sums,
            },
            Path(output_dir) / f"rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def _distributed_kmeans_worker(
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
        torch.manual_seed(100 + rank)
        quantizer = SemanticResidualQuantizer(
            dim=1,
            codebook_size=2,
            num_quantizers=2,
            decay=0.8,
            dead_code_threshold=0,
            kmeans_init=True,
            kmeans_iters=10,
        )
        quantizer.train()
        rank_latents = (
            torch.tensor([[[0.0], [2.0]]])
            if rank == 0
            else torch.tensor([[[8.0], [10.0]]])
        )

        quantizer(rank_latents)

        torch.save(
            {
                "codebooks": quantizer.codebooks,
                "initialized": quantizer.initialized,
            },
            Path(output_dir) / f"kmeans-rank-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def test_quantizer_applies_all_four_levels_to_the_running_residual() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=4,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    _set_codebooks(
        quantizer,
        torch.tensor(
            [
                [[0.0], [6.0]],
                [[3.0], [9.0]],
                [[0.75], [4.0]],
                [[0.20], [1.5]],
            ]
        ),
    )
    quantizer.eval()

    output = quantizer(torch.tensor([[[10.0]]]))

    assert output.indices.shape == (1, 1, 4)
    assert torch.equal(output.indices, torch.tensor([[[1, 0, 0, 0]]]))
    assert output.quantized.shape == (1, 1, 1)
    assert torch.allclose(output.quantized, torch.tensor([[[9.95]]]))
    assert output.commitment_loss.ndim == 0
    assert output.perplexity.shape == (4,)
    assert torch.allclose(output.perplexity, torch.ones(4))


def test_decode_encode_equals_the_sum_of_selected_codewords() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=2,
        codebook_size=3,
        num_quantizers=3,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    _set_codebooks(
        quantizer,
        torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
                [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
                [[0.0, 0.0], [0.0, 0.25], [0.25, 0.0]],
            ]
        ),
    )
    quantizer.eval()
    latents = torch.tensor([[[2.6, 0.2], [0.1, 2.4]]])

    indices = quantizer.encode(latents)
    decoded = quantizer.decode(indices)
    expected = torch.stack(
        [
            quantizer.codebooks[level][indices[..., level]]
            for level in range(quantizer.num_quantizers)
        ]
    ).sum(dim=0)

    assert indices.shape == (1, 2, 3)
    assert indices.dtype == torch.long
    assert decoded.shape == latents.shape
    assert torch.equal(decoded, expected)
    assert torch.equal(decoded, quantizer(latents).quantized)


def test_commitment_loss_and_straight_through_gradient() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    _set_codebooks(
        quantizer,
        torch.tensor(
            [
                [[0.0], [2.0]],
                [[0.0], [1.0]],
            ]
        ),
    )
    quantizer.eval()
    latents = torch.tensor([[[3.0], [100.0]]], requires_grad=True)
    valid_mask = torch.tensor([[True, False]])

    output = quantizer(latents, valid_mask)

    assert torch.allclose(output.commitment_loss, torch.tensor(0.5))
    output.quantized.sum().backward()
    assert latents.grad is not None
    assert torch.equal(latents.grad, torch.ones_like(latents))


def test_commitment_loss_contributes_latent_gradients() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=1,
        num_quantizers=1,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    _set_codebooks(quantizer, torch.zeros(1, 1, 1))
    quantizer.eval()
    latents = torch.tensor([[[2.0]]], requires_grad=True)

    quantizer(latents).commitment_loss.backward()

    assert latents.grad is not None
    assert torch.equal(latents.grad, torch.tensor([[[4.0]]]))


def test_non_finite_padding_does_not_contaminate_commitment_loss() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=1,
        num_quantizers=1,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    _set_codebooks(quantizer, torch.zeros(1, 1, 1))
    quantizer.eval()
    latents = torch.tensor([[[2.0], [float("nan")], [float("inf")]]])
    valid_mask = torch.tensor([[True, False, False]])

    output = quantizer(latents, valid_mask)

    assert torch.isfinite(output.commitment_loss)
    assert torch.equal(output.commitment_loss, torch.tensor(4.0))


@pytest.mark.parametrize("latent_dtype", [torch.bfloat16, torch.float16])
def test_dtype_conversion_keeps_ema_state_fp32_and_returns_latent_dtype(
    latent_dtype: torch.dtype,
) -> None:
    quantizer = SemanticResidualQuantizer(
        dim=2,
        codebook_size=3,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    fp32_state = {name: value.clone() for name, value in quantizer.state_dict().items()}
    quantizer.to(dtype=latent_dtype)
    quantizer.eval()

    assert quantizer.codebooks.dtype is torch.float32
    assert quantizer.ema_counts.dtype is torch.float32
    assert quantizer.ema_sums.dtype is torch.float32
    assert all(
        torch.equal(value, fp32_state[name])
        for name, value in quantizer.state_dict().items()
    )

    latents = torch.tensor([[[0.5, -0.25]]], dtype=latent_dtype)
    output = quantizer(latents)

    assert output.quantized.dtype is latent_dtype
    assert output.commitment_loss.dtype is torch.float32
    assert output.perplexity.dtype is torch.float32


def test_cpu_autocast_preserves_fp32_quantization_decisions() -> None:
    torch.manual_seed(6)
    quantizer = SemanticResidualQuantizer(
        dim=8,
        codebook_size=16,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    quantizer.eval()
    latents = torch.randn(4, 7, 8)
    expected = quantizer(latents)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = quantizer(latents)

    assert actual.quantized.dtype is latents.dtype
    assert torch.equal(actual.indices, expected.indices)
    assert torch.equal(actual.quantized, expected.quantized)
    assert torch.equal(actual.commitment_loss, expected.commitment_loss)
    assert torch.equal(actual.perplexity, expected.perplexity)
    assert quantizer.codebooks.dtype is torch.float32
    assert quantizer.ema_counts.dtype is torch.float32
    assert quantizer.ema_sums.dtype is torch.float32


def test_valid_mask_excludes_padded_positions_from_ema_counts_and_sums() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=1,
        decay=0.5,
        dead_code_threshold=0,
        kmeans_init=False,
    )
    with torch.no_grad():
        quantizer.codebooks.copy_(torch.tensor([[[-1.0], [1.0]]]))
        quantizer.ema_counts.copy_(torch.tensor([[2.0, 2.0]]))
        quantizer.ema_sums.copy_(torch.tensor([[[-2.0], [2.0]]]))
    quantizer.train()
    latents = torch.tensor([[[-1.0], [1.0], [100.0]]], requires_grad=True)
    valid_mask = torch.tensor([[True, True, False]])

    quantizer(latents, valid_mask)

    assert torch.allclose(quantizer.ema_counts, torch.tensor([[1.5, 1.5]]))
    assert torch.allclose(
        quantizer.ema_sums,
        torch.tensor([[[-1.5], [1.5]]]),
    )
    assert torch.allclose(
        quantizer.codebooks,
        torch.tensor([[[-1.0], [1.0]]]),
    )


def test_dead_codes_reset_deterministically_from_valid_latents() -> None:
    first = SemanticResidualQuantizer(
        dim=2,
        codebook_size=3,
        num_quantizers=1,
        decay=0.9,
        dead_code_threshold=2,
        kmeans_init=False,
    )
    second = SemanticResidualQuantizer(
        dim=2,
        codebook_size=3,
        num_quantizers=1,
        decay=0.9,
        dead_code_threshold=2,
        kmeans_init=False,
    )
    with torch.no_grad():
        first.codebooks.zero_()
        first.ema_counts.copy_(torch.tensor([[10.0, 0.0, 0.0]]))
        first.ema_sums.zero_()
        second.load_state_dict(first.state_dict())
    first.train()
    second.train()
    latents = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [99.0, 99.0]]],
        requires_grad=True,
    )
    valid_mask = torch.tensor([[True, True, False]])

    torch.manual_seed(1234)
    first(latents, valid_mask)
    torch.manual_seed(1234)
    second(latents.detach().clone().requires_grad_(True), valid_mask)

    assert torch.equal(first.codebooks, second.codebooks)
    assert torch.equal(first.ema_counts, second.ema_counts)
    assert torch.equal(first.ema_sums, second.ema_sums)
    assert torch.equal(first.ema_counts[0, 1:], torch.full((2,), 2.0))
    assert torch.equal(
        first.ema_sums[0, 1:],
        first.codebooks[0, 1:] * 2.0,
    )
    valid_latents = latents.detach()[valid_mask]
    for replacement in first.codebooks[0, 1:]:
        assert bool(torch.all(replacement == valid_latents, dim=-1).any())


def test_all_invalid_positions_produce_finite_zero_statistics() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=2,
        codebook_size=4,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=2,
        kmeans_init=False,
    )
    before = {name: value.clone() for name, value in quantizer.state_dict().items()}
    latents = torch.randn(2, 3, 2, requires_grad=True)

    output = quantizer(
        latents,
        valid_mask=torch.zeros((2, 3), dtype=torch.bool),
    )

    assert output.commitment_loss == 0.0
    assert torch.equal(output.perplexity, torch.zeros(2))
    assert torch.isfinite(output.quantized).all()
    assert all(
        torch.equal(value, before[name])
        for name, value in quantizer.state_dict().items()
    )


def test_distributed_ema_and_dead_code_replacement_are_synchronized(
    tmp_path: Path,
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch gloo distributed support is unavailable.")

    world_size = 2
    init_file = tmp_path / "quantizer-init"
    spawn(
        _distributed_quantizer_worker,
        args=(world_size, str(init_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]

    for name in ("codebooks", "ema_counts", "ema_sums"):
        torch.testing.assert_close(
            states[0][name],
            states[1][name],
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        states[0]["ema_counts"][0, 0],
        torch.tensor(6.0),
    )
    torch.testing.assert_close(
        states[0]["ema_sums"][0, 0],
        torch.tensor([0.5, 1.0]),
    )
    replacements = states[0]["codebooks"][0, 1:]
    global_latents = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    for replacement in replacements:
        assert bool(torch.all(replacement == global_latents, dim=-1).any())


def test_kmeans_initializes_each_level_from_its_running_residual() -> None:
    torch.manual_seed(0)
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=True,
        kmeans_iters=10,
    )
    quantizer.train()

    quantizer(torch.tensor([[[0.0], [2.0], [8.0], [10.0]]]))

    assert torch.equal(quantizer.initialized, torch.tensor([True, True]))
    torch.testing.assert_close(
        quantizer.codebooks[0].sort(dim=0).values,
        torch.tensor([[1.0], [9.0]]),
    )
    torch.testing.assert_close(
        quantizer.codebooks[1].sort(dim=0).values,
        torch.tensor([[-1.0], [1.0]]),
    )


def test_quantizer_uses_the_paper_kmeans_defaults() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=4,
        decay=0.8,
        dead_code_threshold=2,
    )

    assert quantizer.kmeans_init is True
    assert quantizer.kmeans_iters == 100
    assert torch.equal(quantizer.initialized, torch.zeros(4, dtype=torch.bool))


def test_kmeans_waits_for_the_first_globally_valid_batch() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=True,
        kmeans_iters=5,
    )
    quantizer.train()
    initial_codebooks = quantizer.codebooks.clone()

    quantizer(
        torch.tensor([[[1000.0], [2000.0]]]),
        valid_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    assert not bool(quantizer.initialized.any())
    assert torch.equal(quantizer.codebooks, initial_codebooks)

    quantizer(torch.tensor([[[0.0], [2.0], [8.0], [10.0]]]))

    assert bool(quantizer.initialized.all())
    assert bool((quantizer.codebooks.abs() < 20.0).all())


def test_kmeans_initialization_state_round_trips_through_state_dict() -> None:
    quantizer = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=True,
        kmeans_iters=5,
    )
    quantizer.train()
    quantizer(torch.tensor([[[0.0], [2.0], [8.0], [10.0]]]))

    restored = SemanticResidualQuantizer(
        dim=1,
        codebook_size=2,
        num_quantizers=2,
        decay=0.8,
        dead_code_threshold=0,
        kmeans_init=True,
        kmeans_iters=5,
    )
    restored.load_state_dict(quantizer.state_dict())

    assert torch.equal(restored.initialized, torch.tensor([True, True]))
    assert torch.equal(restored.codebooks, quantizer.codebooks)


def test_distributed_kmeans_uses_global_residuals_and_is_synchronized(
    tmp_path: Path,
) -> None:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("PyTorch gloo distributed support is unavailable.")

    world_size = 2
    init_file = tmp_path / "kmeans-init"
    spawn(
        _distributed_kmeans_worker,
        args=(world_size, str(init_file), str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    states = [
        torch.load(tmp_path / f"kmeans-rank-{rank}.pt", weights_only=True)
        for rank in range(world_size)
    ]

    torch.testing.assert_close(
        states[0]["codebooks"],
        states[1]["codebooks"],
        rtol=0,
        atol=0,
    )
    assert torch.equal(states[0]["initialized"], torch.tensor([True, True]))
    assert torch.equal(states[1]["initialized"], torch.tensor([True, True]))
    torch.testing.assert_close(
        states[0]["codebooks"][0].sort(dim=0).values,
        torch.tensor([[1.0], [9.0]]),
    )
    torch.testing.assert_close(
        states[0]["codebooks"][1].sort(dim=0).values,
        torch.tensor([[-1.0], [1.0]]),
    )
