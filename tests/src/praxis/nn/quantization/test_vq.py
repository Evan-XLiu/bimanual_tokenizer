"""Unit tests for EMA vector quantization module."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from praxis.nn.quantization.vq import VectorQuantization


def test_vq_forward_shapes():
    vq = VectorQuantization(
        input_dim=8,
        codebook_size=16,
        codebook_dim=8,
        kmeans_init=False,
    )
    x = torch.randn(2, 5, 8)
    quantized, indices, commitment_loss, codebook_loss = vq(x)

    assert quantized.shape == x.shape
    assert indices.shape == (2, 5)
    assert commitment_loss.ndim == 0
    assert codebook_loss.ndim == 0


def test_vq_indices_roundtrip_to_quantized_shape():
    vq = VectorQuantization(
        input_dim=8,
        codebook_size=16,
        codebook_dim=8,
        kmeans_init=False,
    )
    x = torch.randn(2, 5, 8)
    _quantized, indices, _commitment_loss, _codebook_loss = vq(x)

    decoded = vq.indices_to_quantized(indices)
    assert decoded.shape == x.shape


def test_vq_kmeans_init_runs_in_single_process():
    vq = VectorQuantization(
        input_dim=2,
        codebook_size=4,
        codebook_dim=2,
        kmeans_init=True,
        kmeans_iters=2,
    )
    encodings = torch.tensor(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [-1.0, 0.0],
            [-1.1, -0.1],
        ],
        dtype=torch.float32,
    )

    vq._init_codebook(encodings)

    assert bool(vq.inited.item())
    assert torch.count_nonzero(vq.codebook).item() > 0
    assert int((vq.cluster_size > 0).sum().item()) >= 2
    assert torch.isclose(vq.cluster_size.sum(), torch.tensor(4.0))


def test_vq_replace_dead_codes_refreshes_ema_state():
    vq = VectorQuantization(
        input_dim=4,
        codebook_size=4,
        codebook_dim=4,
        decay=0.9,
        threshold_ema_dead=2,
        kmeans_init=False,
    )
    with torch.no_grad():
        vq.cluster_size.copy_(torch.tensor([10.0, 0.0, 0.0, 0.0]))
        vq.embed_avg.zero_()
        vq.codebook.zero_()

        replacements = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        vq._replace_dead_codes(replacements)

        dead_mask = torch.tensor([False, True, True, True])
        expected_cluster = torch.full((3,), 2.0, dtype=torch.float32)
        assert torch.allclose(vq.cluster_size[dead_mask], expected_cluster)
        actual_codebook = vq.codebook[dead_mask]
        order = actual_codebook.argmax(dim=1).argsort()
        expected_order = replacements.argmax(dim=1).argsort()
        assert torch.allclose(actual_codebook[order], replacements[expected_order])
        assert torch.allclose(
            vq.embed_avg[dead_mask][order],
            (replacements * expected_cluster.unsqueeze(1))[expected_order],
        )

        prev_codebook = vq.codebook[dead_mask].clone()
        encodings = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        onehot = F.one_hot(torch.zeros(2, dtype=torch.long), num_classes=4).float()
        vq._ema_update(encodings, onehot)

        assert torch.allclose(vq.codebook[dead_mask], prev_codebook)


def test_vq_non_kmeans_init_keeps_inactive_codes_stable():
    vq = VectorQuantization(
        input_dim=4,
        codebook_size=8,
        codebook_dim=4,
        decay=0.9,
        threshold_ema_dead=0,
        kmeans_init=False,
    )
    with torch.no_grad():
        initial_codebook = vq.codebook.clone()
        encodings = torch.randn(16, 4)
        onehot = F.one_hot(torch.zeros(16, dtype=torch.long), num_classes=8).float()

        vq._ema_update(encodings, onehot)

        assert torch.allclose(vq.codebook[1:], initial_codebook[1:], atol=1e-4)
        assert float(vq.codebook.abs().max()) < 10.0
