# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from praxis.training.core import eval_memory


def test_cuda_memory_snapshot_formats_bytes_as_gib() -> None:
    snapshot = eval_memory.CudaMemorySnapshot(
        free=2 * 1024**3,
        total=4 * 1024**3,
        allocated=1024**3,
        reserved=3 * 1024**3,
        max_allocated=5 * 1024**3,
    )

    assert snapshot.format_gib() == (
        "free=2.00 GiB total=4.00 GiB allocated=1.00 GiB "
        "reserved=3.00 GiB max_allocated=5.00 GiB"
    )


def test_optimizer_state_cpu_offload_restores_tensor_state() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    state_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    originals = [tensor.detach().clone() for tensor in state_tensors]

    offload = eval_memory.offload_optimizer_state_to_cpu(
        optimizer,
        source_device_types=("cpu",),
    )

    assert offload.moved_tensors == len(state_tensors)
    assert offload.moved_bytes == sum(
        tensor.numel() * tensor.element_size() for tensor in originals
    )
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"

    offload.restore()

    restored = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    for before, after in zip(originals, restored, strict=True):
        torch.testing.assert_close(after, before)

    optimizer.zero_grad(set_to_none=True)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()


def test_optimizer_state_cpu_offload_context_restores_on_error() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()

    with (
        pytest.raises(RuntimeError, match="eval failed"),
        eval_memory.optimizer_state_cpu_offload(
            optimizer,
            source_device_types=("cpu",),
        ) as offload,
    ):
        assert offload.moved_tensors > 0
        raise RuntimeError("eval failed")

    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
