"""Memory cleanup helpers shared by trainer eval hooks."""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

_GIB = 1024**3


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def bytes_to_gib(value: int) -> float:
    return value / _GIB


@dataclass(frozen=True)
class CudaMemorySnapshot:
    """Structured CUDA memory counters for eval handoff diagnostics."""

    free: int
    total: int
    allocated: int
    reserved: int
    max_allocated: int

    def format_gib(self) -> str:
        return (
            f"free={bytes_to_gib(self.free):.2f} GiB "
            f"total={bytes_to_gib(self.total):.2f} GiB "
            f"allocated={bytes_to_gib(self.allocated):.2f} GiB "
            f"reserved={bytes_to_gib(self.reserved):.2f} GiB "
            f"max_allocated={bytes_to_gib(self.max_allocated):.2f} GiB"
        )


@dataclass
class OptimizerStateCpuOffload:
    """Tracks optimizer tensors temporarily moved from selected devices to CPU."""

    _records: list[tuple[dict[Any, Any], Any, torch.device]]
    moved_tensors: int
    moved_bytes: int
    restored: bool = False

    @property
    def moved_gib(self) -> float:
        return bytes_to_gib(self.moved_bytes)

    def restore(self) -> None:
        if self.restored:
            return
        for state, key, device in reversed(self._records):
            value = state[key]
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, copy=value.device == device)
        self.restored = True


def offload_optimizer_state_to_cpu(
    optimizer: torch.optim.Optimizer | None,
    *,
    source_device_types: tuple[str, ...] = ("cuda",),
) -> OptimizerStateCpuOffload:
    """Move optimizer state tensors from selected devices to CPU."""
    records: list[tuple[dict[Any, Any], Any, torch.device]] = []
    offload = OptimizerStateCpuOffload(records, moved_tensors=0, moved_bytes=0)
    if optimizer is None:
        return offload

    selected_devices = {str(device_type) for device_type in source_device_types}
    try:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if value.device.type not in selected_devices:
                    continue
                moved = value.to(device=torch.device("cpu"), copy=True)
                records.append((state, key, value.device))
                offload.moved_tensors += 1
                offload.moved_bytes += _tensor_nbytes(value)
                state[key] = moved
    except BaseException:
        offload.restore()
        raise
    return offload


@contextlib.contextmanager
def optimizer_state_cpu_offload(
    optimizer: torch.optim.Optimizer | None,
    *,
    source_device_types: tuple[str, ...] = ("cuda",),
) -> Iterator[OptimizerStateCpuOffload]:
    offload = offload_optimizer_state_to_cpu(
        optimizer,
        source_device_types=source_device_types,
    )
    try:
        yield offload
    finally:
        offload.restore()


def cuda_memory_snapshot(device: torch.device | str) -> CudaMemorySnapshot | None:
    """Return current CUDA memory counters for logging, or None off CUDA."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return None
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return None
    with torch.cuda.device(device_obj):
        free, total = torch.cuda.mem_get_info()
        return CudaMemorySnapshot(
            free=int(free),
            total=int(total),
            allocated=int(torch.cuda.memory_allocated()),
            reserved=int(torch.cuda.memory_reserved()),
            max_allocated=int(torch.cuda.max_memory_allocated()),
        )


def release_cuda_eval_memory(device: torch.device | str) -> None:
    """Release CUDA allocations left by long eval hooks before training resumes."""
    gc.collect()
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return
    if device_obj.index is None:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return
    with torch.cuda.device(device_obj):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
