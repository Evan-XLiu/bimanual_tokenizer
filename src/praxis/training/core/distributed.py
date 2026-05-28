"""PyTorch-native distributed training runtime.

This module owns the trainer's distributed substrate so DDP and FSDP behavior
stay explicit and auditable.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DataLoader, Sampler

from praxis.core.devices import require_available_device
from praxis.training.core.dtensor_utils import (
    dtensor_has_sharded_placement,
    dtensor_to_local,
    is_dtensor,
)
from praxis.training.core.fsdp import (
    ignored_fsdp_parameters,
    select_fsdp_wrap_modules,
    set_fsdp_accumulation_sync,
    validate_fsdp_parameter_group,
)


@dataclass(frozen=True)
class DistributedRuntimeConfig:
    strategy: str = "ddp"
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    fsdp_min_num_params: int = 100_000_000
    fsdp_reshard_after_forward: bool | int = True
    fsdp_cpu_offload: bool = False
    data_seed: int = 0


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or raw == "" else int(raw)


def _dist_available_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank_error_message(context: str, exc: BaseException, *, rank: int) -> str:
    return f"{context} failed on rank {rank}: {type(exc).__name__}: {exc}"


def _has_trainable_parameters(model: nn.Module) -> bool:
    return any(param.requires_grad for param in model.parameters())


def resolve_fsdp_reshard_after_forward(value: object) -> bool | int:
    """Parse FSDP's bool-or-int reshard policy from config/CLI values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(
                f"fsdp_reshard_after_forward integer values must be > 0, got {value}."
            )
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on"}:
            return True
        if normalized in {"false", "no", "off"}:
            return False
        try:
            parsed = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "fsdp_reshard_after_forward must be a bool or positive int, "
                f"got {value!r}."
            ) from exc
        return resolve_fsdp_reshard_after_forward(parsed)
    raise TypeError(
        "fsdp_reshard_after_forward must be a bool or positive int, "
        f"got {type(value).__name__}."
    )


def _resolve_device(requested_device: object, local_rank: int) -> torch.device:
    requested = str(requested_device).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if not requested.startswith("cuda"):
        return torch.device(requested)
    require_available_device(requested, purpose="Training")
    device_count = torch.cuda.device_count()
    local_world_size = _env_int("LOCAL_WORLD_SIZE", 1)
    if local_world_size > device_count:
        raise RuntimeError(
            "torchrun requested more local processes than visible CUDA devices: "
            f"LOCAL_WORLD_SIZE={local_world_size}, cuda_device_count={device_count}. "
            "Check --nproc_per_node and the visible GPU allocation."
        )
    if ":" in requested and _env_int("WORLD_SIZE", 1) == 1:
        index = int(requested.split(":", 1)[1])
    else:
        index = local_rank if local_rank >= 0 else 0
    if index >= device_count:
        raise RuntimeError(
            f"Resolved CUDA device index {index}, but only {device_count} CUDA "
            "device(s) are visible. Check LOCAL_RANK, CUDA_VISIBLE_DEVICES, and "
            "the visible GPU allocation."
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


class _DistributedBatchSampler(Sampler[list[int]]):
    """Shard an existing batch sampler by distributed rank.

    This preserves dataset-owned sampling decisions such as EpisodeAwareSampler
    while giving each rank a distinct subset of complete per-rank batches. The
    wrapped sampler is iterated under a runtime-owned seed so rank-local RNG
    drift cannot make ranks shard different permutations.
    """

    def __init__(
        self,
        batch_sampler: BatchSampler,
        *,
        rank: int,
        world_size: int,
        seed: int = 0,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self._epoch = 0
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(
                f"rank must be in [0, world_size), got rank={rank}, world_size={world_size}"
            )

    def __iter__(self) -> Iterator[list[int]]:
        seed = (self.seed + self._epoch) % (2**63)
        self._epoch += 1
        first_batches: list[list[int]] = []
        total = 0
        sampler_state = _RankIndependentSamplerState(self.batch_sampler, seed)
        sampler_iter = sampler_state.run(lambda: iter(self.batch_sampler))
        batch_index = 0

        while True:
            try:
                batch_list = sampler_state.run(lambda: list(next(sampler_iter)))
            except StopIteration:
                break
            if len(first_batches) < self.world_size:
                first_batches.append(batch_list)
            if batch_index % self.world_size == self.rank:
                yield batch_list
            total = batch_index + 1
            batch_index += 1

        if total == 0:
            return

        remainder = total % self.world_size
        if remainder:
            for offset in range(self.world_size - remainder):
                padded_index = total + offset
                if padded_index % self.world_size == self.rank:
                    yield first_batches[offset % len(first_batches)]

    def __len__(self) -> int:
        total = len(self.batch_sampler)
        return (total + self.world_size - 1) // self.world_size

    @property
    def num_real_batches(self) -> int:
        total = len(self.batch_sampler)
        if total <= self.rank:
            return 0
        return ((total - 1 - self.rank) // self.world_size) + 1

    def is_padding_batch(self, batch_index: int) -> bool:
        return int(batch_index) >= self.num_real_batches


def _iter_sampler_graph(root: object) -> Iterator[object]:
    stack = [root]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        marker = id(obj)
        if marker in seen:
            continue
        seen.add(marker)
        yield obj
        for child_name in ("batch_sampler", "sampler"):
            child = getattr(obj, child_name, None)
            if child is not None:
                stack.append(child)


class _RankIndependentSamplerState:
    """Own rank-independent RNG state without leaking it across yielded batches."""

    def __init__(self, root: object, seed: int) -> None:
        self.root = root
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        torch_seed_generator = torch.Generator()
        torch_seed_generator.manual_seed(seed)
        self._torch_state = torch_seed_generator.get_state()
        py_seed_generator = random.Random(seed)
        self._py_state = py_seed_generator.getstate()

    def run(self, fn):
        caller_py_state = random.getstate()
        patched_generators: list[tuple[object, object]] = []
        try:
            random.setstate(self._py_state)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(self._torch_state)
                for sampler in _iter_sampler_graph(self.root):
                    sampler_with_generator = cast(Any, sampler)
                    try:
                        original = sampler_with_generator.generator
                        sampler_with_generator.generator = self.generator
                    except (AttributeError, TypeError):
                        continue
                    patched_generators.append((sampler, original))
                result = fn()
                self._torch_state = torch.get_rng_state()
                self._py_state = random.getstate()
                return result
        finally:
            for sampler, original in reversed(patched_generators):
                cast(Any, sampler).generator = original
            random.setstate(caller_py_state)


def is_distributed_padding_batch(dataloader: DataLoader, batch_index: int) -> bool:
    """Return whether ``batch_index`` is a collective-safety padding batch."""
    checker = getattr(
        getattr(dataloader, "batch_sampler", None), "is_padding_batch", None
    )
    return bool(callable(checker) and checker(batch_index))


@torch.no_grad()
def _move_parameters_to_device(
    parameters: set[nn.Parameter] | frozenset[nn.Parameter],
    device: torch.device,
) -> None:
    for param in parameters:
        if param.device != device:
            param.data = param.data.to(device=device)


def _register_fsdp_forward_methods(model: nn.Module) -> None:
    """Register non-forward policy methods that execute model parameters."""
    from torch.distributed.fsdp import register_fsdp_forward_method

    method_hook = getattr(model, "fsdp_forward_methods", None)
    if not callable(method_hook):
        return
    raw_method_names = method_hook()
    if not isinstance(raw_method_names, Iterable) or isinstance(
        raw_method_names,
        str | bytes,
    ):
        raise TypeError(
            "fsdp_forward_methods() must return an iterable of method names, "
            f"got {type(raw_method_names).__name__}."
        )
    method_names = tuple(raw_method_names)
    for method_name in method_names:
        if not isinstance(method_name, str):
            raise TypeError(
                "fsdp_forward_methods() must return method names as strings, "
                f"got {type(method_name).__name__}."
            )
        if callable(getattr(model, method_name, None)):
            register_fsdp_forward_method(model, method_name)


class TrainingRuntime:
    """Small runtime facade around raw PyTorch distributed primitives."""

    def __init__(self, *, cfg: DistributedRuntimeConfig, requested_device: object):
        if cfg.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be > 0, "
                f"got {cfg.gradient_accumulation_steps}."
            )
        self.cfg = cfg
        self.global_rank = _env_int("RANK", 0)
        self.local_rank = _env_int("LOCAL_RANK", 0)
        self.world_size = _env_int("WORLD_SIZE", 1)
        self._micro_step = 0
        self._owns_process_group = False
        self.device = _resolve_device(requested_device, self.local_rank)

        if self.world_size > 1 and not _dist_available_and_initialized():
            backend = "nccl" if self.device.type == "cuda" else "gloo"
            init_kwargs: dict[str, Any] = {"backend": backend}
            if backend == "nccl":
                init_kwargs["device_id"] = self.device
            dist.init_process_group(**init_kwargs)
            self._owns_process_group = True

        if _dist_available_and_initialized():
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        if cfg.strategy == "fsdp" and self.device.type != "cuda":
            raise ValueError("trainer.distributed_strategy=fsdp requires CUDA.")
        if cfg.strategy == "fsdp" and self.world_size <= 1:
            raise ValueError(
                "trainer.distributed_strategy=fsdp requires a distributed torchrun "
                "launch with WORLD_SIZE > 1. Use ddp for single-GPU training."
            )
        self.device_mesh = self._init_device_mesh()

    @property
    def is_main_process(self) -> bool:
        return self.global_rank == 0

    @property
    def num_processes(self) -> int:
        return self.world_size

    @property
    def sync_gradients(self) -> bool:
        return (self._micro_step + 1) % self.cfg.gradient_accumulation_steps == 0

    @contextlib.contextmanager
    def main_process_first(self):
        if not _dist_available_and_initialized():
            yield
            return

        context = "main_process_first"
        if self.is_main_process:
            main_error = None
            try:
                yield
            except BaseException as exc:
                main_error = _rank_error_message(
                    context,
                    exc,
                    rank=self.global_rank,
                )
            main_error = self.broadcast_object(main_error, src=0)
            if main_error is not None:
                raise RuntimeError(main_error)

            rank_errors = [
                error for error in self.gather_object(None) if error is not None
            ]
            if rank_errors:
                raise RuntimeError("; ".join(str(error) for error in rank_errors))
            return

        main_error = self.broadcast_object(None, src=0)
        if main_error is not None:
            raise RuntimeError(main_error)

        local_error = None
        try:
            yield
        except BaseException as exc:
            local_error = _rank_error_message(
                context,
                exc,
                rank=self.global_rank,
            )
        rank_errors = [
            error for error in self.gather_object(local_error) if error is not None
        ]
        if rank_errors:
            raise RuntimeError("; ".join(str(error) for error in rank_errors))

    def wait_for_everyone(self) -> None:
        if _dist_available_and_initialized():
            if self.device.type == "cuda" and self.device.index is not None:
                dist.barrier(device_ids=[self.device.index])
                return
            dist.barrier()

    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap_for_training: bool = True,
    ) -> nn.Module:
        if self.cfg.strategy == "ddp":
            model = model.to(self.device)
            if not wrap_for_training or not _has_trainable_parameters(model):
                return model
            return self._prepare_ddp_model(model)
        if self.cfg.strategy == "fsdp":
            if not wrap_for_training:
                return model.to(self.device)
            return self._prepare_fsdp_model(model)
        raise ValueError(f"Unsupported distributed strategy: {self.cfg.strategy!r}")

    def _prepare_ddp_model(self, model: nn.Module) -> nn.Module:
        if self.world_size == 1:
            return model
        device_ids = [self.device.index] if self.device.type == "cuda" else None
        output_device = self.device.index if self.device.type == "cuda" else None
        return DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=self.cfg.find_unused_parameters,
            gradient_as_bucket_view=True,
        )

    def _prepare_fsdp_model(self, model: nn.Module) -> nn.Module:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            OffloadPolicy,
            fully_shard,
        )

        mp_policy = MixedPrecisionPolicy(
            param_dtype=self._mixed_precision_dtype(),
            reduce_dtype=self._mixed_precision_dtype(),
            output_dtype=None,
        )
        offload_policy = (
            CPUOffloadPolicy() if self.cfg.fsdp_cpu_offload else OffloadPolicy()
        )
        assigned_params: set[nn.Parameter] = set()
        globally_ignored_params: set[nn.Parameter] = set()
        mesh = self.device_mesh
        if mesh is None:
            raise RuntimeError("FSDP requested without an initialized device mesh.")
        for module in select_fsdp_wrap_modules(
            model,
            min_num_params=int(self.cfg.fsdp_min_num_params),
        ):
            ignored_param_frozenset = ignored_fsdp_parameters(
                module,
                excluded_params=assigned_params,
            )
            ignored_param_set = set(ignored_param_frozenset or frozenset())
            validate_fsdp_parameter_group(
                module,
                ignored_params=ignored_param_frozenset,
                excluded_params=assigned_params,
            )
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=self.cfg.fsdp_reshard_after_forward,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                ignored_params=ignored_param_set or None,
            )
            globally_ignored_params.update(ignored_param_set)
            assigned_params.update(
                param
                for param in module.parameters(recurse=True)
                if param not in ignored_param_set
            )
        ignored_param_set = set(
            ignored_fsdp_parameters(model, excluded_params=assigned_params)
            or frozenset()
        )
        ignored_param_set.update(globally_ignored_params)
        ignored_param_frozenset = frozenset(ignored_param_set) or None
        validate_fsdp_parameter_group(
            model,
            ignored_params=ignored_param_frozenset,
            excluded_params=assigned_params,
        )
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=self.cfg.fsdp_reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            ignored_params=ignored_param_set or None,
        )
        _move_parameters_to_device(ignored_param_set, self.device)
        _register_fsdp_forward_methods(model)
        return model

    def _init_device_mesh(self) -> DeviceMesh | None:
        if self.cfg.strategy != "fsdp":
            return None
        return init_device_mesh(
            self.device.type,
            (self.world_size,),
            mesh_dim_names=("dp",),
        )

    def prepare_dataloader(self, dataloader: DataLoader | None) -> DataLoader | None:
        if dataloader is None or self.world_size == 1:
            return dataloader
        return DataLoader(
            dataloader.dataset,
            batch_sampler=_DistributedBatchSampler(
                cast(BatchSampler, dataloader.batch_sampler),
                rank=self.global_rank,
                world_size=self.world_size,
                seed=self.cfg.data_seed,
            ),
            num_workers=dataloader.num_workers,
            collate_fn=dataloader.collate_fn,
            pin_memory=dataloader.pin_memory,
            drop_last=False,
            timeout=dataloader.timeout,
            worker_init_fn=dataloader.worker_init_fn,
            multiprocessing_context=dataloader.multiprocessing_context,
            generator=dataloader.generator,
            prefetch_factor=dataloader.prefetch_factor,
            persistent_workers=dataloader.persistent_workers,
            pin_memory_device=dataloader.pin_memory_device,
        )

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return cast(nn.Module, model.module) if hasattr(model, "module") else model

    @contextlib.contextmanager
    def autocast(self):
        dtype = self._mixed_precision_dtype()
        if dtype is None or self.device.type != "cuda":
            yield
            return
        with torch.autocast(device_type=self.device.type, dtype=dtype):
            yield

    @contextlib.contextmanager
    def accumulate(self, model: nn.Module):
        sync = self.sync_gradients
        context = contextlib.nullcontext()
        if not sync and isinstance(model, DistributedDataParallel):
            context = model.no_sync()
        else:
            set_fsdp_accumulation_sync(model, sync)
        try:
            with context:
                yield
        finally:
            set_fsdp_accumulation_sync(model, True)

    def backward(self, loss: Tensor) -> None:
        scaled_loss = loss / float(self.cfg.gradient_accumulation_steps)
        scaled_loss.backward()

    def mark_micro_step_complete(self) -> None:
        self._micro_step += 1

    def clip_grad_norm_(self, parameters, max_norm: float) -> Tensor:
        params = [param for param in parameters if param.grad is not None]
        if (
            self.cfg.strategy != "fsdp"
            or not params
            or not _dist_available_and_initialized()
        ):
            return torch.nn.utils.clip_grad_norm_(params, max_norm)
        return self._clip_fsdp_grad_norm_(params, max_norm)

    def _clip_fsdp_grad_norm_(
        self, parameters: list[nn.Parameter], max_norm: float
    ) -> Tensor:
        local_sq = torch.zeros((), device=self.device, dtype=torch.float32)
        for param in parameters:
            grad = cast(Tensor, param.grad)
            contributes_on_all_ranks = is_dtensor(
                grad
            ) and dtensor_has_sharded_placement(grad)
            local_grad = dtensor_to_local(grad)
            contribution = local_grad.detach().to(torch.float32).pow(2).sum()
            if contributes_on_all_ranks or self.global_rank == 0:
                local_sq += contribution.to(device=self.device)
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
        total_norm = local_sq.sqrt()
        clip_coef = float(max_norm) / (total_norm.item() + 1e-6)
        if clip_coef < 1.0:
            for param in parameters:
                cast(Tensor, param.grad).mul_(clip_coef)
        return total_norm

    def reduce(self, tensor: Tensor, reduction: str = "sum") -> Tensor:
        if reduction != "sum":
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        if not _dist_available_and_initialized():
            return tensor
        reduced = tensor.detach().clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced

    def gather_for_metrics(
        self,
        input_data: Any,
        *,
        use_gather_object: bool = False,
    ) -> Any:
        if not _dist_available_and_initialized():
            return input_data
        if not use_gather_object:
            raise ValueError("Only object metric gathering is supported.")
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, input_data)
        if isinstance(input_data, list) and all(
            isinstance(item, list) for item in gathered
        ):
            flattened: list[Any] = []
            for item in gathered:
                flattened.extend(item)
            return flattened
        return gathered

    def gather_object(self, obj: Any) -> list[Any]:
        if not _dist_available_and_initialized():
            return [obj]
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, obj)
        return gathered

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        if not _dist_available_and_initialized():
            return obj
        payload = [obj if self.global_rank == src else None]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]

    def end_training(self) -> None:
        self.wait_for_everyone()
        if self._owns_process_group and _dist_available_and_initialized():
            dist.destroy_process_group()

    def _mixed_precision_dtype(self) -> torch.dtype | None:
        precision = str(self.cfg.mixed_precision).lower()
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp16":
            return torch.float16
        if precision in {"no", "none", "false"}:
            return None
        raise ValueError(
            "mixed_precision must be one of 'bf16', 'fp16', or 'no', "
            f"got {self.cfg.mixed_precision!r}."
        )
