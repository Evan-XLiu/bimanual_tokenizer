from __future__ import annotations

import os
import random
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, Dataset, TensorDataset

from praxis.training.core.distributed import (
    DistributedRuntimeConfig,
    TrainingRuntime,
    _DistributedBatchSampler,
    is_distributed_padding_batch,
    resolve_fsdp_reshard_after_forward,
)
from praxis.training.core.ema import EMAModel


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _NoTrainableParamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FSDPLikeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, bool]] = []

    def set_requires_gradient_sync(self, enabled: bool) -> None:
        self.calls.append(("gradient_sync", bool(enabled)))

    def set_requires_all_reduce(self, enabled: bool) -> None:
        self.calls.append(("all_reduce", bool(enabled)))

    def set_reshard_after_backward(self, enabled: bool) -> None:
        self.calls.append(("reshard_after_backward", bool(enabled)))

    def set_is_last_backward(self, enabled: bool) -> None:
        self.calls.append(("is_last_backward", bool(enabled)))


class _FrozenScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.scale


class _RandomProbeDataset(Dataset):
    def __init__(self, n: int) -> None:
        self.n = int(n)
        self.random_values: list[tuple[float, float]] = []

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        self.random_values.append((random.random(), float(torch.rand(()))))
        return int(idx)


@pytest.fixture(autouse=True)
def _clear_torchrun_env(monkeypatch: pytest.MonkeyPatch):
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(key, raising=False)


def _runtime(
    *,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "no",
) -> TrainingRuntime:
    return TrainingRuntime(
        cfg=DistributedRuntimeConfig(
            strategy="ddp",
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        requested_device="cpu",
    )


def test_runtime_init_single_process_cpu() -> None:
    runtime = _runtime()
    assert runtime.is_main_process is True
    assert runtime.num_processes == 1
    assert runtime.device == torch.device("cpu")


def test_prepare_model_keeps_single_process_model_unwrapped() -> None:
    runtime = _runtime()
    model = _DummyModel()

    prepared = runtime.prepare_model(model)

    assert prepared is model
    assert runtime.unwrap_model(prepared) is model


def test_prepare_ddp_enables_gradient_bucket_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeDDP(nn.Module):
        def __init__(self, module: nn.Module, **kwargs) -> None:
            super().__init__()
            self.module = module
            captured.update(kwargs)

    monkeypatch.setattr(
        "praxis.training.core.distributed.DistributedDataParallel",
        _FakeDDP,
    )
    runtime = _runtime()
    runtime.world_size = 2

    prepared = runtime.prepare_model(_DummyModel())

    assert isinstance(prepared, _FakeDDP)
    assert captured["gradient_as_bucket_view"] is True


def test_prepare_ddp_skips_wrap_when_not_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_ddp(*_args, **_kwargs) -> None:
        raise AssertionError("DDP should not wrap non-training models")

    monkeypatch.setattr(
        "praxis.training.core.distributed.DistributedDataParallel",
        _fail_ddp,
    )
    runtime = _runtime()
    runtime.world_size = 2
    model = _DummyModel()

    prepared = runtime.prepare_model(model, wrap_for_training=False)

    assert prepared is model


def test_prepare_ddp_skips_wrap_without_trainable_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_ddp(*_args, **_kwargs) -> None:
        raise AssertionError("DDP should not wrap models without trainable params")

    monkeypatch.setattr(
        "praxis.training.core.distributed.DistributedDataParallel",
        _fail_ddp,
    )
    runtime = _runtime()
    runtime.world_size = 2
    model = _NoTrainableParamModel()

    prepared = runtime.prepare_model(model)

    assert prepared is model


def test_backward_and_gradient_clipping() -> None:
    runtime = _runtime()
    model = runtime.prepare_model(_DummyModel())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss = model(torch.randn(2, 4)).sum()
    runtime.backward(loss)
    runtime.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total_sq_norm = torch.zeros(())
    for param in model.parameters():
        if param.grad is not None:
            total_sq_norm = total_sq_norm + param.grad.norm() ** 2
    total_norm = torch.sqrt(total_sq_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    assert total_norm.item() <= 1.0 + 1e-6


def test_gradient_accumulation_sync_flag_and_loss_scaling() -> None:
    runtime = _runtime(gradient_accumulation_steps=2)
    model = runtime.prepare_model(_DummyModel())

    first_loss = model(torch.ones(1, 4)).sum()
    with runtime.accumulate(model):
        assert runtime.sync_gradients is False
        runtime.backward(first_loss)
    runtime.mark_micro_step_complete()

    second_loss = model(torch.ones(1, 4)).sum()
    with runtime.accumulate(model):
        assert runtime.sync_gradients is True
        runtime.backward(second_loss)

    assert any(p.grad is not None for p in model.parameters())


def test_gradient_accumulation_scales_loss_to_match_full_batch() -> None:
    runtime = _runtime(gradient_accumulation_steps=2)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)

    runtime.backward(model(torch.ones(1, 1)).sum())
    runtime.mark_micro_step_complete()
    runtime.backward(model(torch.ones(1, 1)).sum())

    assert model.weight.grad is not None
    assert torch.allclose(model.weight.grad, torch.tensor([[1.0]]))


def test_mixed_precision_dtype_modes() -> None:
    assert _runtime(mixed_precision="bf16")._mixed_precision_dtype() is torch.bfloat16
    assert _runtime(mixed_precision="fp16")._mixed_precision_dtype() is torch.float16
    assert _runtime(mixed_precision="no")._mixed_precision_dtype() is None


def test_resolve_fsdp_reshard_after_forward_preserves_bool_or_int() -> None:
    assert resolve_fsdp_reshard_after_forward(True) is True
    assert resolve_fsdp_reshard_after_forward("false") is False
    assert resolve_fsdp_reshard_after_forward(2) == 2
    assert resolve_fsdp_reshard_after_forward("4") == 4


def test_ema_uses_unwrapped_single_process_model() -> None:
    runtime = _runtime()
    model = runtime.prepare_model(_DummyModel())
    unwrapped = runtime.unwrap_model(model)

    ema = EMAModel(unwrapped, decay=0.99)
    loss = model(torch.randn(2, 4)).sum()
    runtime.backward(loss)
    ema.update(unwrapped)

    for name, param in unwrapped.named_parameters():
        if name in ema.shadow:
            assert ema.shadow[name].shape == param.shape


def test_prepare_dataloader_noops_single_process() -> None:
    runtime = _runtime()
    dataloader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)

    assert runtime.prepare_dataloader(dataloader) is dataloader


def _distributed_batches(
    *,
    num_samples: int,
    batch_size: int,
    world_size: int,
    rank: int,
) -> list[list[int]]:
    dataloader = DataLoader(
        TensorDataset(torch.arange(num_samples)),
        batch_size=batch_size,
        drop_last=False,
    )
    sampler = _DistributedBatchSampler(
        cast(BatchSampler, dataloader.batch_sampler),
        rank=rank,
        world_size=world_size,
    )
    return list(sampler)


def test_distributed_batch_sampler_streams_rank_batches() -> None:
    ranks = [
        _distributed_batches(
            num_samples=5,
            batch_size=2,
            world_size=2,
            rank=rank,
        )
        for rank in range(2)
    ]

    assert ranks == [
        [[0, 1], [4]],
        [[2, 3], [0, 1]],
    ]


def test_distributed_batch_sampler_owns_sampler_rng() -> None:
    seed = 123
    num_samples = 12
    batch_size = 2
    world_size = 2
    reference_generator = torch.Generator()
    reference_generator.manual_seed(seed)
    reference_loader = DataLoader(
        TensorDataset(torch.arange(num_samples)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=reference_generator,
    )
    expected_order = [
        list(batch) for batch in cast(BatchSampler, reference_loader.batch_sampler)
    ]

    ranks: list[list[list[int]]] = []
    for rank, drift_seed in enumerate((111, 999)):
        dataloader = DataLoader(
            TensorDataset(torch.arange(num_samples)),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        sampler = _DistributedBatchSampler(
            cast(BatchSampler, dataloader.batch_sampler),
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
        torch.manual_seed(drift_seed)
        ranks.append(list(sampler))

    interleaved = [
        ranks[rank][local_index]
        for local_index in range(len(ranks[0]))
        for rank in range(world_size)
    ]
    assert interleaved == expected_order


def test_distributed_batch_sampler_does_not_leak_sampler_rng_to_dataset_work() -> None:
    seed = 123
    dataset = _RandomProbeDataset(8)
    source_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    sampler = _DistributedBatchSampler(
        cast(BatchSampler, source_loader.batch_sampler),
        rank=0,
        world_size=2,
        seed=seed,
    )

    random.seed(999)
    torch.manual_seed(999)
    prepared = DataLoader(dataset, batch_sampler=sampler)
    prepared_iter = iter(prepared)
    py_state = random.getstate()
    torch_state = torch.get_rng_state()
    expected_random_values = [
        (random.random(), float(torch.rand(()))) for _ in range(2)
    ]
    random.setstate(py_state)
    torch.set_rng_state(torch_state)

    next(prepared_iter)

    assert dataset.random_values == pytest.approx(expected_random_values)


def test_distributed_batch_sampler_marks_padding_batches() -> None:
    dataloader = DataLoader(
        TensorDataset(torch.arange(5)),
        batch_size=2,
        drop_last=False,
    )
    sampler = _DistributedBatchSampler(
        cast(BatchSampler, dataloader.batch_sampler),
        rank=1,
        world_size=2,
    )
    prepared = DataLoader(
        dataloader.dataset,
        batch_sampler=sampler,
    )

    assert sampler.num_real_batches == 1
    assert is_distributed_padding_batch(prepared, 0) is False
    assert is_distributed_padding_batch(prepared, 1) is True


def test_distributed_batch_sampler_pads_each_rank_when_batches_lt_world_size() -> None:
    ranks = [
        _distributed_batches(
            num_samples=2,
            batch_size=2,
            world_size=4,
            rank=rank,
        )
        for rank in range(4)
    ]

    assert ranks == [
        [[0, 1]],
        [[0, 1]],
        [[0, 1]],
        [[0, 1]],
    ]


def test_fsdp_accumulation_sets_all_sync_controls() -> None:
    runtime = _runtime(gradient_accumulation_steps=2)
    model = _FSDPLikeModel()

    with runtime.accumulate(model):
        pass
    runtime.mark_micro_step_complete()
    with runtime.accumulate(model):
        pass

    assert model.calls == [
        ("gradient_sync", False),
        ("all_reduce", False),
        ("reshard_after_backward", False),
        ("is_last_backward", False),
        ("gradient_sync", True),
        ("all_reduce", True),
        ("reshard_after_backward", True),
        ("is_last_backward", True),
        ("gradient_sync", True),
        ("all_reduce", True),
        ("reshard_after_backward", True),
        ("is_last_backward", True),
        ("gradient_sync", True),
        ("all_reduce", True),
        ("reshard_after_backward", True),
        ("is_last_backward", True),
    ]


def test_broadcast_object_noops_single_process() -> None:
    runtime = _runtime()
    payload = {"checkpoint_token": "abc123"}

    assert runtime.broadcast_object(payload, src=0) == payload


def test_main_process_first_broadcasts_main_rank_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.global_rank = 0
    broadcasts: list[object] = []

    monkeypatch.setattr(
        "praxis.training.core.distributed._dist_available_and_initialized",
        lambda: True,
    )
    runtime.broadcast_object = (  # type: ignore[method-assign]
        lambda obj, *, src=0: broadcasts.append(obj) or obj
    )
    runtime.gather_object = lambda obj: [obj]  # type: ignore[method-assign]

    with (
        pytest.raises(RuntimeError, match="main_process_first failed on rank 0"),
        runtime.main_process_first(),
    ):
        raise ValueError("download failed")

    assert broadcasts and "download failed" in str(broadcasts[0])


def test_main_process_first_gathers_non_main_rank_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.global_rank = 1

    monkeypatch.setattr(
        "praxis.training.core.distributed._dist_available_and_initialized",
        lambda: True,
    )
    runtime.broadcast_object = lambda obj, *, src=0: None  # type: ignore[method-assign]
    runtime.gather_object = lambda obj: [obj]  # type: ignore[method-assign]

    with (
        pytest.raises(RuntimeError, match="main_process_first failed on rank 1"),
        runtime.main_process_first(),
    ):
        raise ValueError("load failed")


def test_env_rank_values_are_read_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")

    runtime = _runtime()

    assert runtime.global_rank == int(os.environ["RANK"])
    assert runtime.local_rank == int(os.environ["LOCAL_RANK"])
    assert runtime.num_processes == int(os.environ["WORLD_SIZE"])


def test_fsdp_requires_multi_rank_torchrun(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "praxis.training.core.distributed._resolve_device",
        lambda requested_device, local_rank: torch.device("cuda", 0),
    )

    with pytest.raises(ValueError, match="WORLD_SIZE > 1"):
        TrainingRuntime(
            cfg=DistributedRuntimeConfig(strategy="fsdp", mixed_precision="no"),
            requested_device="cuda",
        )


def test_fsdp_gradient_clipping_uses_replicated_global_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.cfg = DistributedRuntimeConfig(strategy="fsdp", mixed_precision="no")
    runtime.device = torch.device("cpu")
    runtime.global_rank = 0

    monkeypatch.setattr(
        "praxis.training.core.distributed._dist_available_and_initialized",
        lambda: True,
    )

    def fake_all_reduce(tensor, op):
        del op
        tensor.mul_(2.0)

    monkeypatch.setattr(
        "praxis.training.core.distributed.dist.all_reduce", fake_all_reduce
    )

    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.tensor([3.0, 4.0])
    total_norm = runtime.clip_grad_norm_([param], max_norm=(50.0**0.5) / 2.0)

    assert total_norm.item() == pytest.approx(50.0**0.5)
    torch.testing.assert_close(param.grad, torch.tensor([1.5, 2.0]))


def test_prepare_fsdp_moves_ignored_parameters_after_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.cfg = DistributedRuntimeConfig(strategy="fsdp", mixed_precision="no")
    runtime.device = torch.device("cpu")
    runtime.device_mesh = cast(Any, object())
    model = _FrozenScalarModel()

    fully_shard_calls: list[set[nn.Parameter]] = []
    moved: list[set[nn.Parameter]] = []

    def fake_fully_shard(module, **kwargs):
        del module
        fully_shard_calls.append(set(kwargs.get("ignored_params") or ()))

    def fake_move(params, device):
        del device
        moved.append(set(params))

    monkeypatch.setattr("torch.distributed.fsdp.fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        "praxis.training.core.distributed._move_parameters_to_device",
        fake_move,
    )

    assert runtime._prepare_fsdp_model(model) is model
    assert any(model.scale in params for params in fully_shard_calls)
    assert moved == [{model.scale}]


def test_prepare_fsdp_registers_policy_forward_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = TrainingRuntime.__new__(TrainingRuntime)
    runtime.cfg = DistributedRuntimeConfig(strategy="fsdp", mixed_precision="no")
    runtime.device = torch.device("cpu")
    runtime.device_mesh = cast(Any, object())

    class _PolicyLike(_DummyModel):
        @staticmethod
        def fsdp_forward_methods() -> tuple[str, ...]:
            return ("predict_action_chunk",)

        def predict_action_chunk(self, obs):
            return self.forward(obs), {}

    model = _PolicyLike()
    registered: list[tuple[nn.Module, str]] = []

    def fake_register(module, method_name: str) -> None:
        registered.append((module, method_name))

    monkeypatch.setattr(
        "torch.distributed.fsdp.fully_shard", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "torch.distributed.fsdp.register_fsdp_forward_method",
        fake_register,
    )

    runtime._prepare_fsdp_model(model)

    assert (model, "predict_action_chunk") in registered
