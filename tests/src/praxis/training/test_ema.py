"""Tests for praxis.training.core.ema."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from praxis.training.core.ema import EMAModel, _tensor_for_shadow


def _make_model() -> nn.Linear:
    model = nn.Linear(4, 4, bias=False)
    nn.init.ones_(model.weight)
    return model


class _TwoParamModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Parameter(torch.ones(2, 2))
        self.frozen = nn.Parameter(torch.zeros(2, 2), requires_grad=False)


class _TwoTrainableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.ones(2, 2))
        self.second = nn.Parameter(torch.full((2, 2), 2.0))


class _ReshardableLinear(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4, bias=False)
        nn.init.ones_(self.weight)
        self.reshard_snapshots: list[torch.Tensor] = []

    def reshard(self) -> None:
        self.reshard_snapshots.append(self.weight.detach().clone())


class Shard:
    pass


class _FakeShardedDTensor:
    def __init__(self, local: torch.Tensor, *, global_shape: tuple[int, ...]) -> None:
        self._local = local
        self.shape = torch.Size(global_shape)
        self.device_mesh = object()
        self.placements = (Shard(),)

    def to_local(self) -> torch.Tensor:
        return self._local

    def full_tensor(self) -> torch.Tensor:
        raise AssertionError("sharded EMA update must not gather full tensors")


class TestEMAModel:
    def test_shadow_initialized_to_model_params(self):
        model = _make_model()
        ema = EMAModel(model, decay=0.99)
        assert torch.allclose(ema.shadow["weight"], model.weight.data)

    def test_shadow_uses_fp32_for_bfloat16_model(self):
        model = _make_model().to(dtype=torch.bfloat16)
        ema = EMAModel(model, decay=0.99)

        assert ema.shadow["weight"].dtype is torch.float32
        assert torch.allclose(
            ema.shadow["weight"],
            model.weight.detach().float(),
        )

    def test_update_moves_shadow_toward_model(self):
        model = _make_model()
        ema = EMAModel(model, decay=0.9)

        # Change model params
        model.weight.data.fill_(2.0)
        ema.update(model)

        # Shadow should have moved from 1.0 toward 2.0
        # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        assert torch.allclose(ema.shadow["weight"], torch.full_like(model.weight, 1.1))

    def test_apply_and_restore(self):
        model = _make_model()
        ema = EMAModel(model, decay=0.5)

        # Modify model, update EMA
        model.weight.data.fill_(3.0)
        ema.update(model)
        # shadow = 0.5 * 1.0 + 0.5 * 3.0 = 2.0

        # Now change model again
        model.weight.data.fill_(5.0)

        # Apply: model should get shadow params
        ema.apply(model)
        assert torch.allclose(model.weight.data, torch.full_like(model.weight, 2.0))

        # Restore: model should get original (5.0) back
        ema.restore(model)
        assert torch.allclose(model.weight.data, torch.full_like(model.weight, 5.0))

    def test_apply_can_keep_restore_backup_on_cpu(self):
        model = _make_model()
        ema = EMAModel(model, decay=0.5)

        model.weight.data.fill_(3.0)
        ema.update(model)
        model.weight.data.fill_(5.0)

        ema.apply(model, backup_device=torch.device("cpu"))

        assert ema.backup["weight"].device.type == "cpu"
        assert ema.backup["weight"].data_ptr() != model.weight.data.data_ptr()
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2.0))

        ema.restore(model)

        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5.0))
        assert not ema.restore_needed

    def test_state_dict_roundtrip(self):
        model = _make_model()
        ema = EMAModel(model, decay=0.99)
        model.weight.data.fill_(2.0)
        ema.update(model)

        sd = ema.state_dict()
        ema2 = EMAModel(model, decay=0.99)
        ema2.load_state_dict(sd)

        assert torch.allclose(ema.shadow["weight"], ema2.shadow["weight"])

    def test_tracks_only_trainable_parameters(self):
        model = _TwoParamModel()
        ema = EMAModel(model, decay=0.9)
        assert "trainable" in ema.shadow
        assert "frozen" not in ema.shadow

    def test_refresh_repoints_parameter_references(self):
        model = _make_model()
        ema = EMAModel(model, decay=0.9)
        replacement = _make_model()

        ema.refresh(replacement)

        assert ema._param_refs[0].data_ptr() == replacement.weight.data.data_ptr()
        assert ema._param_refs[0].data_ptr() != model.weight.data.data_ptr()

    def test_load_state_dict_ignores_untracked_keys(self):
        model = _TwoParamModel()
        ema = EMAModel(model, decay=0.9)
        ema.load_state_dict(
            {
                "trainable": torch.full((2, 2), 3.0),
                "frozen": torch.full((2, 2), 7.0),
            }
        )
        assert "trainable" in ema.shadow
        assert "frozen" not in ema.shadow
        assert torch.allclose(ema.shadow["trainable"], torch.full((2, 2), 3.0))

    def test_load_state_dict_rejects_partial_tracked_state(self):
        model = _TwoTrainableModel()
        ema = EMAModel(model, decay=0.9)
        state = ema.state_dict()
        state.pop("second")

        with pytest.raises(RuntimeError, match="missing tracked parameter.*second"):
            ema.load_state_dict(state)

    def test_load_state_dict_rejects_shape_mismatch(self):
        model = _TwoParamModel()
        ema = EMAModel(model, decay=0.9)

        with pytest.raises(RuntimeError, match="shape mismatch.*trainable"):
            ema.load_state_dict({"trainable": torch.ones(1)})

    def test_load_state_dict_rejects_dtype_mismatch(self):
        model = _TwoParamModel()
        ema = EMAModel(model, decay=0.9)

        with pytest.raises(RuntimeError, match="dtype mismatch.*trainable"):
            ema.load_state_dict({"trainable": torch.ones(2, 2, dtype=torch.float16)})

    def test_restore_reshards_fsdp_modules_before_copy(self):
        model = _ReshardableLinear()
        ema = EMAModel(model, decay=0.9)
        backup = torch.ones_like(model.weight)
        ema.backup = {"weight": backup}
        model.weight.data.fill_(5.0)

        ema.restore(model)

        assert len(model.reshard_snapshots) == 1
        torch.testing.assert_close(
            model.reshard_snapshots[0],
            torch.full_like(model.weight, 5.0),
        )
        torch.testing.assert_close(model.weight, backup)
        assert not ema.restore_needed

    def test_sharded_dtensor_shadow_update_uses_local_tensor(self):
        local = torch.tensor([3.0])
        source = _FakeShardedDTensor(local, global_shape=(1,))
        shadow = torch.zeros(1)

        out = _tensor_for_shadow(source, shadow)  # type: ignore[arg-type]

        torch.testing.assert_close(out, local)
