"""Focused tests for the shared VLM-conditioned action expert."""

from __future__ import annotations

import pytest
import torch

import praxis.policies.core.action_expert.expert as expert_module
from praxis.nn.blocks.mlp import GatedMlp
from praxis.nn.embeddings.positional import RotaryPosEmbed
from praxis.nn.layers.norm import Fp32RMSNorm
from praxis.policies.core.action_expert.expert import ActionExpert, ActionExpertLayer
from praxis.policies.core.action_expert.flow import FlowSample
from praxis.policies.core.action_expert.integrated import (
    VlmActionExpertModel,
    context_position_offsets,
)
from praxis.policies.core.action_expert.prefix_cache import (
    PrefixCache,
    build_prefix_cache,
)
from praxis.policies.core.vlm.prefix import (
    LayerKV,
    VlmAttentionGeometry,
    VlmPrefixState,
)


def _prefix_cache(
    *,
    batch_size: int = 2,
    num_layers: int = 4,
    prefix_len: int = 3,
    vlm_kv_dim: int = 4,
) -> PrefixCache:
    layers = tuple(
        LayerKV(
            key=torch.randn(batch_size, prefix_len, vlm_kv_dim),
            value=torch.randn(batch_size, prefix_len, vlm_kv_dim),
        )
        for _ in range(num_layers)
    )
    mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    return PrefixCache(layers=layers, mask=mask)


def _expert(**overrides) -> ActionExpert:
    kwargs = {
        "action_dim": 3,
        "action_horizon": 4,
        "vlm_num_layers": 4,
        "expert_num_layers": None,
        "vlm_attention": VlmAttentionGeometry(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        ),
        "hidden_size": 8,
        "dropout": 0.0,
        "time_min_period": 4e-3,
        "time_max_period": 4.0,
        "self_attn_every_n_layers": 2,
    }
    kwargs.update(overrides)
    return ActionExpert(**kwargs)


class _RecordingPrefixAdapter:
    def __init__(self) -> None:
        self.recorded_position_ids: torch.Tensor | None = None

    @property
    def num_layers(self) -> int:
        return 1

    @property
    def hidden_size(self) -> int:
        return 8

    @property
    def attention_geometry(self) -> VlmAttentionGeometry:
        return VlmAttentionGeometry(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
        )

    def prepare_prefix(
        self,
        vlm_inputs: dict[str, torch.Tensor],
        *,
        layer_attention_mask: torch.Tensor | None = None,
    ) -> VlmPrefixState:
        del layer_attention_mask
        input_ids = vlm_inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros(batch_size, seq_len, self.hidden_size)
        return VlmPrefixState(
            hidden=hidden,
            mask=torch.ones_like(input_ids, dtype=torch.bool),
            attention_mask=None,
            position_ids=torch.full_like(input_ids, 100),
            cache_position=torch.arange(seq_len),
            position_embeddings=(hidden, hidden),
            extra={"expert_position_ids": torch.arange(seq_len).expand_as(input_ids)},
        )

    def forward_layer(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
    ) -> tuple[VlmPrefixState, LayerKV]:
        del layer_idx
        key = torch.zeros(state.hidden.shape[0], 1, state.hidden.shape[1], 4)
        value = torch.zeros_like(key)
        return state, LayerKV(key=key, value=value)

    def forward_prefix_layer(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
    ) -> VlmPrefixState:
        del layer_idx
        return state

    def project_layer_kv(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
        position_ids: torch.Tensor | None = None,
    ) -> LayerKV:
        del layer_idx
        self.recorded_position_ids = None if position_ids is None else position_ids
        key = torch.zeros(state.hidden.shape[0], 1, state.hidden.shape[1], 4)
        value = torch.zeros_like(key)
        return LayerKV(key=key, value=value)


def test_expert_uses_smolvla_style_primitives() -> None:
    expert = _expert()

    assert expert.expert_num_layers == expert.vlm_num_layers
    assert expert.self_attn_every_n_layers == 2
    assert not hasattr(expert, "position_embedding")
    assert expert.attention_dim == 8
    assert expert.kv_dim == 4
    assert isinstance(expert.norm, Fp32RMSNorm)
    layer = expert.layers[0]
    assert layer.attention_kind == "self_attn"
    assert isinstance(layer.input_layernorm, Fp32RMSNorm)
    assert isinstance(layer.post_attention_layernorm, Fp32RMSNorm)
    assert isinstance(layer.mlp, GatedMlp)
    assert layer.mlp.fc1.out_features == int(8 * expert.hidden_size / 3)
    assert isinstance(layer.rope, RotaryPosEmbed)
    assert layer.rope.pairing == "split"
    assert expert.layers[1].attention_kind == "cross_attn"
    self_attn_k_proj = expert.layers[0].k_proj
    cross_attn_k_proj = expert.layers[1].k_proj
    assert isinstance(self_attn_k_proj, torch.nn.Linear)
    assert isinstance(cross_attn_k_proj, torch.nn.Linear)
    assert self_attn_k_proj.in_features == expert.hidden_size
    assert cross_attn_k_proj.in_features == expert.kv_dim


def test_prefix_cache_detach_flag_controls_gradient_boundary() -> None:
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    key = hidden * 2.0
    value = hidden * 3.0
    prefix_mask = torch.tensor(
        [[True, True, False], [True, False, False]],
        dtype=torch.bool,
    )

    detached = build_prefix_cache(
        past_key_values=((key, value),),
        prefix_mask=prefix_mask,
        detach=True,
        suffix_position_offsets=torch.tensor([2, 1]),
    )
    differentiable = build_prefix_cache(
        past_key_values=((key, value),),
        prefix_mask=prefix_mask,
        detach=False,
        suffix_position_offsets=torch.tensor([2, 1]),
    )

    assert not detached.layers[0].key.requires_grad
    assert detached.suffix_position_offsets is not None
    assert not detached.suffix_position_offsets.requires_grad
    assert differentiable.layers[0].key.requires_grad
    differentiable.layers[0].key.sum().backward()
    assert hidden.grad is not None
    assert hidden.grad.abs().sum().item() > 0.0


def test_expert_layer_count_matches_smolvla_constraints() -> None:
    expert = _expert(expert_num_layers=2)

    assert expert.layer_stride == 2
    assert [expert._expert_layer_index(i) for i in range(4)] == [
        0,
        None,
        1,
        None,
    ]
    assert [expert._attention_kind(i) for i in range(4)] == [
        "self_attn",
        "cross_attn",
        "self_attn",
        "cross_attn",
    ]
    assert [layer.attention_kind for layer in expert.layers] == [
        "self_attn",
        "self_attn",
    ]
    with pytest.raises(ValueError, match="divisible"):
        _expert(vlm_num_layers=3, expert_num_layers=2)
    with pytest.raises(ValueError, match="<="):
        _expert(vlm_num_layers=2, expert_num_layers=3)


def test_expert_positions_follow_smolvla_offsets() -> None:
    expert_pad_mask = torch.tensor([[True, True, True, True]])
    prefix_mask = torch.tensor([[True, True, False]])

    cross_positions = ActionExpert._expert_positions(
        expert_pad_mask=expert_pad_mask,
        prefix_mask=prefix_mask,
        offset_after_prefix=False,
    )
    self_positions = ActionExpert._expert_positions(
        expert_pad_mask=expert_pad_mask,
        prefix_mask=prefix_mask,
        offset_after_prefix=True,
    )

    assert cross_positions.tolist() == [[0, 1, 2, 3]]
    assert self_positions.tolist() == [[2, 3, 4, 5]]


def test_expert_positions_accept_backbone_specific_suffix_offsets() -> None:
    expert_pad_mask = torch.tensor([[True, True, True], [True, True, True]])
    prefix_mask = torch.tensor(
        [[False, True, True], [True, True, True]],
        dtype=torch.bool,
    )

    positions = ActionExpert._expert_positions(
        expert_pad_mask=expert_pad_mask,
        prefix_mask=prefix_mask,
        prefix_position_offsets=torch.tensor([3, 4]),
        offset_after_prefix=True,
    )

    assert positions.tolist() == [[3, 4, 5], [4, 5, 6]]


def test_context_offsets_use_backbone_expert_position_ids() -> None:
    state = VlmPrefixState(
        hidden=torch.zeros(2, 3, 4),
        mask=torch.tensor([[False, True, True], [True, True, True]]),
        attention_mask=None,
        position_ids=torch.tensor([[0, 0, 1], [0, 1, 2]]),
        cache_position=torch.arange(3),
        position_embeddings=(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)),
        extra={
            "expert_position_ids": torch.tensor(
                [
                    [[1, 1, 2], [1, 2, 3]],
                    [[1, 50, 51], [1, 2, 3]],
                    [[1, 4, 5], [1, 2, 9]],
                ],
                dtype=torch.long,
            )
        },
    )

    offsets = context_position_offsets(
        state,
        context_mask=torch.tensor([[False, True, True], [True, True, True]]),
    )

    assert offsets.tolist() == [52, 10]


def test_context_forward_reprojects_kv_with_expert_position_ids() -> None:
    adapter = _RecordingPrefixAdapter()
    model = VlmActionExpertModel(
        prefix_adapter=adapter,
        action_expert=_expert(
            action_dim=2,
            action_horizon=3,
            vlm_num_layers=1,
            expert_num_layers=1,
            hidden_size=8,
        ),
    )
    sample = FlowSample(
        noisy_actions=torch.zeros(2, 3, 2),
        target_velocity=torch.zeros(2, 3, 2),
        timesteps=torch.full((2,), 0.5),
        noise=torch.zeros(2, 3, 2),
    )

    model.loss_with_context(
        vlm_inputs={"input_ids": torch.ones(2, 4, dtype=torch.long)},
        expert_context_mask=torch.ones(2, 4, dtype=torch.bool),
        sample=sample,
        detach_expert_context=True,
        gradient_checkpointing=False,
    )

    assert adapter.recorded_position_ids is not None
    assert adapter.recorded_position_ids.tolist() == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]


def test_action_is_pad_masks_loss_not_expert_token_stream() -> None:
    expert = _expert()
    hidden, expert_pad_mask, expert_attention_mask = expert.prepare_suffix_inputs(
        noisy_actions=torch.randn(2, 4, 3),
        timesteps=torch.tensor([0.2, 0.8]),
        action_is_pad=torch.tensor(
            [
                [False, False, True, True],
                [False, True, False, True],
            ],
            dtype=torch.bool,
        ),
    )

    assert expert_pad_mask.tolist() == [
        [True, True, True, True],
        [True, True, True, True],
    ]
    positions = ActionExpert._expert_positions(
        expert_pad_mask=expert_pad_mask,
        prefix_mask=torch.tensor([[True, True, True], [True, True, False]]),
        offset_after_prefix=True,
    )
    assert positions.tolist() == [
        [3, 4, 5, 6],
        [2, 3, 4, 5],
    ]
    assert expert_attention_mask.tolist() == [
        [True, True, True, True],
        [True, True, True, True],
    ]
    pred = torch.ones((1, 4, 3), dtype=torch.float32)
    target = torch.zeros_like(pred)
    assert (
        expert_module.masked_action_mse_loss(
            pred,
            target,
            action_is_pad=torch.tensor([[False, True, True, True]], dtype=torch.bool),
        ).item()
        == 1.0
    )
    assert torch.count_nonzero(hidden).item() > 0


def test_action_is_pad_shape_is_validated_for_suffix_inputs() -> None:
    expert = _expert()

    with pytest.raises(ValueError, match="action_is_pad must have shape"):
        expert.prepare_suffix_inputs(
            noisy_actions=torch.randn(2, 4, 3),
            timesteps=torch.tensor([0.2, 0.8]),
            action_is_pad=torch.zeros((2, 3), dtype=torch.bool),
        )


def test_action_pad_mask_can_exclude_suffix_attention_when_requested(
    monkeypatch,
) -> None:
    layer = ActionExpertLayer(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_kind="self_attn",
        dropout=0.0,
    )
    hidden = torch.randn(2, 5, 8)
    layer_kv = _prefix_cache(num_layers=1).layers[0]
    prefix_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    expert_pad_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, False, True, False],
        ],
        dtype=torch.bool,
    )
    expert_attention_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    positions = ActionExpert._expert_positions(
        expert_pad_mask=expert_pad_mask,
        prefix_mask=prefix_mask,
        offset_after_prefix=True,
    )
    masks: list[torch.Tensor] = []

    def _fake_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        del key, value, dropout_p
        assert attn_mask is not None
        masks.append(attn_mask)
        return torch.zeros_like(query)

    monkeypatch.setattr(
        expert_module.F,
        "scaled_dot_product_attention",
        _fake_sdpa,
    )

    layer(
        hidden=hidden,
        layer_kv=layer_kv,
        prefix_mask=prefix_mask,
        expert_pad_mask=expert_pad_mask,
        expert_attention_mask=expert_attention_mask,
        expert_positions=positions,
    )

    assert masks[0].shape == torch.Size([2, 1, 5, 8])
    assert masks[0][0, 0, 3, :3].tolist() == [True, True, True]
    assert not masks[0][0, 0, 3, 3:].any()
    assert masks[0][1, 0, 2, :2].tolist() == [True, True]
    assert not masks[0][1, 0, 2, 2:].any()


def test_gqa_key_value_expansion_preserves_sequence_axis() -> None:
    layer = ActionExpertLayer(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        attention_kind="cross_attn",
        dropout=0.0,
    )
    kv = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)

    expanded = layer._expand_key_value_heads(kv)

    assert expanded.shape == (1, 4, 3, 2)
    assert torch.equal(expanded[:, 0], kv[:, 0])
    assert torch.equal(expanded[:, 1], kv[:, 0])
    assert torch.equal(expanded[:, 2], kv[:, 1])
    assert torch.equal(expanded[:, 3], kv[:, 1])


def test_forward_runs_with_smolvla_layer_schedule() -> None:
    expert = _expert()
    seen: list[str] = []

    def _record_attention_kind(module, _args, _kwargs) -> None:
        seen.append(str(module.attention_kind))

    handles = [
        layer.register_forward_pre_hook(_record_attention_kind, with_kwargs=True)
        for layer in expert.layers
    ]
    try:
        out = expert(
            cache=_prefix_cache(),
            noisy_actions=torch.randn(2, 4, 3),
            timesteps=torch.tensor([0.2, 0.8]),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert out.shape == (2, 4, 3)
    assert torch.isfinite(out).all()
    assert seen == ["self_attn", "cross_attn", "self_attn", "cross_attn"]


def test_layer_uses_sdpa_with_cross_and_self_key_lengths(monkeypatch) -> None:
    layer = ActionExpertLayer(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_kind="self_attn",
        dropout=0.0,
    )
    cross_layer = ActionExpertLayer(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_kind="cross_attn",
        dropout=0.0,
    )
    hidden = torch.randn(2, 5, 8)
    layer_kv = _prefix_cache(num_layers=1).layers[0]
    prefix_mask = torch.tensor(
        [[True, True, True], [True, True, False]],
        dtype=torch.bool,
    )
    expert_pad_mask = torch.ones((2, 5), dtype=torch.bool)
    expert_attention_mask = torch.tensor(
        [
            [False, True, True, True, True],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    positions = ActionExpert._expert_positions(
        expert_pad_mask=expert_pad_mask,
        prefix_mask=prefix_mask,
        offset_after_prefix=True,
    )
    cross_positions = ActionExpert._expert_positions(
        expert_pad_mask=expert_pad_mask,
        prefix_mask=prefix_mask,
        offset_after_prefix=False,
    )
    calls: list[tuple[torch.Size, torch.Size, torch.Tensor | None]] = []

    def _fake_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> torch.Tensor:
        del value, dropout_p
        calls.append((query.shape, key.shape, attn_mask))
        return torch.zeros_like(query)

    monkeypatch.setattr(
        expert_module.F,
        "scaled_dot_product_attention",
        _fake_sdpa,
    )

    layer(
        hidden=hidden,
        layer_kv=layer_kv,
        prefix_mask=prefix_mask,
        expert_pad_mask=expert_pad_mask,
        expert_attention_mask=expert_attention_mask,
        expert_positions=positions,
    )
    cross_layer(
        hidden=hidden,
        layer_kv=layer_kv,
        prefix_mask=prefix_mask,
        expert_pad_mask=expert_pad_mask,
        expert_attention_mask=expert_attention_mask,
        expert_positions=cross_positions,
    )

    assert calls[0][0] == torch.Size([2, 2, 5, 4])
    assert calls[0][1] == torch.Size([2, 2, 8, 4])
    assert calls[1][0] == torch.Size([2, 2, 5, 4])
    assert calls[1][1] == torch.Size([2, 2, 3, 4])
    assert calls[0][2] is not None
    assert calls[0][2].shape == torch.Size([2, 1, 5, 8])
    assert calls[0][2][0, 0, 1, 3:].tolist() == [True, True, False, False, False]
