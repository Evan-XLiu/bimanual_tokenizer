"""Tests for action-token runtime helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from praxis.policies.core.vlm.checkpoint import (
    filter_frozen_vlm_state_dict,
    validate_filtered_load_state,
)
from praxis.policies.core.vlm.lora import build_lora_optim_param_groups
from praxis.policies.core.vlm.prefix import VlmPrefixState
from praxis.policies.core.vlm.supervision import (
    IGNORE_INDEX,
    build_eos_supervision_targets,
)
from praxis.policies.core.vlm.training import (
    remap_labels_to_candidate_token_indices,
    run_action_token_ce_prefix_training_step,
)
from praxis.policies.core.vlm.vocab import ActionTokenVocabulary


def _dummy_action_vocab() -> ActionTokenVocabulary:
    return ActionTokenVocabulary(
        tokens=tuple(f"<|action_{i}|>" for i in range(11)),
        token_ids=tuple(range(10, 21)),
        mask_token="<|action_mask|>",
        mask_token_id=1000,
    )


def test_remap_labels_to_candidate_token_indices_maps_action_eos_and_ignore() -> None:
    candidate_token_ids = torch.tensor([100, 101, 102, 103, 1])
    labels = torch.tensor(
        [
            [100, 102, IGNORE_INDEX, 1],
            [103, 101, 102, IGNORE_INDEX],
        ]
    )

    mapped = remap_labels_to_candidate_token_indices(
        labels,
        candidate_token_ids=candidate_token_ids,
        ignore_index=IGNORE_INDEX,
        owner_name="UnitTestPolicy",
    )

    torch.testing.assert_close(
        mapped,
        torch.tensor(
            [
                [0, 2, IGNORE_INDEX, 4],
                [3, 1, 2, IGNORE_INDEX],
            ]
        ),
    )


def test_remap_labels_to_candidate_token_indices_rejects_out_of_scope_labels() -> None:
    candidate_token_ids = torch.tensor([100, 101, 102, 1])
    labels = torch.tensor([100, 1000, IGNORE_INDEX])

    with pytest.raises(ValueError, match="outside ce_vocab_scope='action'.*1000"):
        remap_labels_to_candidate_token_indices(
            labels,
            candidate_token_ids=candidate_token_ids,
            ignore_index=IGNORE_INDEX,
            owner_name="UnitTestPolicy",
        )


class _DummyCeMixin:
    _SCHEDULE_STRIP_KEYS = ("labels",)
    _supervise_eos = False
    _eos_token_id = None
    config = SimpleNamespace(block_ce_weighting="token_mean", ce_vocab_scope="full")
    processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=None))

    def _get_action_token_lengths(self, token_ids_batch):
        del token_ids_batch

    def _resolve_action_token_horizon(self, token_ids_batch, *, token_lengths):
        return max(
            max((len(ids) for ids in token_ids_batch), default=0),
            max((int(length) for length in token_lengths), default=0),
            1,
        )


class _DummyPrefixAdapter:
    def __init__(self, logits: torch.Tensor) -> None:
        self._logits = logits

    @property
    def num_layers(self) -> int:
        return 0

    @property
    def hidden_size(self) -> int:
        return int(self._logits.shape[-1])

    def prepare_prefix(
        self,
        vlm_inputs: dict[str, torch.Tensor],
        *,
        layer_attention_mask: torch.Tensor | None = None,
    ) -> VlmPrefixState:
        input_ids = vlm_inputs["input_ids"]
        hidden = self._logits.to(device=input_ids.device).clone()
        mask = vlm_inputs["attention_mask"].to(
            device=input_ids.device, dtype=torch.bool
        )
        position_ids = torch.arange(
            int(input_ids.shape[1]),
            device=input_ids.device,
            dtype=torch.long,
        ).expand_as(input_ids)
        return VlmPrefixState(
            hidden=hidden,
            mask=mask,
            attention_mask=layer_attention_mask,
            position_ids=position_ids,
            cache_position=position_ids[0],
            position_embeddings=(hidden, hidden),
        )

    def forward_prefix_layer(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
    ) -> VlmPrefixState:
        del layer_idx
        return state

    def final_hidden_to_logits(
        self,
        hidden: torch.Tensor,
        *,
        candidate_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_token_ids is None:
            return hidden
        return hidden.index_select(-1, candidate_token_ids.to(hidden.device))


def test_run_action_token_ce_prefix_training_step_uses_unshifted_ce():
    vocab = 21
    logits = torch.full((1, 5, vocab), -10.0)
    logits[0, 2, 11] = 10.0
    logits[0, 3, 12] = 10.0

    class DummyBlockPolicy(_DummyCeMixin):
        _PREFIX_ATTENTION = "bidirectional"

        def __init__(self) -> None:
            self.normalize = None
            self.io = SimpleNamespace(action_feature=SimpleNamespace(key="action"))
            self.training = True
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.action_token_vocabulary = _dummy_action_vocab()
            self._mask_sentinel = 999
            self._action_mask_token_id = 1000
            self._action_mask_token = "<|action_mask|>"

            class _VLM:
                def __init__(self, owner):
                    self.owner = owner

                def parameters(self):
                    yield self.owner._param

            self.vlm = _VLM(self)

        def _resolve_action_token_block_schedule(self, *, token_horizon):
            del token_horizon
            return (1, 1)

        def tokenize_actions_to_policy_tokens(self, actions):
            del actions
            return [[1, 2]]

        def _require_action_token_vocabulary(self):
            return self.action_token_vocabulary

        def _require_prefix_attention(self):
            return self._PREFIX_ATTENTION

        def _build_vlm_inputs(self, batch, *, action_token_strings, instructions):
            del batch, action_token_strings, instructions
            return {
                "input_ids": torch.tensor([[42, 43, 1000, 11, 1]], dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }

        def _autocast_context(self):
            from contextlib import nullcontext

            return nullcontext()

    policy = DummyBlockPolicy()
    loss, info = run_action_token_ce_prefix_training_step(
        policy,
        {"action": torch.zeros(1, 1, 1)},
        prefix_adapter=_DummyPrefixAdapter(logits),
        ignore_index=-100,
    )
    assert info == {}
    assert float(loss.item()) < 0.01


def test_run_action_token_ce_prefix_training_step_shifted_ce_would_fail():
    vocab = 21
    logits = torch.full((1, 5, vocab), -10.0)
    logits[0, 2, 12] = 10.0
    logits[0, 3, 11] = 10.0

    class DummyBlockPolicy(_DummyCeMixin):
        _PREFIX_ATTENTION = "bidirectional"

        def __init__(self) -> None:
            self.normalize = None
            self.io = SimpleNamespace(action_feature=SimpleNamespace(key="action"))
            self.training = True
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.action_token_vocabulary = _dummy_action_vocab()
            self._mask_sentinel = 999
            self._action_mask_token_id = 1000
            self._action_mask_token = "<|action_mask|>"

            class _VLM:
                def __init__(self, owner):
                    self.owner = owner

                def parameters(self):
                    yield self.owner._param

            self.vlm = _VLM(self)

        def _resolve_action_token_block_schedule(self, *, token_horizon):
            del token_horizon
            return (1, 1)

        def tokenize_actions_to_policy_tokens(self, actions):
            del actions
            return [[1, 2]]

        def _require_action_token_vocabulary(self):
            return self.action_token_vocabulary

        def _require_prefix_attention(self):
            return self._PREFIX_ATTENTION

        def _build_vlm_inputs(self, batch, *, action_token_strings, instructions):
            del batch, action_token_strings, instructions
            return {
                "input_ids": torch.tensor([[42, 43, 1000, 11, 1]], dtype=torch.long),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }

        def _autocast_context(self):
            from contextlib import nullcontext

            return nullcontext()

    policy = DummyBlockPolicy()
    loss, info = run_action_token_ce_prefix_training_step(
        policy,
        {"action": torch.zeros(1, 1, 1)},
        prefix_adapter=_DummyPrefixAdapter(logits),
        ignore_index=-100,
    )
    assert info == {}
    assert float(loss.item()) > 5.0


def _run_blockwise_training_loss_for_test(
    *,
    block_schedule: tuple[int, ...],
    raw_token_ids: list[list[int]],
    input_ids: torch.Tensor,
    logits: torch.Tensor,
    weighting_mode: str,
    supervise_eos: bool = False,
    token_lengths: list[int] | None = None,
    eos_token_id: int | None = None,
) -> tuple[float, torch.Tensor]:
    class DummyBlockPolicy(_DummyCeMixin):
        _PREFIX_ATTENTION = "bidirectional"

        def __init__(self) -> None:
            self.normalize = None
            self.io = SimpleNamespace(action_feature=SimpleNamespace(key="action"))
            self.training = True
            self.config = SimpleNamespace(
                block_ce_weighting=weighting_mode,
                ce_vocab_scope="full",
            )
            self._param = torch.nn.Parameter(torch.zeros(1))
            self.action_token_vocabulary = _dummy_action_vocab()
            self._mask_sentinel = 999
            self._action_mask_token_id = 1000
            self._action_mask_token = "<|action_mask|>"
            self._supervise_eos = supervise_eos
            self._eos_token_id = eos_token_id
            if token_lengths is not None:
                self._get_action_token_lengths = lambda token_ids_batch: list(
                    token_lengths
                )

            class _VLM:
                def __init__(self, owner):
                    self.owner = owner

                def parameters(self):
                    yield self.owner._param

            self.vlm = _VLM(self)

        def _resolve_action_token_block_schedule(self, *, token_horizon):
            del token_horizon
            return block_schedule

        def tokenize_actions_to_policy_tokens(self, actions):
            del actions
            return [list(ids) for ids in raw_token_ids]

        def _require_action_token_vocabulary(self):
            return self.action_token_vocabulary

        def _require_prefix_attention(self):
            return self._PREFIX_ATTENTION

        def _build_vlm_inputs(self, batch, *, action_token_strings, instructions):
            del batch, action_token_strings, instructions
            return {
                "input_ids": input_ids.clone(),
                "attention_mask": torch.ones_like(input_ids),
            }

        def _autocast_context(self):
            from contextlib import nullcontext

            return nullcontext()

    policy = DummyBlockPolicy()
    loss, info = run_action_token_ce_prefix_training_step(
        policy,
        {"action": torch.zeros(len(raw_token_ids), 1, 1)},
        prefix_adapter=_DummyPrefixAdapter(logits.clone()),
        ignore_index=-100,
    )
    assert info == {}

    with torch.no_grad():
        token_horizon = int(sum(block_schedule))
        clamped_lengths = (
            [max(0, min(int(v), token_horizon)) for v in token_lengths]
            if token_lengths is not None
            else None
        )
        padded_token_ids = []
        for ids in raw_token_ids:
            row = [int(v) for v in ids[:token_horizon]]
            if len(row) < token_horizon:
                row.extend([0] * (token_horizon - len(row)))
            padded_token_ids.append(row)
        target_vlm_token_ids = torch.tensor(padded_token_ids, dtype=torch.long) + 10
        if supervise_eos and token_lengths is not None and eos_token_id is not None:
            target_vlm_token_ids = build_eos_supervision_targets(
                target_vlm_token_ids=target_vlm_token_ids,
                lengths=clamped_lengths,
                eos_token_id=eos_token_id,
                ignore_index=-100,
            )
        labels = torch.full_like(input_ids, -100)
        action_vocab = policy._require_action_token_vocabulary()
        prediction_mask = (
            (input_ids >= action_vocab.token_min)
            & (input_ids <= action_vocab.token_max)
        ) | input_ids.eq(1000)
        labels[prediction_mask] = target_vlm_token_ids.reshape(-1)
    return float(loss.item()), labels


def test_run_action_token_ce_prefix_training_step_clamps_overflow_lengths_to_horizon():
    vocab = 32
    eos_token_id = 19
    input_ids = torch.tensor([[42, 1000, 11, 12, 13, 14, 1]], dtype=torch.long)
    logits = torch.zeros((1, input_ids.shape[1], vocab), dtype=torch.float32)

    overflow_loss, overflow_labels = _run_blockwise_training_loss_for_test(
        block_schedule=(1, 1, 1, 1, 1),
        raw_token_ids=[[1, 2, 3, 4, 5, 6, 7]],
        input_ids=input_ids,
        logits=logits,
        weighting_mode="block_mean",
        supervise_eos=True,
        token_lengths=[7],
        eos_token_id=eos_token_id,
    )
    capped_loss, capped_labels = _run_blockwise_training_loss_for_test(
        block_schedule=(1, 1, 1, 1, 1),
        raw_token_ids=[[1, 2, 3, 4, 5]],
        input_ids=input_ids,
        logits=logits,
        weighting_mode="block_mean",
        supervise_eos=True,
        token_lengths=[5],
        eos_token_id=eos_token_id,
    )

    assert overflow_labels.equal(capped_labels)
    assert overflow_loss == capped_loss


def test_run_action_token_ce_prefix_training_step_block_mean_reweights_nonuniform_schedule():
    vocab = 32
    input_ids = torch.tensor([[42, 43, 1000, 11, 12, 1]], dtype=torch.long)
    logits = torch.zeros((1, input_ids.shape[1], vocab), dtype=torch.float32)
    logits[0, 2, 11] = 2.5
    logits[0, 3, 12] = 0.5
    logits[0, 4, 13] = -0.75

    token_mean_loss, labels = _run_blockwise_training_loss_for_test(
        block_schedule=(1, 2),
        raw_token_ids=[[1, 2, 3]],
        input_ids=input_ids,
        logits=logits,
        weighting_mode="token_mean",
    )
    block_mean_loss, _ = _run_blockwise_training_loss_for_test(
        block_schedule=(1, 2),
        raw_token_ids=[[1, 2, 3]],
        input_ids=input_ids,
        logits=logits,
        weighting_mode="block_mean",
    )

    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(labels)
    supervised = labels.ne(-100)
    expected_token_mean = float(ce[supervised].mean().item())
    expected_block_mean = float(
        (ce[0, supervised[0]] * torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32))
        .sum()
        .div(torch.tensor(2.0))
        .item()
    )

    assert abs(token_mean_loss - expected_token_mean) < 1e-6
    assert abs(block_mean_loss - expected_block_mean) < 1e-6
    assert abs(block_mean_loss - token_mean_loss) > 1e-4


def test_run_action_token_ce_prefix_training_step_block_mean_equals_token_mean_for_all_ones():
    vocab = 24
    input_ids = torch.tensor([[50, 1000, 11, 12, 1]], dtype=torch.long)
    logits = torch.zeros((1, input_ids.shape[1], vocab), dtype=torch.float32)
    logits[0, 1, 11] = 1.0
    logits[0, 2, 12] = 0.5
    logits[0, 3, 13] = -0.25

    token_mean_loss, _ = _run_blockwise_training_loss_for_test(
        block_schedule=(1, 1, 1),
        raw_token_ids=[[1, 2, 3]],
        input_ids=input_ids,
        logits=logits,
        weighting_mode="token_mean",
    )
    block_mean_loss, _ = _run_blockwise_training_loss_for_test(
        block_schedule=(1, 1, 1),
        raw_token_ids=[[1, 2, 3]],
        input_ids=input_ids,
        logits=logits,
        weighting_mode="block_mean",
    )
    assert abs(block_mean_loss - token_mean_loss) < 1e-6


def test_run_action_token_ce_prefix_training_step_rejects_unreachable_eos_length():
    vocab = 32
    eos_token_id = 19
    input_ids = torch.tensor([[42, 1000, 11, 12, 13, 1]], dtype=torch.long)
    logits = torch.zeros((1, input_ids.shape[1], vocab), dtype=torch.float32)

    with pytest.raises(ValueError, match="cumulative block boundaries"):
        _run_blockwise_training_loss_for_test(
            block_schedule=(1, 1, 2),
            raw_token_ids=[[1, 2, 3, 4]],
            input_ids=input_ids,
            logits=logits,
            weighting_mode="block_mean",
            supervise_eos=True,
            token_lengths=[3],
            eos_token_id=eos_token_id,
        )


def test_filter_frozen_vlm_state_dict_keeps_only_trainable_vlm_rows():
    state = {
        "vlm.trainable": torch.tensor([1]),
        "vlm.frozen": torch.tensor([2]),
        "other.weight": torch.tensor([3]),
    }
    out = filter_frozen_vlm_state_dict(
        state,
        trainable_param_names={"vlm.trainable"},
        is_vlm_state_key=lambda key: key.startswith("vlm."),
    )
    assert set(out.keys()) == {"vlm.trainable", "other.weight"}


def test_validate_filtered_load_state_allows_missing_frozen_vlm_keys():
    incompatible = SimpleNamespace(
        missing_keys=["vlm.frozen", "tokenizer.foo"],
        unexpected_keys=["tokenizer.bar"],
    )
    validate_filtered_load_state(
        incompatible,
        strict=True,
        trainable_param_names={"vlm.trainable"},
        is_vlm_state_key=lambda key: key.startswith("vlm."),
        is_tokenizer_state_key=lambda key: key.startswith("tokenizer."),
        ignore_tokenizer_state_keys=True,
        owner_name="Dummy",
    )


def test_validate_filtered_load_state_requires_tokenizer_keys_when_not_frozen():
    incompatible = SimpleNamespace(
        missing_keys=["tokenizer.foo"],
        unexpected_keys=[],
    )

    with pytest.raises(RuntimeError, match="tokenizer.foo"):
        validate_filtered_load_state(
            incompatible,
            strict=True,
            trainable_param_names={"vlm.trainable"},
            is_vlm_state_key=lambda key: key.startswith("vlm."),
            is_tokenizer_state_key=lambda key: key.startswith("tokenizer."),
            ignore_tokenizer_state_keys=False,
            owner_name="Dummy",
        )


def test_build_lora_optim_param_groups_splits_vocab_rows_from_decay():
    class DummyModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.foo = torch.nn.Parameter(torch.ones(2))
            self.embed_tokens = torch.nn.Parameter(torch.ones(2, 2))

    module = DummyModule()
    groups = build_lora_optim_param_groups(
        named_parameters=module.named_parameters(),
        use_lora=True,
        is_vocab_matrix_param_name=lambda name: "embed_tokens" in name,
    )
    assert len(groups) == 2
    assert "weight_decay" not in groups[0]
    assert groups[1]["weight_decay"] == 0.0
