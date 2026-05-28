"""Focused tests for shared VLM state-prefix conditioning."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from praxis.core.types import FeatureSpec, FeatureType, NormMode
from praxis.policies.core.state_prefix import (
    DEFAULT_STATE_PREFIX_TOKEN,
    STATE_PREFIX_EMBEDDINGS_KEY,
    STATE_PREFIX_PLACEHOLDER_MASK_KEY,
    prepare_state_prefix,
    replace_state_prefix_placeholder_embeddings,
    setup_state_prefix_token,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "State": 1}

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                added += 1
        return added

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self.vocab[text]]


class _Vlm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(2, 4)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def resize_token_embeddings(
        self,
        size: int,
        *,
        mean_resizing: bool = False,
    ) -> nn.Embedding:
        del mean_resizing
        old_weight = self.embed.weight.detach().clone()
        self.embed = nn.Embedding(int(size), int(old_weight.shape[1]))
        with torch.no_grad():
            self.embed.weight[: int(old_weight.shape[0])] = old_weight
        return self.embed


def _state_feature() -> FeatureSpec:
    return FeatureSpec(
        key="observation.state",
        type=FeatureType.STATE,
        shape=(3,),
        norm_mode=NormMode.MIN_MAX,
    )


def _scalar_state_feature() -> FeatureSpec:
    return FeatureSpec(
        key="observation.scalar",
        type=FeatureType.STATE,
        shape=(),
        norm_mode=NormMode.MIN_MAX,
    )


def test_discrete_state_prefix_renders_256_bin_text() -> None:
    prepared = prepare_state_prefix(
        {"observation.state": torch.tensor([[-1.0, 0.0, 1.0]])},
        state_features=[_state_feature()],
        encoding="discrete_text",
        num_bins=256,
        placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
    )

    assert prepared.texts == ["State: 0 128 255"]
    assert prepared.state_vector is None


def test_continuous_state_prefix_uses_placeholder_text() -> None:
    prepared = prepare_state_prefix(
        {"observation.state": torch.zeros(2, 3)},
        state_features=[_state_feature()],
        encoding="continuous_projection",
        num_bins=256,
        placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
    )

    assert prepared.texts == [
        f"State: {DEFAULT_STATE_PREFIX_TOKEN}",
        f"State: {DEFAULT_STATE_PREFIX_TOKEN}",
    ]


def test_continuous_state_prefix_carries_source_state() -> None:
    state = torch.tensor([[-0.5, 0.0, 0.5], [0.25, -0.25, 1.0]])

    prepared = prepare_state_prefix(
        {"observation.state": state},
        state_features=[_state_feature()],
        encoding="continuous_projection",
        num_bins=256,
        placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
    )

    assert prepared.texts == [
        f"State: {DEFAULT_STATE_PREFIX_TOKEN}",
        f"State: {DEFAULT_STATE_PREFIX_TOKEN}",
    ]
    torch.testing.assert_close(prepared.state_vector, state)


def test_state_prefix_rejects_batch_size_mismatch() -> None:
    with pytest.raises(ValueError, match="batch size"):
        prepare_state_prefix(
            {"observation.state": torch.zeros(2, 3)},
            state_features=[_state_feature()],
            encoding="discrete_text",
            num_bins=256,
            placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
            batch_size=3,
        )


def test_scalar_state_prefix_uses_last_temporal_step() -> None:
    prepared = prepare_state_prefix(
        {"observation.scalar": torch.tensor([[0.0, 0.5], [-0.5, 1.0]])},
        state_features=[_scalar_state_feature()],
        encoding="continuous_projection",
        num_bins=256,
        placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
        batch_size=2,
    )

    torch.testing.assert_close(
        prepared.state_vector,
        torch.tensor([[0.5], [1.0]]),
    )


def test_state_prefix_requires_declared_state_keys() -> None:
    with pytest.raises(KeyError, match="observation.state"):
        prepare_state_prefix(
            {},
            state_features=[_state_feature()],
            encoding="discrete_text",
            num_bins=256,
            placeholder_token=DEFAULT_STATE_PREFIX_TOKEN,
        )


def test_state_prefix_token_must_be_new_to_vocab() -> None:
    tokenizer = _Tokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|state|>"]})

    with pytest.raises(ValueError, match="already present"):
        setup_state_prefix_token(
            tokenizer=tokenizer,
            vlm=_Vlm(),
            token="<|state|>",
            init_mode="zero",
        )


def test_setup_state_prefix_token_adds_single_reserved_token() -> None:
    tokenizer = _Tokenizer()
    vlm = _Vlm()

    token_id = setup_state_prefix_token(
        tokenizer=tokenizer,
        vlm=vlm,
        token="<|state|>",
        init_mode="zero",
    )

    assert token_id == 2
    assert tokenizer.encode("<|state|>", add_special_tokens=False) == [2]
    assert vlm.get_input_embeddings().weight.shape[0] == 3


def test_replace_state_prefix_placeholder_embeddings_replaces_placeholder() -> None:
    input_ids = torch.tensor([[1, 7, 2], [7, 3, 4]])
    inputs_embeds = torch.zeros(2, 3, 4)
    state_embeds = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    out = replace_state_prefix_placeholder_embeddings(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        vlm_inputs={
            STATE_PREFIX_EMBEDDINGS_KEY: state_embeds,
            STATE_PREFIX_PLACEHOLDER_MASK_KEY: input_ids.eq(7),
        },
    )

    torch.testing.assert_close(out[0, 1], state_embeds[0])
    torch.testing.assert_close(out[1, 0], state_embeds[1])
    assert out.sum().item() == state_embeds.sum().item()


def test_state_prefix_embedding_replacement_requires_one_placeholder_per_sample() -> (
    None
):
    with pytest.raises(ValueError, match="exactly one"):
        replace_state_prefix_placeholder_embeddings(
            input_ids=torch.tensor([[7, 7], [1, 2]]),
            inputs_embeds=torch.zeros(2, 2, 4),
            vlm_inputs={
                STATE_PREFIX_EMBEDDINGS_KEY: torch.zeros(2, 4),
                STATE_PREFIX_PLACEHOLDER_MASK_KEY: torch.tensor(
                    [[True, True], [False, False]]
                ),
            },
        )
