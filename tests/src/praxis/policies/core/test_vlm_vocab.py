# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for action-token vocab helpers."""

from __future__ import annotations

import torch

from praxis.policies.core.vlm.vocab import (
    ActionTokenVocabulary,
    build_trainable_vocab_rows,
    initialize_embedding_rows,
    resolve_num_action_tokens,
    resolve_or_add_special_tokens,
    validate_contiguous_token_ids,
)


class DummyHFTokenizer:
    def __init__(self) -> None:
        self._vocab = {"base_0": 0, "base_1": 1}

    def get_vocab(self):
        return dict(self._vocab)

    def add_special_tokens(self, payload):
        tokens = payload["additional_special_tokens"]
        next_id = max(self._vocab.values(), default=-1) + 1
        for token in tokens:
            self._vocab[token] = next_id
            next_id += 1


class DummyVLM:
    def __init__(self, vocab_size: int = 2, width: int = 3) -> None:
        self._embedding = torch.nn.Embedding(vocab_size, width)

    def get_input_embeddings(self):
        return self._embedding

    def resize_token_embeddings(self, size, mean_resizing=False):
        del mean_resizing
        old = self._embedding
        new = torch.nn.Embedding(size, old.embedding_dim)
        with torch.no_grad():
            new.weight[: old.num_embeddings] = old.weight
        self._embedding = new


def test_resolve_num_action_tokens_accepts_inferred_codebook_size():
    assert (
        resolve_num_action_tokens(codebook_size=8, configured=None, owner_name="Dummy")
        == 8
    )


def test_build_trainable_vocab_rows_includes_action_mask_row():
    vocab = ActionTokenVocabulary(
        tokens=("<a0>", "<a1>", "<a2>"),
        token_ids=(10, 11, 12),
        mask_token="<mask>",
        mask_token_id=99,
    )
    rows = build_trainable_vocab_rows(
        vocab=vocab,
        vocab_size=128,
        owner_name="Dummy",
    )
    assert rows.tolist() == [10, 11, 12, 99]


def test_validate_contiguous_token_ids_rejects_gaps():
    try:
        validate_contiguous_token_ids([2, 4])
    except RuntimeError as exc:
        assert "contiguous" in str(exc)
    else:
        raise AssertionError("validate_contiguous_token_ids should reject gapped ids")


def test_action_token_vocabulary_rejects_reordered_contiguous_ids():
    try:
        ActionTokenVocabulary(
            tokens=("<a0>", "<a1>", "<a2>"),
            token_ids=(10, 12, 11),
            mask_token="<mask>",
            mask_token_id=13,
        )
    except RuntimeError as exc:
        assert "raw-id order" in str(exc)
    else:
        raise AssertionError("ActionTokenVocabulary should reject reordered ids")


def test_action_token_vocabulary_requires_and_tracks_mask_token():
    vocab = ActionTokenVocabulary(
        tokens=("<a0>", "<a1>"),
        token_ids=(10, 11),
        mask_token="<mask>",
        mask_token_id=12,
    )
    assert vocab.mask_token_id == 12
    assert vocab.all_tokens == ("<a0>", "<a1>", "<mask>")
    assert vocab.all_token_ids == (10, 11, 12)


def test_action_token_vocabulary_rejects_missing_mask_token():
    try:
        ActionTokenVocabulary(
            tokens=("<a0>", "<a1>"),
            token_ids=(10, 11),
            mask_token="",
            mask_token_id=12,
        )
    except ValueError as exc:
        assert "requires an action mask token" in str(exc)
    else:
        raise AssertionError("ActionTokenVocabulary should require a mask token")


def test_action_token_vocabulary_rejects_mask_inside_action_range():
    try:
        ActionTokenVocabulary(
            tokens=("<a0>", "<a1>"),
            token_ids=(10, 11),
            mask_token="<mask>",
            mask_token_id=11,
        )
    except ValueError as exc:
        assert "outside the action-token decode range" in str(exc)
    else:
        raise AssertionError("ActionTokenVocabulary should reject in-range mask ids")


def test_initialize_embedding_rows_zeroes_requested_rows():
    embedding = torch.nn.Embedding(5, 3)
    with torch.no_grad():
        embedding.weight.fill_(1.0)
    initialize_embedding_rows(
        embedding=embedding,
        row_ids=[3, 4],
        old_size=3,
        init_mode="zero",
    )
    assert embedding.weight[3:].equal(torch.zeros_like(embedding.weight[3:]))


def test_resolve_or_add_special_tokens_adds_and_initializes_rows():
    tokenizer = DummyHFTokenizer()
    vlm = DummyVLM()
    token_ids = resolve_or_add_special_tokens(
        tokenizer=tokenizer,
        vlm=vlm,
        tokens=["<|a|>", "<|b|>"],
        init_mode="zero",
    )
    assert token_ids == [2, 3]
    assert vlm.get_input_embeddings().weight.shape[0] == 4
    assert torch.equal(
        vlm.get_input_embeddings().weight[2:4],
        torch.zeros_like(vlm.get_input_embeddings().weight[2:4]),
    )


def test_setup_action_token_vocabulary_installs_mask_with_action_tokens():
    tokenizer = DummyHFTokenizer()
    vlm = DummyVLM()
    from praxis.policies.core.vlm.vocab import setup_action_token_vocabulary

    vocab = setup_action_token_vocabulary(
        tokenizer=tokenizer,
        vlm=vlm,
        num_action_tokens=2,
        token_template="<|action_{i}|>",
        mask_token="<|action_mask|>",
        init_mode="zero",
    )

    assert vocab.tokens == ("<|action_0|>", "<|action_1|>")
    assert vocab.all_tokens == (
        "<|action_0|>",
        "<|action_1|>",
        "<|action_mask|>",
    )
    assert vocab.all_token_ids == (2, 3, 4)
    assert tokenizer.get_vocab()["<|action_mask|>"] == 4
