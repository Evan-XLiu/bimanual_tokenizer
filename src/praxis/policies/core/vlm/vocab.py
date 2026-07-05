# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Action-token vocabulary helpers for VLM policies."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class ActionTokenVocabulary:
    """VLM vocabulary rows reserved for action tokens and their input mask."""

    tokens: tuple[str, ...]
    token_ids: tuple[int, ...]
    mask_token: str
    mask_token_id: int

    def __post_init__(self) -> None:
        if not self.tokens:
            raise ValueError("Action token vocabulary must contain at least one token.")
        if len(self.tokens) != len(self.token_ids):
            raise ValueError(
                "Action token strings and ids must have the same length, got "
                f"{len(self.tokens)} and {len(self.token_ids)}."
            )
        if not self.mask_token:
            raise ValueError("Action token vocabulary requires an action mask token.")
        validate_contiguous_token_ids([int(token_id) for token_id in self.token_ids])
        mask_token_id = int(self.mask_token_id)
        if self.token_min <= mask_token_id <= self.token_max:
            raise ValueError(
                "The action mask token must be outside the action-token decode "
                f"range [{self.token_min}, {self.token_max}], got {mask_token_id}."
            )

    @property
    def token_min(self) -> int:
        return min(int(token_id) for token_id in self.token_ids)

    @property
    def token_max(self) -> int:
        return max(int(token_id) for token_id in self.token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def all_tokens(self) -> tuple[str, ...]:
        """Return action-token strings plus the input-only action mask token."""
        return (*self.tokens, self.mask_token)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        """Return action-token ids plus the input-only action mask token id."""
        return (*self.token_ids, int(self.mask_token_id))

    def render_raw_ids(self, token_ids: list[int]) -> str:
        return "".join(self.tokens[int(token_id)] for token_id in token_ids)

    def raw_ids_to_vlm_ids(self, token_ids: Tensor) -> Tensor:
        return token_ids.long() + int(self.token_min)

    def vlm_ids_to_raw_ids(self, token_ids: Tensor) -> Tensor:
        return token_ids.long() - int(self.token_min)


def resolve_num_action_tokens(
    *,
    codebook_size: int,
    configured: int | None,
    owner_name: str,
) -> int:
    """Resolve action-token vocabulary size from config or tokenizer metadata."""
    inferred = int(codebook_size)
    if configured is not None and int(configured) != inferred:
        raise ValueError(
            f"{owner_name} action-token vocabulary size mismatch: "
            f"config.num_action_tokens={configured} but action_tokenizer.codebook_size={inferred}. "
            "Set num_action_tokens to null/None to infer automatically, or make them match."
        )
    if inferred < 1:
        raise ValueError(
            f"{owner_name} action tokenizer codebook_size must be positive."
        )
    return inferred


def setup_action_token_vocabulary(
    *,
    tokenizer,
    vlm,
    num_action_tokens: int,
    token_template: str,
    mask_token: str,
    init_mode: str,
) -> ActionTokenVocabulary:
    """Resolve or add the VLM rows used for action tokens and action masking."""
    tokens = tuple(
        str(token_template).format(i=i) for i in range(int(num_action_tokens))
    )
    all_token_ids = resolve_or_add_special_tokens(
        tokenizer=tokenizer,
        vlm=vlm,
        tokens=[*tokens, mask_token],
        init_mode=init_mode,
    )
    token_ids = tuple(int(token_id) for token_id in all_token_ids[:-1])
    return ActionTokenVocabulary(
        tokens=tokens,
        token_ids=token_ids,
        mask_token=mask_token,
        mask_token_id=int(all_token_ids[-1]),
    )


def build_trainable_vocab_rows(
    *,
    vocab: ActionTokenVocabulary,
    vocab_size: int,
    owner_name: str,
    supervise_eos: bool = False,
    eos_token_id: int | None = None,
) -> torch.LongTensor:
    """Return action-token, action-mask, and optional EOS rows allowed to train."""
    rows = torch.tensor(vocab.all_token_ids, dtype=torch.long)
    expected = len(vocab.all_token_ids)
    if supervise_eos and eos_token_id is not None:
        rows = torch.cat([rows, torch.tensor([int(eos_token_id)], dtype=torch.long)])
        expected += 1

    in_vocab = (rows >= 0) & (rows < int(vocab_size))
    if not in_vocab.all():
        bad = rows[~in_vocab].tolist()
        raise ValueError(
            f"{owner_name} trainable vocab rows out of range for vocab_size={vocab_size}: {bad}"
        )

    rows = torch.unique(rows, sorted=True)
    if int(rows.numel()) != expected:
        raise ValueError(
            f"{owner_name} expected {expected} trainable vocab rows, got {int(rows.numel())}."
        )
    return rows


def resolve_or_add_special_tokens(
    *,
    tokenizer,
    vlm,
    tokens: list[str],
    init_mode: str,
) -> list[int]:
    """Resolve existing special-token ids or add them contiguously."""
    if not tokens:
        raise ValueError("tokens must be non-empty.")

    vocab = tokenizer.get_vocab()
    first_token = tokens[0]
    if first_token in vocab:
        missing = [token for token in tokens if token not in vocab]
        if missing:
            raise RuntimeError(
                f"Special tokens were partially present in the tokenizer vocabulary; missing={missing}."
            )
        token_ids = [int(vocab[token]) for token in tokens]
        validate_contiguous_token_ids(token_ids)
        return token_ids

    old_size = int(vlm.get_input_embeddings().weight.shape[0])
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    vocab = tokenizer.get_vocab()
    token_ids = [int(vocab[token]) for token in tokens]
    validate_contiguous_token_ids(token_ids)

    required_size = max(old_size, max(token_ids) + 1)
    vlm.resize_token_embeddings(required_size, mean_resizing=False)
    initialize_embedding_rows(
        embedding=vlm.get_input_embeddings(),
        row_ids=token_ids,
        old_size=old_size,
        init_mode=init_mode,
    )
    return token_ids


def validate_contiguous_token_ids(token_ids: list[int]) -> None:
    """Require token ids to be consecutive in token/raw-id order."""
    if not token_ids:
        raise ValueError("token_ids must be non-empty.")
    token_min = min(int(token_id) for token_id in token_ids)
    expected_ids = list(range(token_min, token_min + len(token_ids)))
    if [int(token_id) for token_id in token_ids] != expected_ids:
        raise RuntimeError(
            "Special token ids must be contiguous in token/raw-id order after "
            f"tokenizer extension; got {tuple(int(token_id) for token_id in token_ids)}."
        )


def initialize_embedding_rows(
    *,
    embedding,
    row_ids: list[int],
    old_size: int,
    init_mode: str,
) -> None:
    """Initialize newly-added embedding rows using the requested strategy."""
    rows = torch.as_tensor(row_ids, dtype=torch.long, device=embedding.weight.device)
    if rows.numel() == 0:
        return
    if init_mode == "normal":
        values = torch.empty(
            (len(row_ids), int(embedding.weight.shape[1])),
            dtype=embedding.weight.dtype,
            device=embedding.weight.device,
        )
        nn.init.normal_(values, mean=0.0, std=0.02)
    elif init_mode == "avg":
        values = (
            embedding.weight.data[:old_size]
            .mean(dim=0)
            .expand(len(row_ids), -1)
            .clone()
        )
    elif init_mode == "zero":
        values = torch.zeros(
            (len(row_ids), int(embedding.weight.shape[1])),
            dtype=embedding.weight.dtype,
            device=embedding.weight.device,
        )
    else:
        raise ValueError(
            f"Unsupported action_embed_init={init_mode!r}. Expected one of {{'normal', 'avg', 'zero'}}."
        )
    embedding.weight.data.index_copy_(0, rows, values)


def install_vocab_row_gradient_masks(
    *,
    named_parameters: Iterable[tuple[str, Tensor]],
    use_lora: bool,
    is_vocab_matrix_param_name: Callable[[str], bool],
    build_trainable_rows: Callable[[int], torch.LongTensor],
    hook_handles: list[RemovableHandle],
) -> None:
    """Restrict LoRA-updated vocab-matrix grads to the selected trainable rows."""
    for handle in hook_handles:
        handle.remove()
    hook_handles.clear()

    if not use_lora:
        return

    for name, param in named_parameters:
        if not param.requires_grad or param.ndim != 2:
            continue
        if not is_vocab_matrix_param_name(name):
            continue

        allowed_rows = build_trainable_rows(int(param.shape[0]))
        if allowed_rows.numel() == 0:
            continue

        def _mask_rows(grad: Tensor, *, _rows: Tensor = allowed_rows) -> Tensor:
            row_ids = _rows.to(device=grad.device)
            row_ids = row_ids[(row_ids >= 0) & (row_ids < grad.shape[0])]
            if row_ids.numel() == 0:
                return torch.zeros_like(grad)
            out = torch.zeros_like(grad)
            out.index_copy_(0, row_ids, grad.index_select(0, row_ids))
            return out

        hook_handles.append(param.register_hook(_mask_rows))
