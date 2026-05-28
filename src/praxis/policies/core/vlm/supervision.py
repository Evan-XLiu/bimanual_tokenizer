"""Action-token CE supervision helpers for VLM policies."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from praxis.policies.core.vlm.loss import (
    collect_block_loss_chunks,
    reduce_block_loss_chunks,
)
from praxis.policies.core.vlm.vocab import ActionTokenVocabulary

IGNORE_INDEX = -100


@dataclass(frozen=True)
class ActionTokenSupervision:
    """Prepared CE labels and prefix mask for VLM action-token objectives."""

    labels: Tensor
    prefix_mask: Tensor
    prediction_position_mask: Tensor
    target_vlm_token_ids: Tensor


@dataclass(frozen=True)
class ActionTokenTargetRows:
    """Raw action-token target rows padded to a scheduled horizon."""

    target_raw_token_ids: Tensor
    token_lengths: list[int] | None


def prepare_action_token_target_rows(
    *,
    token_rows: list[list[int]],
    token_lengths: list[int] | None,
    token_horizon: int,
    block_schedule: tuple[int, ...],
    device: torch.device,
    supervise_eos: bool,
    pad_value: int = 0,
) -> ActionTokenTargetRows:
    """Pad/truncate raw action-token rows and reserve EOS space when requested."""
    horizon = int(token_horizon)
    if horizon < 1:
        raise ValueError(f"token_horizon must be positive, got {token_horizon}.")
    validate_block_schedule_sum(block_schedule=block_schedule, token_horizon=horizon)

    if token_lengths is None:
        target_raw_token_ids = tensorize_token_rows(token_rows, device=device)
        if int(target_raw_token_ids.shape[1]) != horizon:
            raise ValueError(
                "Fixed-length token rows must match the scheduled horizon: "
                f"rows={target_raw_token_ids.shape[1]}, horizon={horizon}."
            )
        return ActionTokenTargetRows(
            target_raw_token_ids=target_raw_token_ids,
            token_lengths=None,
        )

    if len(token_lengths) != len(token_rows):
        raise ValueError(
            "Per-sample token lengths must match token batch size: "
            f"len(token_lengths)={len(token_lengths)} vs batch={len(token_rows)}."
        )

    content_horizon = max(0, horizon - 1) if bool(supervise_eos) else horizon
    clamped_lengths = [
        max(0, min(int(length), content_horizon)) for length in token_lengths
    ]
    if supervise_eos:
        validate_eos_supervision_lengths(
            lengths=clamped_lengths,
            block_schedule=block_schedule,
        )

    out = torch.full(
        (len(token_rows), horizon),
        fill_value=int(pad_value),
        dtype=torch.long,
        device=device,
    )
    for row_idx, row in enumerate(token_rows):
        values = [int(value) for value in row[:content_horizon]]
        if values:
            out[row_idx, : len(values)] = torch.as_tensor(
                values,
                dtype=torch.long,
                device=device,
            )
    return ActionTokenTargetRows(
        target_raw_token_ids=out,
        token_lengths=clamped_lengths,
    )


def build_blockwise_action_supervision(
    *,
    input_ids: Tensor,
    attention_mask: Tensor | None,
    target_raw_token_ids: Tensor,
    vocab: ActionTokenVocabulary,
    pad_token_id: int | None,
    token_lengths: list[int] | None = None,
    supervise_eos: bool = False,
    eos_token_id: int | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> ActionTokenSupervision:
    """Build CE labels for BAR-style blockwise action-token prediction inputs."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank-2, got {tuple(input_ids.shape)}.")
    if target_raw_token_ids.ndim != 2:
        raise ValueError(
            "target_raw_token_ids must be rank-2 with shape (B, H), "
            f"got {tuple(target_raw_token_ids.shape)}."
        )
    if input_ids.shape[0] != target_raw_token_ids.shape[0]:
        raise ValueError(
            "input_ids and target_raw_token_ids batch sizes must match, got "
            f"{input_ids.shape[0]} and {target_raw_token_ids.shape[0]}."
        )

    if attention_mask is None:
        valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        valid_mask = attention_mask.to(device=input_ids.device).bool()

    prediction_position_mask = (
        ((input_ids >= int(vocab.token_min)) & (input_ids <= int(vocab.token_max)))
        | input_ids.eq(int(vocab.mask_token_id))
    ) & valid_mask

    token_horizon = int(target_raw_token_ids.shape[1])
    prediction_counts = prediction_position_mask.sum(dim=1)
    expected_counts = torch.full(
        (input_ids.shape[0],),
        token_horizon,
        dtype=torch.long,
        device=input_ids.device,
    )
    if bool((prediction_counts != expected_counts).any()):
        raise RuntimeError(
            "VLM CE inputs must expose exact prediction-token positions; "
            f"got counts={prediction_counts.tolist()} expected={expected_counts.tolist()}."
        )

    target_vlm_token_ids = vocab.raw_ids_to_vlm_ids(target_raw_token_ids).to(
        device=input_ids.device
    )
    if token_lengths is not None:
        if supervise_eos:
            if eos_token_id is None:
                raise RuntimeError("EOS supervision requires a valid tokenizer EOS id.")
            target_vlm_token_ids = build_eos_supervision_targets(
                target_vlm_token_ids=target_vlm_token_ids,
                lengths=token_lengths,
                eos_token_id=int(eos_token_id),
                ignore_index=int(ignore_index),
            )
        else:
            keep_mask = torch.zeros_like(target_vlm_token_ids, dtype=torch.bool)
            for row, length in enumerate(token_lengths):
                if int(length) > 0:
                    keep_mask[row, : int(length)] = True
            target_vlm_token_ids = target_vlm_token_ids.clone()
            target_vlm_token_ids[~keep_mask] = int(ignore_index)
    labels = build_blockwise_labels_from_prediction_positions(
        input_ids=input_ids,
        target_vlm_token_ids=target_vlm_token_ids,
        vocab=vocab,
        ignore_index=ignore_index,
    )
    if labels is None:
        raise RuntimeError(
            "VLM CE inputs must expose exact prediction-token positions; "
            f"got counts={prediction_counts.tolist()} expected={expected_counts.tolist()}."
        )
    if pad_token_id is not None:
        labels[input_ids == int(pad_token_id)] = int(ignore_index)

    prefix_mask = build_prefix_mask_from_prediction_positions(
        input_ids=input_ids,
        valid_mask=valid_mask,
        prediction_position_mask=prediction_position_mask,
    )
    return ActionTokenSupervision(
        labels=labels,
        prefix_mask=prefix_mask,
        prediction_position_mask=prediction_position_mask,
        target_vlm_token_ids=target_vlm_token_ids,
    )


def validate_eos_supervision_lengths(
    *, lengths: list[int], block_schedule: tuple[int, ...]
) -> None:
    """Fail fast unless EOS-supervised lengths land on prediction boundaries."""
    valid_lengths = {0}
    valid_lengths.update(
        int(pos) for pos in itertools.accumulate(int(size) for size in block_schedule)
    )
    invalid_lengths = sorted({int(v) for v in lengths if int(v) not in valid_lengths})
    if invalid_lengths:
        raise ValueError(
            "EOS-supervised token lengths must land on cumulative block "
            f"boundaries {sorted(valid_lengths)}, got invalid lengths "
            f"{invalid_lengths}."
        )


def validate_block_schedule_sum(
    *, block_schedule: tuple[int, ...], token_horizon: int
) -> None:
    """Validate that a scheduled token target shape is internally consistent."""
    if sum(int(size) for size in block_schedule) != int(token_horizon):
        raise ValueError(
            f"block_schedule sum {sum(block_schedule)} must match token_horizon={token_horizon}."
        )


def build_eos_supervision_targets(
    *,
    target_vlm_token_ids: Tensor,
    lengths: list[int],
    eos_token_id: int,
    ignore_index: int,
) -> Tensor:
    """Build per-position targets with EOS-at-length semantics."""
    if target_vlm_token_ids.ndim != 2:
        raise ValueError(
            "target_vlm_token_ids must be rank-2 with shape (B, H), "
            f"got {tuple(target_vlm_token_ids.shape)}."
        )
    batch_size, horizon = target_vlm_token_ids.shape
    if len(lengths) != int(batch_size):
        raise ValueError(
            "lengths must match batch size: "
            f"len(lengths)={len(lengths)}, batch_size={batch_size}."
        )

    labels = target_vlm_token_ids.clone()
    for row, raw_len in enumerate(lengths):
        k = max(0, min(int(raw_len), int(horizon)))
        if k < int(horizon):
            labels[row, k] = int(eos_token_id)
            if k + 1 < int(horizon):
                labels[row, k + 1 :] = int(ignore_index)
    return labels


def build_blockwise_labels_from_prediction_positions(
    *,
    input_ids: Tensor,
    target_vlm_token_ids: Tensor,
    vocab: ActionTokenVocabulary,
    ignore_index: int,
) -> Tensor | None:
    """Build labels on BAR/KI prediction-token positions."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank-2, got {tuple(input_ids.shape)}.")
    if target_vlm_token_ids.ndim != 2:
        raise ValueError(
            f"target_vlm_token_ids must be rank-2, got {tuple(target_vlm_token_ids.shape)}."
        )
    if input_ids.shape[0] != target_vlm_token_ids.shape[0]:
        raise ValueError(
            "input_ids and target_vlm_token_ids batch sizes must match, "
            f"got {input_ids.shape[0]} and {target_vlm_token_ids.shape[0]}."
        )

    labels = torch.full_like(input_ids, int(ignore_index))
    token_horizon = int(target_vlm_token_ids.shape[1])
    prediction_token_mask = (
        (input_ids >= int(vocab.token_min)) & (input_ids <= int(vocab.token_max))
    ) | input_ids.eq(int(vocab.mask_token_id))
    prediction_counts = prediction_token_mask.sum(dim=1)
    if bool((prediction_counts != token_horizon).any()):
        return None

    labels[prediction_token_mask] = target_vlm_token_ids.reshape(-1)
    return labels


def compute_blockwise_token_loss(
    *,
    logits: Tensor,
    labels: Tensor,
    block_schedule: tuple[int, ...],
    ignore_index: int,
    weighting_mode: str,
    allow_partial_blocks: bool = False,
) -> Tensor | None:
    """Reduce per-token CE over BAR/KI blockwise prediction positions."""
    ce_per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=int(ignore_index),
        reduction="none",
    ).reshape_as(labels)

    return reduce_blockwise_token_ce_loss(
        ce_per_token=ce_per_token,
        supervised_mask=labels.ne(int(ignore_index)),
        block_schedule=block_schedule,
        weighting_mode=weighting_mode,
        allow_partial_blocks=allow_partial_blocks,
    )


def reduce_blockwise_token_ce_loss(
    *,
    ce_per_token: Tensor,
    supervised_mask: Tensor,
    block_schedule: tuple[int, ...],
    weighting_mode: str,
    allow_partial_blocks: bool = False,
) -> Tensor | None:
    """Reduce precomputed per-token CE over BAR/KI blockwise prediction positions."""
    block_chunks = collect_block_loss_chunks(
        ce_per_token=ce_per_token,
        supervised_mask=supervised_mask.to(
            device=ce_per_token.device,
            dtype=torch.bool,
        ),
        block_schedule=block_schedule,
        allow_partial_blocks=allow_partial_blocks,
    )
    return reduce_block_loss_chunks(
        block_chunks,
        weighting_mode=weighting_mode,
    )


def build_prefix_mask_from_prediction_positions(
    *,
    input_ids: Tensor,
    valid_mask: Tensor,
    prediction_position_mask: Tensor,
) -> Tensor:
    """Return VLM prefix positions before the first action prediction token."""
    seq_len = int(input_ids.shape[1])
    positions = torch.arange(seq_len, device=input_ids.device)
    first_prediction = (
        torch.where(
            prediction_position_mask,
            positions.unsqueeze(0),
            torch.full_like(input_ids, seq_len),
        )
        .min(dim=1)
        .values
    )
    if bool((first_prediction == seq_len).any()):
        raise RuntimeError("Every CE row must contain at least one prediction token.")
    return valid_mask & (positions.unsqueeze(0) < first_prediction.unsqueeze(1))


def tensorize_token_rows(rows: list[list[int]], *, device: torch.device) -> Tensor:
    """Convert fixed-length token rows to a rank-2 long tensor."""
    if not rows:
        return torch.empty((0, 0), dtype=torch.long, device=device)
    lengths = {len(row) for row in rows}
    if len(lengths) != 1:
        raise ValueError(
            f"Expected fixed-length token rows, got lengths={sorted(lengths)}."
        )
    return torch.as_tensor(rows, dtype=torch.long, device=device)
