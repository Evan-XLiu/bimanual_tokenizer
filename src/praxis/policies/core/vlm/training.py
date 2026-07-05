# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Shared VLM action-token CE training helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.vlm.blockwise import render_block_token_strings
from praxis.policies.core.vlm.loss import validate_ce_vocab_scope
from praxis.policies.core.vlm.prefix import (
    VlmInputBatch,
    VlmPrefixLogitsAdapter,
    VlmPrefixState,
)
from praxis.policies.core.vlm.schedule import (
    PrefixAttentionMode,
    build_attention_mask,
    build_prediction_block_input_ids,
)
from praxis.policies.core.vlm.supervision import (
    IGNORE_INDEX,
    ActionTokenSupervision,
    build_blockwise_action_supervision,
    compute_blockwise_token_loss,
    prepare_action_token_target_rows,
    reduce_blockwise_token_ce_loss,
    tensorize_token_rows,
    validate_block_schedule_sum,
)
from praxis.policies.core.vlm.vocab import ActionTokenVocabulary

DEFAULT_SPARSE_CE_PROJECTION_CHUNK_SIZE = 512


class ActionTokenCeConfig(Protocol):
    """Config fields required by shared action-token CE training."""

    @property
    def block_ce_weighting(self) -> str: ...

    @property
    def ce_vocab_scope(self) -> str: ...


class ActionTokenCePolicy(Protocol):
    """Policy interface required by the shared VLM action-token CE lane."""

    normalize: Any
    io: Any | None
    vlm: Any
    processor: Any
    training: bool
    _SCHEDULE_STRIP_KEYS: ClassVar[tuple[str, ...]]
    _eos_token_id: int | None

    @property
    def config(self) -> ActionTokenCeConfig: ...

    @property
    def _supervise_eos(self) -> bool: ...

    def tokenize_actions_to_policy_tokens(self, actions: Tensor) -> list[list[int]]: ...

    def _require_action_token_vocabulary(self) -> ActionTokenVocabulary: ...

    def _build_vlm_inputs(
        self,
        obs: dict[str, Tensor],
        *,
        action_token_strings: list[str] | None,
        instructions: list[str] | None,
    ) -> VlmInputBatch: ...

    def _require_prefix_attention(self) -> PrefixAttentionMode: ...

    def _autocast_context(self) -> Any: ...

    def _get_action_token_lengths(
        self,
        token_ids_batch: list[list[int]],
    ) -> list[int] | None: ...

    def _resolve_action_token_horizon(
        self,
        token_ids_batch: list[list[int]],
        *,
        token_lengths: list[int],
    ) -> int: ...

    def _resolve_action_token_block_schedule(
        self, *, token_horizon: int
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class PreparedActionTokenCe:
    """Prepared teacher-forced VLM inputs and labels for action-token CE."""

    batch: dict[str, Tensor]
    block_schedule: tuple[int, ...]
    vlm_inputs: VlmInputBatch
    labels: Tensor
    supervision: ActionTokenSupervision
    token_lengths: list[int] | None


@dataclass(frozen=True)
class ActionTokenCeForward:
    """Result of one VLM action-token CE forward pass."""

    loss: Tensor
    outputs: Any
    prepared: PreparedActionTokenCe


@dataclass(frozen=True)
class ActionTokenCePrefixOutput:
    """Output of a hidden-state VLM CE prefix forward."""

    prefix_state: VlmPrefixState


def zero_action_token_ce_loss(
    policy: ActionTokenCePolicy, *, device: torch.device
) -> Tensor:
    """Return a scalar zero loss that preserves train-time autograd contracts."""
    zero = torch.zeros((), device=device, dtype=torch.float32)
    if policy.training:
        zero = zero.requires_grad_(True)
    return zero


def skipped_all_ignore_result(
    policy: ActionTokenCePolicy, *, device: torch.device
) -> tuple[Tensor, dict[str, float]]:
    """Return the standard loss/info pair for fully ignored CE batches."""
    return zero_action_token_ce_loss(policy, device=device), {
        "skipped_all_ignore_batch": 1.0
    }


def normalize_action_token_ce_batch(
    policy: ActionTokenCePolicy,
    batch: dict[str, Tensor],
) -> tuple[dict[str, Tensor], tuple[Tensor, dict[str, float]] | None]:
    """Normalize a CE batch and short-circuit fully padded action samples."""
    batch = dict(batch)
    if policy.normalize is not None:
        batch = policy.normalize(batch)

    action_key = _action_key(policy)
    pad_key = f"{action_key}_is_pad"
    device = next(policy.vlm.parameters()).device
    if pad_key in batch and batch[pad_key].all():
        return batch, skipped_all_ignore_result(policy, device=device)
    return batch, None


def prepare_action_token_ce(
    policy: ActionTokenCePolicy,
    batch: dict[str, Tensor],
    *,
    actions: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> PreparedActionTokenCe:
    """Prepare shared BAR/KI-style blockwise action-token CE inputs."""
    if actions is None:
        actions = batch[_action_key(policy)]
    token_ids_batch = policy.tokenize_actions_to_policy_tokens(actions)
    token_lengths = policy._get_action_token_lengths(token_ids_batch)
    return prepare_action_token_ce_from_token_targets(
        policy,
        batch,
        token_ids_batch=token_ids_batch,
        token_lengths=token_lengths,
        actions=actions,
        ignore_index=ignore_index,
    )


def prepare_action_token_ce_from_token_targets(
    policy: ActionTokenCePolicy,
    batch: dict[str, Tensor],
    *,
    token_ids_batch: list[list[int]],
    token_lengths: list[int] | None,
    actions: Tensor | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> PreparedActionTokenCe:
    """Prepare action-token CE inputs from explicit raw token targets."""
    device = _action_token_device(policy, actions=actions)
    target_token_ids, token_lengths, block_schedule = _prepare_target_token_ids(
        policy,
        token_ids_batch=token_ids_batch,
        token_lengths=token_lengths,
        device=device,
    )

    action_vocab = policy._require_action_token_vocabulary()
    _validate_raw_action_token_ids(
        target_token_ids,
        num_tokens=action_vocab.num_tokens,
        owner_name=policy.__class__.__name__,
    )
    raw_mask_token_id = int(action_vocab.num_tokens)
    pred_token_ids = build_prediction_block_input_ids(
        target_token_ids,
        block_schedule,
        mask_token_id=raw_mask_token_id,
    )
    action_strings = render_block_token_strings(
        pred_token_ids,
        mask_sentinel=raw_mask_token_id,
        vocab=action_vocab,
    )
    instructions = extract_instructions(
        batch,
        language_key=None
        if policy.io is None
        else getattr(policy.io, "language_key", None),
    )
    vlm_inputs = dict(
        policy._build_vlm_inputs(
            batch,
            action_token_strings=action_strings,
            instructions=instructions,
        )
    )
    for key in policy._SCHEDULE_STRIP_KEYS:
        vlm_inputs.pop(key, None)

    processor_tokenizer = getattr(policy.processor, "tokenizer", None)
    supervision = build_blockwise_action_supervision(
        input_ids=vlm_inputs["input_ids"],
        attention_mask=vlm_inputs.get("attention_mask"),
        target_raw_token_ids=target_token_ids.to(device=vlm_inputs["input_ids"].device),
        vocab=action_vocab,
        pad_token_id=getattr(processor_tokenizer, "pad_token_id", None),
        token_lengths=token_lengths,
        supervise_eos=policy._supervise_eos,
        eos_token_id=policy._eos_token_id,
        ignore_index=ignore_index,
    )
    labels = supervision.labels
    action_is_pad = _action_pad_mask_from_batch(policy, batch, actions=actions)
    if action_is_pad is not None:
        labels = labels.clone()
        # Token CE trains only on samples with a complete supervised horizon.
        padded_samples = action_is_pad.any(dim=1).to(device=labels.device)
        labels[padded_samples] = int(ignore_index)

    return PreparedActionTokenCe(
        batch=batch,
        block_schedule=block_schedule,
        vlm_inputs=vlm_inputs,
        labels=labels,
        supervision=supervision,
        token_lengths=token_lengths,
    )


def run_action_token_ce_prefix_forward(
    policy: ActionTokenCePolicy,
    prepared: PreparedActionTokenCe,
    *,
    prefix_adapter: VlmPrefixLogitsAdapter,
    ignore_index: int = IGNORE_INDEX,
    gradient_checkpointing: bool = False,
    projection_chunk_size: int | None = DEFAULT_SPARSE_CE_PROJECTION_CHUNK_SIZE,
) -> ActionTokenCeForward:
    """Run scheduled VLM CE through the prefix adapter and sparse hidden CE."""
    schedule_attention_mask = build_prepared_action_token_attention_mask(
        policy,
        prepared,
    )
    with policy._autocast_context():
        prefix_state = prefix_adapter.prepare_prefix(
            prepared.vlm_inputs,
            layer_attention_mask=schedule_attention_mask,
        )
        prefix_state = forward_vlm_prefix_state(
            prefix_adapter,
            prefix_state,
            gradient_checkpointing=gradient_checkpointing,
        )
        loss = compute_prepared_action_token_ce_loss_from_hidden(
            policy,
            prepared,
            hidden=prefix_state.hidden,
            project_hidden_to_logits=prefix_adapter.final_hidden_to_logits,
            ignore_index=ignore_index,
            projection_chunk_size=projection_chunk_size,
        )
    return ActionTokenCeForward(
        loss=loss,
        outputs=ActionTokenCePrefixOutput(prefix_state=prefix_state),
        prepared=prepared,
    )


def forward_vlm_prefix_state(
    prefix_adapter: VlmPrefixLogitsAdapter,
    state: VlmPrefixState,
    *,
    gradient_checkpointing: bool,
) -> VlmPrefixState:
    """Advance a prepared VLM prefix stream without building expert K/V tensors."""
    for layer_idx in range(int(prefix_adapter.num_layers)):
        if bool(gradient_checkpointing) and torch.is_grad_enabled():

            def layer_block(
                hidden: Tensor,
                *,
                bound_state: VlmPrefixState = state,
                bound_layer_idx: int = layer_idx,
            ) -> Tensor:
                return prefix_adapter.forward_prefix_layer(
                    replace(bound_state, hidden=hidden),
                    layer_idx=bound_layer_idx,
                ).hidden

            hidden = checkpoint(
                layer_block,
                state.hidden,
                use_reentrant=False,
            )
            state = replace(state, hidden=hidden)
        else:
            state = prefix_adapter.forward_prefix_layer(state, layer_idx=layer_idx)
    return state


def build_prepared_action_token_attention_mask(
    policy: ActionTokenCePolicy,
    prepared: PreparedActionTokenCe,
) -> Tensor:
    """Build the canonical scheduled attention mask for prepared action-token CE."""
    return build_attention_mask(
        prepared.vlm_inputs["attention_mask"],
        prepared.block_schedule,
        dtype=next(policy.vlm.parameters()).dtype,
        prefix_attention=policy._require_prefix_attention(),
        prediction_position_mask=prepared.supervision.prediction_position_mask,
    )


def compute_prepared_action_token_ce_loss(
    policy: ActionTokenCePolicy,
    prepared: PreparedActionTokenCe,
    *,
    logits: Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> Tensor:
    """Reduce prepared action-token CE from logits produced by another VLM path."""
    loss = None
    labels = prepared.labels.to(device=logits.device)
    candidate_token_ids = build_action_token_ce_candidate_token_ids(
        policy,
        device=logits.device,
    )
    if candidate_token_ids is not None:
        logits = logits.index_select(-1, candidate_token_ids)
        labels = remap_labels_to_candidate_token_indices(
            labels,
            candidate_token_ids=candidate_token_ids,
            ignore_index=ignore_index,
            owner_name=policy.__class__.__name__,
        )
    if bool(labels.ne(int(ignore_index)).any()):
        loss = compute_blockwise_token_loss(
            logits=logits.float(),
            labels=labels,
            block_schedule=prepared.block_schedule,
            ignore_index=ignore_index,
            weighting_mode=str(policy.config.block_ce_weighting),
            allow_partial_blocks=prepared.token_lengths is not None,
        )
    if loss is None:
        loss = zero_action_token_ce_loss(
            policy,
            device=prepared.vlm_inputs["input_ids"].device,
        )
    return loss


def compute_prepared_action_token_ce_loss_from_hidden(
    policy: ActionTokenCePolicy,
    prepared: PreparedActionTokenCe,
    *,
    hidden: Tensor,
    project_hidden_to_logits: Callable[..., Tensor],
    ignore_index: int = IGNORE_INDEX,
    projection_chunk_size: int | None = DEFAULT_SPARSE_CE_PROJECTION_CHUNK_SIZE,
) -> Tensor:
    """Reduce prepared action-token CE by projecting only supervised hidden rows."""
    labels = prepared.labels.to(device=hidden.device)
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be rank-3, got {tuple(hidden.shape)}.")
    if tuple(hidden.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            "hidden prefix dimensions must match prepared labels, got "
            f"hidden={tuple(hidden.shape[:2])}, labels={tuple(labels.shape)}."
        )

    supervised_mask = labels.ne(int(ignore_index))
    if not bool(supervised_mask.any()):
        return zero_action_token_ce_loss(policy, device=hidden.device)

    if projection_chunk_size is None:
        chunk_size = int(supervised_mask.sum().item())
    else:
        chunk_size = int(projection_chunk_size)
        if chunk_size <= 0:
            raise ValueError(
                f"projection_chunk_size must be positive, got {projection_chunk_size}."
            )

    selected_hidden = hidden[supervised_mask]
    selected_labels = labels[supervised_mask].long()
    candidate_token_ids = build_action_token_ce_candidate_token_ids(
        policy,
        device=hidden.device,
    )
    if candidate_token_ids is not None:
        selected_labels = remap_labels_to_candidate_token_indices(
            selected_labels,
            candidate_token_ids=candidate_token_ids,
            ignore_index=ignore_index,
            owner_name=policy.__class__.__name__,
        )
    ce_chunks: list[Tensor] = []
    for start in range(0, int(selected_hidden.shape[0]), chunk_size):
        end = min(start + chunk_size, int(selected_hidden.shape[0]))
        if candidate_token_ids is None:
            logits = project_hidden_to_logits(selected_hidden[start:end])
        else:
            logits = project_hidden_to_logits(
                selected_hidden[start:end],
                candidate_token_ids=candidate_token_ids,
            )
        if logits.ndim != 2:
            raise ValueError(
                "project_hidden_to_logits must return rank-2 logits for sparse CE, "
                f"got {tuple(logits.shape)}."
            )
        if int(logits.shape[0]) != end - start:
            raise ValueError(
                "project_hidden_to_logits returned a mismatched row count: "
                f"expected {end - start}, got {int(logits.shape[0])}."
            )
        if candidate_token_ids is not None and int(logits.shape[-1]) != int(
            candidate_token_ids.numel()
        ):
            raise ValueError(
                "project_hidden_to_logits returned a mismatched candidate vocab size: "
                f"expected {int(candidate_token_ids.numel())}, got {int(logits.shape[-1])}."
            )
        ce_chunks.append(
            F.cross_entropy(
                logits.float(),
                selected_labels[start:end],
                reduction="none",
            )
        )

    selected_ce = torch.cat(ce_chunks, dim=0)
    ce_per_token = torch.zeros(
        labels.shape,
        device=hidden.device,
        dtype=selected_ce.dtype,
    )
    ce_per_token = ce_per_token.masked_scatter(supervised_mask, selected_ce)
    loss = reduce_blockwise_token_ce_loss(
        ce_per_token=ce_per_token,
        supervised_mask=supervised_mask,
        block_schedule=prepared.block_schedule,
        weighting_mode=str(policy.config.block_ce_weighting),
        allow_partial_blocks=prepared.token_lengths is not None,
    )
    if loss is None:
        return zero_action_token_ce_loss(policy, device=hidden.device)
    return loss


def run_action_token_ce_prefix_training_step(
    policy: ActionTokenCePolicy,
    batch: dict[str, Tensor],
    *,
    prefix_adapter: VlmPrefixLogitsAdapter,
    ignore_index: int = IGNORE_INDEX,
    gradient_checkpointing: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Run schedule-driven VLM CE using hidden-state sparse projection."""
    batch, skipped = normalize_action_token_ce_batch(policy, batch)
    if skipped is not None:
        return skipped

    prepared = prepare_action_token_ce(
        policy,
        batch,
        ignore_index=ignore_index,
    )
    if not prepared.labels.ne(int(ignore_index)).any():
        return skipped_all_ignore_result(policy, device=prepared.labels.device)
    result = run_action_token_ce_prefix_forward(
        policy,
        prepared,
        prefix_adapter=prefix_adapter,
        ignore_index=ignore_index,
        gradient_checkpointing=gradient_checkpointing,
    )
    return result.loss, {}


def build_action_token_ce_candidate_token_ids(
    policy: ActionTokenCePolicy,
    *,
    device: torch.device,
) -> Tensor | None:
    """Return candidate-token ids for configured CE vocab scope."""
    scope = validate_ce_vocab_scope(getattr(policy.config, "ce_vocab_scope", "full"))
    if scope == "full":
        return None
    if scope != "action":
        raise AssertionError(f"unreachable CE vocab scope: {scope!r}")

    vocab = policy._require_action_token_vocabulary()
    candidate_ids = torch.tensor(vocab.token_ids, dtype=torch.long, device=device)
    expected = int(candidate_ids.numel())
    if policy._supervise_eos:
        eos_token_id = policy._eos_token_id
        if eos_token_id is None:
            raise ValueError(
                f"{policy.__class__.__name__} cannot use ce_vocab_scope='action' "
                "with EOS supervision because tokenizer.eos_token_id is unset."
            )
        candidate_ids = torch.cat(
            [
                candidate_ids,
                torch.tensor([int(eos_token_id)], dtype=torch.long, device=device),
            ]
        )
        expected += 1
    if int(torch.unique(candidate_ids).numel()) != expected:
        raise ValueError(
            f"{policy.__class__.__name__} action CE candidate ids must be unique, "
            f"got {candidate_ids.detach().cpu().tolist()}."
        )
    return candidate_ids


def remap_labels_to_candidate_token_indices(
    labels: Tensor,
    *,
    candidate_token_ids: Tensor,
    ignore_index: int = IGNORE_INDEX,
    owner_name: str,
) -> Tensor:
    """Map absolute vocab labels into candidate-class indices for scoped CE."""
    candidate_token_ids = candidate_token_ids.to(device=labels.device, dtype=torch.long)
    mapped = torch.full_like(labels.long(), int(ignore_index))
    supervised_mask = labels.ne(int(ignore_index))
    if not bool(supervised_mask.any()):
        return mapped

    if int(candidate_token_ids.numel()) == 0:
        raise ValueError(f"{owner_name} CE candidate-token ids must be non-empty.")

    selected_labels = labels[supervised_mask].long()
    selected_mapped = torch.full_like(selected_labels, int(ignore_index))
    first_candidate_id = int(candidate_token_ids[0].item())
    expected_contiguous_ids = torch.arange(
        first_candidate_id,
        first_candidate_id + int(candidate_token_ids.numel()),
        device=candidate_token_ids.device,
        dtype=torch.long,
    )
    contiguous_prefix_mask = candidate_token_ids.eq(expected_contiguous_ids)
    first_gap = (~contiguous_prefix_mask).nonzero(as_tuple=False)
    contiguous_count = (
        int(candidate_token_ids.numel())
        if int(first_gap.numel()) == 0
        else int(first_gap[0].item())
    )

    if contiguous_count > 0:
        contiguous_label_mask = selected_labels.ge(
            first_candidate_id
        ) & selected_labels.lt(first_candidate_id + contiguous_count)
        selected_mapped[contiguous_label_mask] = (
            selected_labels[contiguous_label_mask] - first_candidate_id
        )

    for candidate_idx in range(contiguous_count, int(candidate_token_ids.numel())):
        token_id = int(candidate_token_ids[candidate_idx].item())
        selected_mapped[selected_labels.eq(token_id)] = candidate_idx

    unmatched = selected_mapped.eq(int(ignore_index))
    if bool(unmatched.any()):
        bad_values = sorted(
            {int(value) for value in selected_labels[unmatched].detach().cpu().tolist()}
        )
        raise ValueError(
            f"{owner_name} CE labels contain token ids outside ce_vocab_scope='action': "
            f"{bad_values}."
        )
    mapped[supervised_mask] = selected_mapped
    return mapped


def _prepare_target_token_ids(
    policy: ActionTokenCePolicy,
    *,
    token_ids_batch: list[list[int]],
    token_lengths: list[int] | None,
    device: torch.device,
) -> tuple[Tensor, list[int] | None, tuple[int, ...]]:
    if token_lengths is None:
        target_token_ids = tensorize_token_rows(token_ids_batch, device=device)
        block_schedule = _resolve_block_schedule(
            policy,
            token_horizon=int(target_token_ids.shape[1]),
        )
        validate_block_schedule_sum(
            block_schedule=block_schedule,
            token_horizon=int(target_token_ids.shape[1]),
        )
        return target_token_ids, None, block_schedule

    if len(token_lengths) != len(token_ids_batch):
        raise ValueError(
            "Per-sample token lengths must match token batch size: "
            f"len(token_lengths)={len(token_lengths)} vs batch={len(token_ids_batch)}."
        )
    token_horizon = int(
        policy._resolve_action_token_horizon(
            token_ids_batch,
            token_lengths=token_lengths,
        )
    )
    block_schedule = _resolve_block_schedule(policy, token_horizon=token_horizon)
    token_horizon = int(sum(block_schedule))
    prepared_targets = prepare_action_token_target_rows(
        token_rows=token_ids_batch,
        token_lengths=token_lengths,
        token_horizon=token_horizon,
        block_schedule=block_schedule,
        device=device,
        supervise_eos=policy._supervise_eos,
        pad_value=0,
    )
    return (
        prepared_targets.target_raw_token_ids,
        prepared_targets.token_lengths,
        block_schedule,
    )


def _validate_raw_action_token_ids(
    token_ids: Tensor,
    *,
    num_tokens: int,
    owner_name: str,
) -> None:
    if token_ids.numel() == 0:
        return
    invalid = token_ids.lt(0) | token_ids.ge(int(num_tokens))
    if not bool(invalid.any()):
        return
    bad_values = sorted({int(value) for value in token_ids[invalid].detach().cpu()})
    raise ValueError(
        f"{owner_name} action tokenizer emitted raw token ids outside "
        f"[0, {int(num_tokens)}): {bad_values}."
    )


def _resolve_block_schedule(
    policy: ActionTokenCePolicy, *, token_horizon: int
) -> tuple[int, ...]:
    values = policy._resolve_action_token_block_schedule(token_horizon=token_horizon)
    return tuple(int(v) for v in values)


def _action_key(policy: ActionTokenCePolicy) -> str:
    if policy.io is not None and policy.io.action_feature is not None:
        return str(policy.io.action_feature.key)
    return "action"


def _action_token_device(
    policy: ActionTokenCePolicy, *, actions: Tensor | None
) -> torch.device:
    if actions is not None:
        return actions.device
    return next(policy.vlm.parameters()).device


def _action_pad_mask_from_batch(
    policy: ActionTokenCePolicy,
    batch: dict[str, Tensor],
    *,
    actions: Tensor | None,
) -> Tensor | None:
    pad_key = f"{_action_key(policy)}_is_pad"
    value = batch.get(pad_key)
    if value is None:
        return None
    if value.ndim != 2:
        raise ValueError(f"{pad_key} must have shape (B, H), got {tuple(value.shape)}.")
    if actions is not None and value.shape != actions.shape[:2]:
        raise ValueError(
            f"{pad_key} shape must match action batch/horizon "
            f"{tuple(actions.shape[:2])}, got {tuple(value.shape)}."
        )
    device = actions.device if actions is not None else value.device
    return value.to(device=device, dtype=torch.bool)
