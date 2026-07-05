# SPDX-FileCopyrightText: 2026 Chaoqi Liu
#
# SPDX-License-Identifier: Apache-2.0

"""Integrated VLM-prefix/action-expert training loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, cast

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from praxis.policies.core.action_expert.expert import (
    ActionExpert,
    masked_action_mse_loss,
)
from praxis.policies.core.action_expert.flow import FlowSample
from praxis.policies.core.action_expert.prefix_cache import PrefixCache
from praxis.policies.core.vlm.prefix import (
    LayerKV,
    VlmAttentionGeometry,
    VlmInputBatch,
    VlmPrefixAdapter as VlmPrefixStreamAdapter,
    VlmPrefixState,
)


class VlmActionExpertPrefixAdapter(VlmPrefixStreamAdapter, Protocol):
    """Backbone-specific VLM prefix stream used by action experts."""

    @property
    def hidden_size(self) -> int: ...

    @property
    def attention_geometry(self) -> VlmAttentionGeometry: ...

    def forward_layer(
        self, state: VlmPrefixState, *, layer_idx: int
    ) -> tuple[VlmPrefixState, LayerKV]: ...

    def project_layer_kv(
        self,
        state: VlmPrefixState,
        *,
        layer_idx: int,
        position_ids: Tensor | None = None,
    ) -> LayerKV: ...


@dataclass(frozen=True)
class IntegratedVlmActionExpertForward:
    """Outputs from one VLM stream plus one action-expert suffix stream."""

    pred_velocity: Tensor
    prefix_state: VlmPrefixState


class VlmActionExpertModel:
    """Run the VLM prefix stream and action-expert suffix stream layer by layer."""

    def __init__(
        self,
        *,
        prefix_adapter: VlmActionExpertPrefixAdapter,
        action_expert: ActionExpert,
    ) -> None:
        self.prefix_adapter = prefix_adapter
        self.action_expert = action_expert

    def __call__(
        self,
        *,
        vlm_inputs: VlmInputBatch,
        noisy_actions: Tensor,
        timesteps: Tensor,
        action_is_pad: Tensor | None = None,
        detach_prefix: bool,
        gradient_checkpointing: bool,
    ) -> Tensor:
        if detach_prefix:
            with torch.no_grad():
                prefix_state = self.prefix_adapter.prepare_prefix(vlm_inputs)
        else:
            prefix_state = self.prefix_adapter.prepare_prefix(vlm_inputs)
        hidden, expert_pad_mask, expert_attention_mask = (
            self.action_expert.prepare_suffix_inputs(
                noisy_actions=noisy_actions,
                timesteps=timesteps,
                action_is_pad=action_is_pad,
            )
        )
        if prefix_state.hidden.shape[0] != hidden.shape[0]:
            raise ValueError(
                "VLM prefix batch size must match expert batch size, got "
                f"{prefix_state.hidden.shape[0]} and {hidden.shape[0]}."
            )

        for layer_idx in range(self.action_expert.vlm_num_layers):
            use_checkpoint = (
                bool(gradient_checkpointing)
                and self.action_expert.training
                and not detach_prefix
            )
            if use_checkpoint:

                def layer_block(
                    prefix_hidden: Tensor,
                    suffix_hidden: Tensor,
                    *,
                    bound_layer_idx: int = layer_idx,
                    bound_prefix_state: VlmPrefixState = prefix_state,
                ) -> tuple[Tensor, Tensor]:
                    return self._checkpointed_layer(
                        prefix_hidden,
                        suffix_hidden,
                        layer_idx=bound_layer_idx,
                        prefix_state=bound_prefix_state,
                        expert_pad_mask=expert_pad_mask,
                        expert_attention_mask=expert_attention_mask,
                    )

                checkpoint_output = checkpoint(
                    layer_block,
                    prefix_state.hidden,
                    hidden,
                    use_reentrant=False,
                )
                prefix_hidden, hidden = cast(tuple[Tensor, Tensor], checkpoint_output)
                prefix_state = replace(prefix_state, hidden=prefix_hidden)
            else:
                prefix_state, hidden = self._forward_layer(
                    prefix_state=prefix_state,
                    hidden=hidden,
                    layer_idx=layer_idx,
                    expert_pad_mask=expert_pad_mask,
                    expert_attention_mask=expert_attention_mask,
                    detach_prefix=detach_prefix,
                )

        return self.action_expert.velocity_from_suffix_hidden(hidden)

    def loss(
        self,
        *,
        vlm_inputs: VlmInputBatch,
        sample: FlowSample,
        action_is_pad: Tensor | None = None,
        detach_prefix: bool,
        gradient_checkpointing: bool,
    ) -> tuple[Tensor, Tensor]:
        pred_velocity = self(
            vlm_inputs=vlm_inputs,
            noisy_actions=sample.noisy_actions,
            timesteps=sample.timesteps,
            action_is_pad=action_is_pad,
            detach_prefix=detach_prefix,
            gradient_checkpointing=gradient_checkpointing,
        )
        loss = masked_action_mse_loss(
            pred_velocity,
            sample.target_velocity,
            action_is_pad=action_is_pad,
        )
        return loss, pred_velocity

    def forward_with_context(
        self,
        *,
        vlm_inputs: VlmInputBatch,
        layer_attention_mask: Tensor | None = None,
        expert_context_mask: Tensor,
        noisy_actions: Tensor,
        timesteps: Tensor,
        action_is_pad: Tensor | None = None,
        detach_expert_context: bool,
        gradient_checkpointing: bool,
    ) -> IntegratedVlmActionExpertForward:
        """Run a VLM stream and an expert stream with selected VLM context.

        The expert receives only positions selected by ``expert_context_mask``.
        The VLM stream itself is still advanced over the full supplied prefix.
        """
        prefix_state = self.prefix_adapter.prepare_prefix(
            vlm_inputs,
            layer_attention_mask=layer_attention_mask,
        )
        context_mask = _validate_expert_context_mask(
            expert_context_mask,
            prefix_state=prefix_state,
        )
        context_offsets = context_position_offsets(
            prefix_state,
            context_mask=context_mask,
        )
        hidden, expert_pad_mask, expert_attention_mask = (
            self.action_expert.prepare_suffix_inputs(
                noisy_actions=noisy_actions,
                timesteps=timesteps,
                action_is_pad=action_is_pad,
            )
        )
        if prefix_state.hidden.shape[0] != hidden.shape[0]:
            raise ValueError(
                "VLM stream batch size must match expert batch size, got "
                f"{prefix_state.hidden.shape[0]} and {hidden.shape[0]}."
            )

        for layer_idx in range(self.action_expert.vlm_num_layers):
            use_checkpoint = (
                bool(gradient_checkpointing)
                and self.action_expert.training
                and torch.is_grad_enabled()
            )
            if use_checkpoint:

                def layer_block(
                    prefix_hidden: Tensor,
                    suffix_hidden: Tensor,
                    *,
                    bound_layer_idx: int = layer_idx,
                    bound_prefix_state: VlmPrefixState = prefix_state,
                ) -> tuple[Tensor, Tensor]:
                    return self._checkpointed_context_layer(
                        prefix_hidden,
                        suffix_hidden,
                        layer_idx=bound_layer_idx,
                        prefix_state=bound_prefix_state,
                        expert_context_mask=context_mask,
                        expert_context_offsets=context_offsets,
                        expert_pad_mask=expert_pad_mask,
                        expert_attention_mask=expert_attention_mask,
                        detach_expert_context=detach_expert_context,
                    )

                checkpoint_output = checkpoint(
                    layer_block,
                    prefix_state.hidden,
                    hidden,
                    use_reentrant=False,
                )
                prefix_hidden, hidden = cast(tuple[Tensor, Tensor], checkpoint_output)
                prefix_state = replace(prefix_state, hidden=prefix_hidden)
            else:
                prefix_state, hidden = self._forward_context_layer(
                    prefix_state=prefix_state,
                    hidden=hidden,
                    layer_idx=layer_idx,
                    expert_context_mask=context_mask,
                    expert_context_offsets=context_offsets,
                    expert_pad_mask=expert_pad_mask,
                    expert_attention_mask=expert_attention_mask,
                    detach_expert_context=detach_expert_context,
                )

        return IntegratedVlmActionExpertForward(
            pred_velocity=self.action_expert.velocity_from_suffix_hidden(hidden),
            prefix_state=prefix_state,
        )

    def loss_with_context(
        self,
        *,
        vlm_inputs: VlmInputBatch,
        layer_attention_mask: Tensor | None = None,
        expert_context_mask: Tensor,
        sample: FlowSample,
        action_is_pad: Tensor | None = None,
        detach_expert_context: bool,
        gradient_checkpointing: bool,
    ) -> tuple[Tensor, IntegratedVlmActionExpertForward]:
        output = self.forward_with_context(
            vlm_inputs=vlm_inputs,
            layer_attention_mask=layer_attention_mask,
            expert_context_mask=expert_context_mask,
            noisy_actions=sample.noisy_actions,
            timesteps=sample.timesteps,
            action_is_pad=action_is_pad,
            detach_expert_context=detach_expert_context,
            gradient_checkpointing=gradient_checkpointing,
        )
        loss = masked_action_mse_loss(
            output.pred_velocity,
            sample.target_velocity,
            action_is_pad=action_is_pad,
        )
        return loss, output

    def prefill_cache(
        self,
        vlm_inputs: VlmInputBatch,
        *,
        detach: bool,
    ) -> PrefixCache:
        """Build an action-expert prefix cache through the prefix adapter path."""
        if detach:
            with torch.no_grad():
                return self._prefill_cache_impl(vlm_inputs, detach=True)
        return self._prefill_cache_impl(vlm_inputs, detach=False)

    def _prefill_cache_impl(
        self,
        vlm_inputs: VlmInputBatch,
        *,
        detach: bool,
    ) -> PrefixCache:
        prefix_state = self.prefix_adapter.prepare_prefix(vlm_inputs)
        layers: list[LayerKV] = []
        for layer_idx in range(self.action_expert.vlm_num_layers):
            prefix_state, layer_kv = self.prefix_adapter.forward_layer(
                prefix_state,
                layer_idx=layer_idx,
            )
            layers.append(layer_kv.detach() if detach else layer_kv)
        mask = prefix_state.mask.detach() if detach else prefix_state.mask
        suffix_offsets = prefix_state.suffix_position_offsets
        if detach and suffix_offsets is not None:
            suffix_offsets = suffix_offsets.detach()
        return PrefixCache(
            layers=tuple(layers),
            mask=mask,
            suffix_position_offsets=suffix_offsets,
        )

    def _checkpointed_layer(
        self,
        prefix_hidden: Tensor,
        hidden: Tensor,
        *,
        layer_idx: int,
        prefix_state: VlmPrefixState,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        next_prefix_state, next_hidden = self._forward_layer(
            prefix_state=replace(prefix_state, hidden=prefix_hidden),
            hidden=hidden,
            layer_idx=layer_idx,
            expert_pad_mask=expert_pad_mask,
            expert_attention_mask=expert_attention_mask,
            detach_prefix=False,
        )
        return next_prefix_state.hidden, next_hidden

    def _checkpointed_context_layer(
        self,
        prefix_hidden: Tensor,
        hidden: Tensor,
        *,
        layer_idx: int,
        prefix_state: VlmPrefixState,
        expert_context_mask: Tensor,
        expert_context_offsets: Tensor,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
        detach_expert_context: bool,
    ) -> tuple[Tensor, Tensor]:
        next_prefix_state, next_hidden = self._forward_context_layer(
            prefix_state=replace(prefix_state, hidden=prefix_hidden),
            hidden=hidden,
            layer_idx=layer_idx,
            expert_context_mask=expert_context_mask,
            expert_context_offsets=expert_context_offsets,
            expert_pad_mask=expert_pad_mask,
            expert_attention_mask=expert_attention_mask,
            detach_expert_context=detach_expert_context,
        )
        return next_prefix_state.hidden, next_hidden

    def _forward_layer(
        self,
        *,
        prefix_state: VlmPrefixState,
        hidden: Tensor,
        layer_idx: int,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
        detach_prefix: bool,
    ) -> tuple[VlmPrefixState, Tensor]:
        if detach_prefix:
            with torch.no_grad():
                next_prefix_state, layer_kv = self.prefix_adapter.forward_layer(
                    prefix_state,
                    layer_idx=layer_idx,
                )
            layer_kv = layer_kv.detach()
            next_prefix_state = replace(
                next_prefix_state,
                hidden=next_prefix_state.hidden.detach(),
            )
        else:
            next_prefix_state, layer_kv = self.prefix_adapter.forward_layer(
                prefix_state,
                layer_idx=layer_idx,
            )
        next_hidden = self.action_expert.forward_suffix_layer(
            hidden=hidden,
            vlm_layer_idx=layer_idx,
            layer_kv=layer_kv,
            prefix_mask=next_prefix_state.mask,
            prefix_position_offsets=next_prefix_state.suffix_position_offsets,
            expert_pad_mask=expert_pad_mask,
            expert_attention_mask=expert_attention_mask,
        )
        return next_prefix_state, next_hidden

    def _forward_context_layer(
        self,
        *,
        prefix_state: VlmPrefixState,
        hidden: Tensor,
        layer_idx: int,
        expert_context_mask: Tensor,
        expert_context_offsets: Tensor,
        expert_pad_mask: Tensor,
        expert_attention_mask: Tensor,
        detach_expert_context: bool,
    ) -> tuple[VlmPrefixState, Tensor]:
        next_prefix_state = self.prefix_adapter.forward_prefix_layer(
            prefix_state,
            layer_idx=layer_idx,
        )
        expert_position_ids = _expert_position_ids(prefix_state)
        layer_kv = self.prefix_adapter.project_layer_kv(
            prefix_state,
            layer_idx=layer_idx,
            position_ids=expert_position_ids,
        )
        layer_kv, compact_context_mask = _compact_layer_kv(
            layer_kv,
            context_mask=expert_context_mask,
            detach=detach_expert_context,
        )
        next_hidden = self.action_expert.forward_suffix_layer(
            hidden=hidden,
            vlm_layer_idx=layer_idx,
            layer_kv=layer_kv,
            prefix_mask=compact_context_mask,
            prefix_position_offsets=expert_context_offsets,
            expert_pad_mask=expert_pad_mask,
            expert_attention_mask=expert_attention_mask,
        )
        return next_prefix_state, next_hidden


def _validate_expert_context_mask(
    expert_context_mask: Tensor,
    *,
    prefix_state: VlmPrefixState,
) -> Tensor:
    mask = expert_context_mask.to(device=prefix_state.hidden.device, dtype=torch.bool)
    if mask.shape != prefix_state.mask.shape:
        raise ValueError(
            "expert_context_mask must have shape (B, S) matching the VLM stream, "
            f"got {tuple(mask.shape)} and {tuple(prefix_state.mask.shape)}."
        )
    mask = mask & prefix_state.mask.to(device=mask.device, dtype=torch.bool)
    if not bool(mask.any(dim=1).all()):
        raise ValueError("Every sample must expose at least one expert context token.")
    return mask


def context_position_offsets(
    prefix_state: VlmPrefixState, *, context_mask: Tensor
) -> Tensor:
    positions = prefix_state.position_ids
    expert_positions = (prefix_state.extra or {}).get("expert_position_ids")
    if isinstance(expert_positions, Tensor):
        positions = expert_positions
    if positions.ndim == 3:
        if positions.shape[1:] != context_mask.shape:
            raise ValueError(
                "rank-3 expert position ids must have shape (N, B, S) matching "
                f"context mask (B, S), got {tuple(positions.shape)} and "
                f"{tuple(context_mask.shape)}."
            )
        positions = positions.to(device=context_mask.device, dtype=torch.long)
        valid_positions = positions.masked_fill(
            ~context_mask[None, :, :],
            torch.iinfo(torch.long).min,
        )
        return valid_positions.amax(dim=(0, 2)) + 1
    if positions.ndim != 2:
        raise ValueError(
            f"position_ids must have shape (B, S) or (N, B, S), got {tuple(positions.shape)}."
        )
    if positions.shape != context_mask.shape:
        raise ValueError(
            "position_ids must match expert context mask shape, got "
            f"{tuple(positions.shape)} and {tuple(context_mask.shape)}."
        )
    positions = positions.to(device=context_mask.device, dtype=torch.long)
    valid_positions = positions.masked_fill(~context_mask, torch.iinfo(torch.long).min)
    return valid_positions.max(dim=1).values + 1


def _expert_position_ids(prefix_state: VlmPrefixState) -> Tensor | None:
    positions = (prefix_state.extra or {}).get("expert_position_ids")
    return positions if isinstance(positions, Tensor) else None


def _compact_layer_kv(
    layer_kv: LayerKV, *, context_mask: Tensor, detach: bool
) -> tuple[LayerKV, Tensor]:
    compact_mask = _compact_context_mask(context_mask)
    return (
        LayerKV(
            key=_compact_cache_tensor(
                layer_kv.key,
                context_mask=context_mask,
                detach=detach,
            ),
            value=_compact_cache_tensor(
                layer_kv.value,
                context_mask=context_mask,
                detach=detach,
            ),
        ),
        compact_mask,
    )


def _compact_context_mask(context_mask: Tensor) -> Tensor:
    mask = context_mask.to(dtype=torch.bool)
    counts = mask.sum(dim=1)
    max_count = int(counts.max().item())
    positions = torch.arange(max_count, device=mask.device)
    return positions.unsqueeze(0).lt(counts.unsqueeze(1))


def _compact_cache_tensor(
    tensor: Tensor, *, context_mask: Tensor, detach: bool
) -> Tensor:
    if detach:
        tensor = tensor.detach()
    mask = context_mask.to(device=tensor.device, dtype=torch.bool)
    counts = mask.sum(dim=1)
    max_count = int(counts.max().item())
    if tensor.ndim == 4:
        if tensor.shape[0] != mask.shape[0] or tensor.shape[2] != mask.shape[1]:
            raise ValueError(
                "Rank-4 VLM cache tensor must have shape (B, H, S, D) matching "
                f"context mask (B, S); got tensor={tuple(tensor.shape)} "
                f"mask={tuple(mask.shape)}."
            )
        out = tensor.new_zeros(
            (
                int(tensor.shape[0]),
                int(tensor.shape[1]),
                max_count,
                int(tensor.shape[3]),
            )
        )
        for batch_idx in range(int(tensor.shape[0])):
            count = int(counts[batch_idx].item())
            if count > 0:
                out[batch_idx, :, :count, :] = tensor[
                    batch_idx,
                    :,
                    mask[batch_idx],
                    :,
                ]
        return out
    if tensor.ndim == 3:
        if tensor.shape[0] != mask.shape[0] or tensor.shape[1] != mask.shape[1]:
            raise ValueError(
                "Rank-3 VLM cache tensor must have shape (B, S, C) matching "
                f"context mask (B, S); got tensor={tuple(tensor.shape)} "
                f"mask={tuple(mask.shape)}."
            )
        out = tensor.new_zeros((int(tensor.shape[0]), max_count, int(tensor.shape[2])))
        for batch_idx in range(int(tensor.shape[0])):
            count = int(counts[batch_idx].item())
            if count > 0:
                out[batch_idx, :count, :] = tensor[batch_idx, mask[batch_idx], :]
        return out
    raise ValueError(
        f"Expected VLM cache tensor rank 3 or 4, got {tuple(tensor.shape)}."
    )
