"""Decode helpers for blockwise VLM action-token policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from praxis.policies.bar.core.decode_info import resolve_decode_sampling
from praxis.policies.bar.core.decode_schedule import (
    build_step_attention_mask,
    plan_generation_schedule,
)
from praxis.policies.bar.core.sampling import sample_next_token_ids
from praxis.policies.bar.core.vlm.decode_tokens import (
    build_allowed_decode_token_ids,
    build_prediction_inputs,
    raw_token_ids_to_vlm_tensor,
)
from praxis.policies.core.observations import extract_instructions
from praxis.policies.core.vlm.schedule import (
    PrefixAttentionMode,
    build_attention_mask,
)


@dataclass(frozen=True)
class GeneratedTokenIds:
    """Generated action-token ids plus EOS termination metadata."""

    token_ids: list[list[int]]
    stopped_on_eos: bool | None


@dataclass(frozen=True)
class DecodedActionChunk:
    """Detokenized action chunk plus the token ids that produced it."""

    actions: Tensor
    token_ids: list[list[int]]
    stopped_on_eos: bool | None


def build_vlm_prefill_inputs(
    *,
    vlm,
    base_vlm_inputs: dict[str, Tensor],
    prompt_input_ids: Tensor,
    running_attention_mask_2d: Tensor,
    pred_block_vlm_ids: Tensor,
    block_schedule: tuple[int, ...],
    mask_dtype: torch.dtype,
    prefix_attention: PrefixAttentionMode,
    extra_prepare_keys: tuple[str, ...] = (),
) -> dict[str, Tensor]:
    """Build BAR cached prefill inputs for one blockwise decode step."""
    full_input_ids = torch.cat([prompt_input_ids, pred_block_vlm_ids], dim=1)
    cache_position = torch.arange(
        full_input_ids.shape[1],
        dtype=torch.long,
        device=full_input_ids.device,
    )
    prefill_kwargs = {
        "input_ids": full_input_ids,
        "attention_mask": running_attention_mask_2d,
        "cache_position": cache_position,
        "use_cache": True,
        "logits_to_keep": int(pred_block_vlm_ids.shape[1]),
    }
    for key in extra_prepare_keys:
        prefill_kwargs[key] = base_vlm_inputs.get(key)
    model_inputs = vlm.prepare_inputs_for_generation(**prefill_kwargs)
    model_inputs["attention_mask"] = build_attention_mask(
        running_attention_mask_2d,
        block_schedule,
        dtype=mask_dtype,
        prefix_attention=prefix_attention,
    )
    return model_inputs


def build_vlm_cached_step_inputs(
    *,
    vlm,
    past_key_values: Any,
    running_attention_mask_2d: Tensor,
    pred_block_vlm_ids: Tensor,
    block_schedule: tuple[int, ...],
    mask_dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Build BAR cached incremental inputs for later blockwise decode steps."""
    block_size = int(pred_block_vlm_ids.shape[1])
    total_len = int(running_attention_mask_2d.shape[1])
    cache_position = torch.arange(
        total_len - block_size,
        total_len,
        dtype=torch.long,
        device=pred_block_vlm_ids.device,
    )
    model_inputs = vlm.prepare_inputs_for_generation(
        input_ids=pred_block_vlm_ids,
        past_key_values=past_key_values,
        attention_mask=running_attention_mask_2d,
        cache_position=cache_position,
        logits_to_keep=block_size,
        use_cache=True,
    )
    model_inputs["attention_mask"] = build_step_attention_mask(
        running_attention_mask_2d,
        block_schedule,
        dtype=mask_dtype,
    )
    return model_inputs


@dataclass(frozen=True)
class BlockwiseVlmStepResult:
    """Outputs from one BAR cached blockwise VLM decode step."""

    logits: Tensor
    running_attention_mask_2d: Tensor
    running_input_ids: Tensor
    past_key_values: Any | None
    outputs: Any


def run_blockwise_vlm_forward_step_result(
    policy,
    *,
    base_vlm_inputs: dict[str, Tensor],
    running_input_ids: Tensor,
    running_attention_mask_2d: Tensor,
    pred_block_vlm_ids: Tensor,
    block_schedule: tuple[int, ...],
    mask_dtype: torch.dtype,
    past_key_values: Any | None,
    extra_forward_kwargs: dict[str, Any] | None = None,
) -> BlockwiseVlmStepResult:
    """Append one BAR block-input step and return the full VLM step result."""
    block_size = int(pred_block_vlm_ids.shape[1])
    batch_size = int(running_input_ids.shape[0])
    updated_attention_mask = torch.cat(
        [
            running_attention_mask_2d,
            torch.ones(
                (batch_size, block_size),
                dtype=running_attention_mask_2d.dtype,
                device=running_attention_mask_2d.device,
            ),
        ],
        dim=1,
    )
    updated_input_ids = torch.cat([running_input_ids, pred_block_vlm_ids], dim=1)

    if past_key_values is None:
        model_inputs = build_vlm_prefill_inputs(
            vlm=policy.vlm,
            base_vlm_inputs=base_vlm_inputs,
            prompt_input_ids=running_input_ids,
            running_attention_mask_2d=updated_attention_mask,
            pred_block_vlm_ids=pred_block_vlm_ids,
            block_schedule=block_schedule,
            mask_dtype=mask_dtype,
            prefix_attention=policy._require_prefix_attention(),
            extra_prepare_keys=getattr(policy, "_PREFILL_EXTRA_KEYS", ()),
        )
    else:
        model_inputs = build_vlm_cached_step_inputs(
            vlm=policy.vlm,
            past_key_values=past_key_values,
            running_attention_mask_2d=updated_attention_mask,
            pred_block_vlm_ids=pred_block_vlm_ids,
            block_schedule=block_schedule,
            mask_dtype=mask_dtype,
        )
    if extra_forward_kwargs:
        model_inputs.update(extra_forward_kwargs)

    with policy._autocast_context():
        outputs = policy.vlm(**model_inputs, return_dict=True)

    logits = outputs.logits.float()
    if logits.ndim != 3:
        raise RuntimeError(
            f"Expected blockwise decode logits rank 3 (B,L,V), got shape {tuple(logits.shape)}."
        )
    return BlockwiseVlmStepResult(
        logits=logits[:, -block_size:, :],
        running_attention_mask_2d=updated_attention_mask,
        running_input_ids=updated_input_ids,
        past_key_values=outputs.past_key_values,
        outputs=outputs,
    )


def run_blockwise_token_decode(
    policy,
    obs: dict[str, Tensor],
    *,
    normalize_obs: bool,
    keep_k: int,
    decode_temperature: float | None = None,
    decode_top_k: int | None = None,
    eos_supervision: bool | None = None,
) -> GeneratedTokenIds:
    """Generate raw action-token ids with the shared blockwise decode mechanics."""
    resolved_keep_k = int(keep_k)
    eos_supervision = (
        bool(getattr(policy, "_supervise_eos", False))
        if eos_supervision is None
        else bool(eos_supervision)
    )
    block_schedule = policy._resolve_block_schedule()
    decode_keep_k = resolved_keep_k
    if eos_supervision:
        decode_keep_k = min(int(sum(block_schedule)), resolved_keep_k + 1)
    generation_schedule = plan_generation_schedule(
        block_schedule,
        keep_k=decode_keep_k,
    )

    obs = policy._prepare_chunk_observations(obs, normalize=normalize_obs)
    instructions = extract_instructions(
        obs,
        language_key=None
        if policy.io is None
        else getattr(policy.io, "language_key", None),
    )
    policy_config = getattr(policy, "config", None)
    resolved_temperature, resolved_top_k = resolve_decode_sampling(
        default_temperature=getattr(policy_config, "decode_temperature", None),
        default_top_k=getattr(policy_config, "decode_top_k", None),
        override_temperature=decode_temperature,
        override_top_k=decode_top_k,
    )
    base_vlm_inputs = policy._build_vlm_inputs(obs, instructions=instructions)
    prompt_input_ids = base_vlm_inputs["input_ids"]
    running_attention_mask_2d = base_vlm_inputs["attention_mask"]
    batch_size = int(prompt_input_ids.shape[0])
    device = prompt_input_ids.device
    action_vocab = policy._require_action_token_vocabulary()
    mask_dtype = next(policy.vlm.parameters()).dtype
    allowed_token_ids = torch.as_tensor(
        build_allowed_decode_token_ids(
            vocab=action_vocab,
            eos_token_id=(
                int(policy._eos_token_id)
                if eos_supervision and policy._eos_token_id is not None
                else None
            ),
        ),
        dtype=torch.long,
        device=device,
    )
    eos_token_id: int | None = None
    if eos_supervision:
        policy_eos_id = getattr(policy, "_eos_token_id", None)
        if policy_eos_id is None:
            raise RuntimeError("EOS-supervised decode requires a valid EOS token id.")
        eos_token_id = int(policy_eos_id)
    generated_blocks: list[Tensor] = []
    generated_token_lists: list[list[int]] | None = (
        [[] for _ in range(batch_size)] if eos_supervision else None
    )
    prev_block_ids: Tensor | None = None
    past_key_values = None
    finished = (
        torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_supervision
        else None
    )
    stopped_on_eos = (
        torch.zeros(batch_size, dtype=torch.bool, device=device)
        if eos_supervision
        else None
    )
    if policy._mask_sentinel is None:
        raise RuntimeError("Mask sentinel must be initialized before blockwise decode.")
    mask_sentinel = int(policy._mask_sentinel)

    for step, block_size in enumerate(generation_schedule):
        pred_block_ids = build_prediction_inputs(
            batch_size=batch_size,
            block_size=block_size,
            device=device,
            prev_block_ids=prev_block_ids,
            mask_sentinel=mask_sentinel,
        )
        pred_block_vlm_ids = raw_token_ids_to_vlm_tensor(
            pred_block_ids,
            device=device,
            mask_sentinel=mask_sentinel,
            vocab=action_vocab,
        )
        current_schedule = generation_schedule[: step + 1]
        result = run_blockwise_vlm_forward_step_result(
            policy,
            base_vlm_inputs=base_vlm_inputs,
            running_input_ids=prompt_input_ids,
            running_attention_mask_2d=running_attention_mask_2d,
            pred_block_vlm_ids=pred_block_vlm_ids,
            block_schedule=current_schedule,
            mask_dtype=mask_dtype,
            past_key_values=past_key_values,
            extra_forward_kwargs=None,
        )
        logits = result.logits
        running_attention_mask_2d = result.running_attention_mask_2d
        prompt_input_ids = result.running_input_ids
        past_key_values = result.past_key_values
        if past_key_values is None:
            raise RuntimeError(
                "Cached decode requires model outputs with past_key_values."
            )
        sample_logits = logits
        if eos_token_id is not None:
            sample_logits = logits.clone()
            # EOS marks a boundary after at least one decoded block.
            if step == 0:
                sample_logits[:, :, eos_token_id] = float("-inf")
            elif block_size > 1:
                sample_logits[:, 1:, eos_token_id] = float("-inf")
        sampled_vlm_ids = sample_next_token_ids(
            sample_logits.reshape(-1, sample_logits.shape[-1]),
            temperature=resolved_temperature,
            top_k=resolved_top_k,
            allowed_token_ids=allowed_token_ids,
        ).view(batch_size, block_size)

        if eos_token_id is not None:
            assert finished is not None
            assert generated_token_lists is not None
            assert stopped_on_eos is not None
            sampled_vlm_ids = sampled_vlm_ids.clone()
            sampled_vlm_ids[finished] = int(eos_token_id)

            eos_hit = sampled_vlm_ids[:, 0].eq(int(eos_token_id))
            active_rows = (~finished) & (~eos_hit)
            stopped_on_eos = stopped_on_eos | ((~finished) & eos_hit)

            safe_vlm_ids = sampled_vlm_ids.clone()
            safe_vlm_ids[eos_hit] = action_vocab.token_min
            sampled_raw_ids = action_vocab.vlm_ids_to_raw_ids(safe_vlm_ids)

            active_idx = torch.nonzero(active_rows, as_tuple=False).flatten().tolist()
            for row in active_idx:
                remaining = max(
                    0,
                    int(resolved_keep_k) - len(generated_token_lists[row]),
                )
                if remaining > 0:
                    generated_token_lists[row].extend(
                        sampled_raw_ids[row, :remaining].detach().cpu().tolist()
                    )

            finished = finished | eos_hit
            prev_block_ids = sampled_raw_ids
            if bool(finished.all()):
                break
        else:
            sampled_raw_ids = action_vocab.vlm_ids_to_raw_ids(sampled_vlm_ids)
            prev_block_ids = sampled_raw_ids
            generated_blocks.append(sampled_raw_ids)
            covered = sum(generation_schedule[: step + 1])
            if int(covered) >= int(resolved_keep_k):
                break

    if eos_supervision:
        generated_ids = [[int(v) for v in ids] for ids in (generated_token_lists or [])]
        stopped = (
            bool(stopped_on_eos.any().item()) if stopped_on_eos is not None else False
        )
    else:
        generated_tensor = (
            torch.cat(generated_blocks, dim=1)
            if generated_blocks
            else torch.empty((batch_size, 0), dtype=torch.long, device=device)
        )
        token_rows = generated_tensor.detach().cpu().tolist()
        generated_ids = [
            [int(token_id) for token_id in row[:resolved_keep_k]] for row in token_rows
        ]
        stopped = None
    return GeneratedTokenIds(
        token_ids=generated_ids,
        stopped_on_eos=stopped,
    )


def run_blockwise_decode(
    policy,
    obs: dict[str, Tensor],
    *,
    normalize_obs: bool,
    decode_kwargs: dict,
) -> DecodedActionChunk:
    """Shared schedule-driven chunk decode for VLM-backed fixed-length policies."""
    keep_k = decode_kwargs.pop("decode_keep_k", None)
    if keep_k is not None and not isinstance(keep_k, int):
        raise TypeError(
            f"decode_keep_k must be an int or None, got {type(keep_k).__name__}."
        )
    resolved_keep_k = policy._resolve_decode_keep_k(keep_k)

    decode_temperature = decode_kwargs.pop("decode_temperature", None)
    decode_top_k = decode_kwargs.pop("decode_top_k", None)
    if decode_kwargs:
        unexpected = ", ".join(sorted(str(key) for key in decode_kwargs))
        raise TypeError(f"Unsupported VLM decode kwargs: {unexpected}.")

    decoded = run_blockwise_token_decode(
        policy,
        obs,
        normalize_obs=normalize_obs,
        keep_k=resolved_keep_k,
        decode_temperature=decode_temperature,
        decode_top_k=decode_top_k,
    )
    actions = policy._decode_generated_action_tokens(
        decoded.token_ids,
        device=next(policy.parameters()).device,
        horizon=resolved_keep_k,
    )
    return DecodedActionChunk(
        actions=actions,
        token_ids=decoded.token_ids,
        stopped_on_eos=decoded.stopped_on_eos,
    )
